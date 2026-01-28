import os
import pickle
from datetime import datetime
from tqdm import tqdm

import d4rl
import gym
import pathlib
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.env import pusht
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.pusht_dataset import PushTStateDataset, PushTKeypointDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from utils import set_seed, Logger
from env_utils import create_pusht_zarr, make_info_concat_pusht, PushTPlannerDataset, get_pusht_goal_keypoint, get_pusht_goal_pose
from forward_model import ForwardMLP, ResidualBlock

OBS_KP_GOAL = get_pusht_goal_keypoint(do_normalize=True).reshape(-1)

def generate_random_hex_string(size=5):
    hex_array = np.array([i for i in "0123456789abcdef"])
    return "".join(list(np.random.choice(hex_array, size=size)))

def make_env(args, idx, params={}):
    def thunk():
        damping = params["damping"] if "damping" in params else None
        #gravity = params["gravity"] if "gravity" in params else 0.0
        env = gym.make(args.env_name, damping=damping)  
        video_recorder = VideoRecorder.create_h264(
                            fps=10,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=22,
                            thread_type='FRAME',
                            thread_count=1
                        )
        env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=1)
        # Return max of (obs_steps and action_steps) for data generation
        env_step = max([args.obs_steps, args.action_steps])
        env = MultiStepWrapper(env, n_obs_steps=env_step, n_action_steps=env_step, max_episode_steps=args.max_episode_steps)
        #env = MultiStepWrapper(env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps)
        env.seed(args.seed+idx)
        print("Env seed: ", args.seed+idx)
        return env

    return thunk

def inference(args, envs, dataset, agent, invdyn, logger, current_step, params={}, return_data=False):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    finetune_data = []
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_coverages = []
    obs_dim, act_dim = args.obs_dim, args.act_dim
    damping = params["damping"] if "damping" in params else None
    gravity = params["gravity"] if "gravity" in params else None

        
    for i in range(args.eval_episodes // args.num_envs):
        if args.env_name == "pusht-v0":
            this_finetune_data = {
                "action": [],
                "n_contacts": [],
                "state": [],
            }
        elif args.env_name == "pusht-keypoints-v0":
            this_finetune_data = {
                "keypoint": [],
                "action": [],
                "n_contacts": [],
                "state": [],
            } 
        coverage_area_list = []
        ep_reward = [0.0] * args.num_envs
        step_reward = []
        obs, t = envs.reset(), 0
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
        warm_start = args.warm_start

        # initialize video stream
        if args.save_video:
            if damping:
                logger.video_init(envs.envs[0], enable=True, video_id=f"{current_step}_{i}_damping_{damping}")  # save videos
            else:
                logger.video_init(envs.envs[0], enable=True, video_id=f"{current_step}_{i}")  # save videos

        while t < args.max_episode_steps:
            if args.env_name == 'pusht-v0':
                obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                nobs = dataset.normalizer['obs']['state'].normalize(obs_seq)
                nobs = nobs[:, -args.obs_steps:, :] # (num_envs, obs_steps, obs_dim)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            elif args.env_name == 'pusht-keypoints-v0':
                # Note: keypoint env return 40 dim but need 20 dim
                #obs_seq = obs[:, :args.obs_steps, :].astype(np.float32)  # (num_envs, obs_steps, obs_dim) 
                obs_seq = obs[:, -args.obs_steps:, :].astype(np.float32) # (num_envs, obs_steps, obs_dim) take the last ones
                keypoint_obs_seq = obs_seq[:, :, :18]  # (num_envs, obs_steps, 18)
                agentpos_obs_seq = obs_seq[:, :, 18:20]  # (num_envs, obs_steps, 2)
                # normalize obs
                # keypoint
                keypoint_pos_dim = 18
                keypoint = keypoint_obs_seq.reshape(-1, 2)  # (num_envs*obs_steps*9, 2)
                nkeypoint = dataset.normalizer['obs']['keypoint'].normalize(keypoint)  # (num_envs*obs_staps*9, 2)
                nkeypoint = nkeypoint.reshape(args.num_envs, args.obs_steps, keypoint_pos_dim)  # (num_envs, obs_staps, 18)
                # agent_pos
                nagent_pos = dataset.normalizer['obs']['agent_pos'].normalize(agentpos_obs_seq)  # (num_envs, obs_steps, 2)
                nobs = np.concatenate((nkeypoint, nagent_pos), axis=-1)  # (num_envs, obs_steps, obs_dim)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            # sample trajectories (use no guidance for now)
            last_obs = nobs[:,-1,:]
            prior[:, :args.obs_steps] = nobs
            traj, log = agent.sample(
                prior, solver=args.solver,
                n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                condition_cfg=None, w_cfg=0.0, temperature=args.temperature)
            act_start, act_end = args.obs_steps-1, args.obs_steps-1 + args.action_steps
            with torch.no_grad():
                #naction = invdyn.predict(last_obs, traj[:, 1, :]).cpu().numpy()
                naction = invdyn.predict(traj[:, act_start:act_end, :], traj[:, act_start+1:act_end+1, :]).cpu().numpy()
                
            # unnormalize prediction
            #naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)
            # act_num_envs, act_horizon = np.shape(action_pred)
            # action_pred = np.reshape(action_pred, (act_num_envs, 1, act_horizon))
            
            # # get action
            # start = args.obs_steps - 1
            # end = start + args.action_steps
            # action = action_pred[:, start:end, :]
            
            obs, reward, done, info = envs.step(action_pred)
            info_concat = make_info_concat_pusht(info)
            ep_reward += reward
            step_reward.append(reward)
            t += args.action_steps
            coverage_area_list.append(info_concat["coverage"])
            
            # Add data to `finetune_data`
            this_finetune_data["action"].append(np.squeeze(action_pred))
            num_env, horizon, _ = obs.shape
            
            if args.env_name == "pusht-keypoints-v0":
                kp_obs = obs[:,:,:18]
                kp_obs = kp_obs.reshape((num_env, horizon, 9, 2))
                this_finetune_data["keypoint"].append(np.squeeze(kp_obs))

            this_finetune_data["n_contacts"].append(np.expand_dims(info_concat["n_contacts"],  axis=-1))
            this_finetune_data["state"].append(np.concatenate(
                [info_concat["pos_agent"], info_concat["block_pose"]], axis=1
                )
            )

            if done:
                break
        this_finetune_data = {key: np.concatenate(val) for key, val in this_finetune_data.items()}
        # Align obs with act (current obs are next_obs)
        this_finetune_data_aligned = {
            "action": this_finetune_data["action"][1:],
            "n_contacts": this_finetune_data["n_contacts"][:-1],
            "state": this_finetune_data["state"][:-1],
        }
        if args.env_name == "pusht-keypoints-v0":
            this_finetune_data_aligned["keypoint"] = this_finetune_data["keypoint"][:-1]
        finetune_data.append(this_finetune_data_aligned)
        ep_reward = np.around(np.array(ep_reward), 2)
        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        ep_coverage = np.around(np.max(np.concatenate(coverage_area_list), axis=0), 2)
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {ep_reward} success:{success}")
        episode_rewards.append(ep_reward.item())
        episode_steps.append(t)
        episode_success.append(success.item())
        episode_coverages.append(ep_coverage)
    success_rate = np.nanmean(np.where(np.array(episode_success) == 1.0, 1.0, 0.0))
    mean_coverage = np.nanmean(np.array(episode_coverages))
    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)} Success rate: {success_rate} Mean coverage: {mean_coverage}")
    res = {"step": episode_steps, "reward": episode_rewards, "success": episode_success, "coverage": episode_coverages,
            "summary": {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success), 'success_rate': success_rate, 'mean_coverage': mean_coverage}}
    if not return_data:
        return res
    else:
        return res, finetune_data
    
def compute_reward(obs_kp, obs_kp_goal, rew_weight):
    """
    Compute the difference between ||obs_kp_goal - obs_kp_start|| - ||obs_kp_goal - obs_kp_end||
    """
    reward = 0
    num_step, obs_dim = obs_kp.shape
    for i in range(1, num_step):
        obs_kp_t = obs_kp[i]
        obs_kp_tm1 = obs_kp[i-1]
        coeff_t = rew_weight[i-1]
        reward += coeff_t *((np.linalg.norm(obs_kp_goal - obs_kp_tm1) ** 2) - (np.linalg.norm(obs_kp_goal - obs_kp_t) ** 2))
    return reward

@torch.no_grad
def forward_model_rollout(model, obs_start, action_seq, args):
    """
    Return the rollout trajectory using the forward model
    """
    model.eval()
    obs_preds = obs_start # (sample_k, obs_step, obs_dim)
    # action_seq is (sample_k, horizon, act_dim)
    for step in range(args.obs_steps, args.task.horizon):
        this_input = torch.cat([
            obs_preds[:, -2, :],
            action_seq[:, step-2, :],
            obs_preds[:, -1, :],
            action_seq[:, step-1, :],
        ], dim=1)
        out_1, out_2 = model(this_input)
        out = torch.cat([out_1, out_2], dim=1).unsqueeze(1) # (k_sample, 1, obs_dim)
        obs_preds = torch.cat([obs_preds, out], dim=1)
    return obs_preds

def create_planner_finetune_dataset(args, dataloader, normalizer, 
    agent, invdyn, forward_model, obs_goal, k_sample=10, best_k=2):
    """ Similar to `inference` but have 2 difference
    1. Use the forward model to rollout instead of the actual environment
    2. Sample k trajectory instead of 1 trajectory from the planner, rollout and choose the best one
    (may consider filter out bad episode)
    """
    # ---------------- Start Rollout ----------------
    obs_dim, act_dim = args.obs_dim, args.act_dim
    finetune_traj = []
    reward_traj = []
    plan_traj = []
    count = 0
    total = len(dataloader)
    for batch in tqdm(dataloader):
        # Extract obs and act from batch dataloader
        obs_keypoint = batch["obs"]["keypoint"].to(args.device)
        obs_agent_pos = batch["obs"]["agent_pos"].to(args.device)
        obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
        act = batch["action"].to(args.device)
        batch_size = obs.size()[0]
        # Have a k sample loop here
        # Use agent.sample() and invdyn.predict() to get obs_hat, act_hat
        for i in range(batch_size):

            prior = torch.zeros((k_sample, args.task.horizon, obs_dim), device=args.device)
            this_obs, this_act = obs[i], act[i]
            obs_repeat = this_obs.unsqueeze(0).repeat(k_sample, 1, 1)
            act_repeat = this_act.unsqueeze(0).repeat(k_sample, 1, 1)
            prior[:, :args.obs_steps] = obs_repeat[:, :args.obs_steps] # (k_sample, obs_step, obs_dim)
            traj, _ = agent.sample(
                prior, solver=args.solver,
                n_samples=k_sample, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                condition_cfg=None, w_cfg=0.0, temperature=args.temperature
            )
            plan_traj.append(traj[:best_k].cpu().numpy())
            #act_start, act_end = args.obs_steps - 1, args.obs_steps-1 + args.action_steps
            act_start, act_end = args.obs_steps, args.task.horizon
            with torch.no_grad():
                naction = invdyn.predict(
                    traj[:, act_start - 1 : act_end - 1, :],
                    traj[:, act_start : act_end, :]
                )
            #action_pred = normalizer['action'].unnormalize(naction)
            action_pred = torch.cat([
                act_repeat[:, :args.obs_steps], naction
            ], dim=1) # (k_sample, horizon, act_dim)
            traj_fm = forward_model_rollout(forward_model, obs_repeat[:, :args.obs_steps], action_pred, args)
            traj_fm = traj_fm.cpu().numpy()
            rew_mask = np.ones((args.task.horizon - args.obs_steps))
            rew_mask[:args.action_steps] = args.next_obs_loss_weight # weight the action horizon higher
            if args.env_name == "pusht-keypoints-v0":
                traj_fm_kp = traj_fm[:, args.obs_steps-1:, :18]
            elif args.env_name == "pusht-v0":
                traj_fm_kp = traj_fm[:, args.obs_steps-1:, 2:]

            traj_rew = np.array([compute_reward(i, OBS_KP_GOAL, rew_mask) for i in traj_fm_kp])
            traj_fm_best_k_idx = np.argsort(traj_rew)[-best_k:]
            traj_rew_best_k = traj_rew[traj_fm_best_k_idx]
            traj_fm_best_k = traj_fm[traj_fm_best_k_idx, :]
            #traj_fm_best_k_append = np.concatenate([obs_repeat[:best_k, :args.obs_steps], traj_fm_best_k], axis=1) # concat along horizon
            finetune_traj.append(traj_fm_best_k)
            reward_traj.append(traj_rew_best_k)
        #break
            
        count += 1
    finetune_traj_data = np.concatenate(finetune_traj, axis=0) # (num_sample, horizon, obs_dim)
    reward_traj_data = np.concatenate(reward_traj, axis=0)
    plan_traj_data = np.concatenate(plan_traj, axis=0)
    return {"obs": finetune_traj_data,
            "rew": reward_traj_data,
            "plan_traj": plan_traj_data}

# def format_data_for_forward_model(obs_keypoint, obs_agent_pos, act):
#     # obs_keypoint: (batch_size, time, obs_dim)
#     obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
#     obs_tm1 = obs[:, :-2]
#     act_tm1 = act[:, :-2]
#     obs_t = obs[:, 1:-1]
#     act_t = act[:, 1:-1]
#     obs_tp1 = obs[:, 2:]
#     x = torch.cat([obs_tm1, act_tm1, obs_t, act_t], dim=-1)
#     x_flat = torch.flatten(x, start_dim=0, end_dim=1)
#     y_flat = torch.flatten(obs_tp1, start_dim=0, end_dim=1)
#     keypoint_dims = obs_keypoint.size()[2]
#     y_flat_1 = y_flat[:, :keypoint_dims]
#     y_flat_2 = y_flat[:, keypoint_dims:]
#     return x_flat, y_flat_1, y_flat_2

def format_data_for_forward_model(obs, act):
    if "keypoint" in obs.keys():
        obs_keypoint, obs_agent_pos = obs["keypoint"], obs["agent_pos"]
        obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
        obs_tm1 = obs[:, :-2]
        act_tm1 = act[:, :-2]
        obs_t = obs[:, 1:-1]
        act_t = act[:, 1:-1]
        obs_tp1 = obs[:, 2:]
        x = torch.cat([obs_tm1, act_tm1, obs_t, act_t], dim=-1)
        x_flat = torch.flatten(x, start_dim=0, end_dim=1)
        y_flat = torch.flatten(obs_tp1, start_dim=0, end_dim=1)
        keypoint_dims = obs_keypoint.size()[2]
        y_flat_1 = y_flat[:, :keypoint_dims]
        y_flat_2 = y_flat[:, keypoint_dims:]
    elif "state" in obs.keys():
        obs = obs["state"]
        obs_tm1 = obs[:, :-2]
        act_tm1 = act[:, :-2]
        obs_t = obs[:, 1:-1]
        act_t = act[:, 1:-1]
        obs_tp1 = obs[:, 2:]
        x = torch.cat([obs_tm1, act_tm1, obs_t, act_t], dim=-1)
        x_flat = torch.flatten(x, start_dim=0, end_dim=1)
        y_flat = torch.flatten(obs_tp1, start_dim=0, end_dim=1)
        agent_dims = 2
        y_flat_1 = y_flat[:, :agent_dims]
        y_flat_2 = y_flat[:, agent_dims:]
    return x_flat, y_flat_1, y_flat_2

@hydra.main(config_path="../configs/dd/pusht", config_name="pusht_state", version_base=None)
def pipeline(args):
    
    #return_scale = DD_RETURN_SCALE[args.task.env_name]
    set_seed(args.seed)

    if args.save_model_folder == "placeholder":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{timestamp}/'
    else:
        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{args.save_model_folder}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, idx) for idx in range(args.num_envs)],
    )

    # ---------------- Create Dataset ----------------
    # TODO: 2. (Optional) Also add `terminal_penalty`, `discount` (not need for now)
    # Can just ignore `reward-to-go` for now and train an unconditional diffusion planner (push-t dataset already have expert demonstration)
    dataset_path = os.path.expanduser(args.dataset_path)
    if args.env_name == 'pusht-v0':
        dataset = PushTStateDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    elif args.env_name == 'pusht-keypoints-v0':
        dataset = PushTKeypointDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    # TODO: Check whether obs_dim include both `keypoint` + `agent_pos`
    obs_dim, act_dim = args.obs_dim, args.act_dim
    obs_goal = get_pusht_goal_pose(dataset)

    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        obs_dim, emb_dim=args.emb_dim,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    #fix_mask[0] = 1.
    fix_mask[:args.obs_steps] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    #loss_weight[1] = args.next_obs_loss_weight
    #loss_weight[:args.action_steps] = args.next_obs_loss_weight
    loss_weight[args.obs_steps:args.obs_steps+args.action_steps] = args.next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    agent = ContinuousDiffusionSDE(
        nn_diffusion,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic -------------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
    fm_input_dims = (obs_dim + act_dim)*2
    if args.env_name == "pusht-keypoints-v0":
        fm_output_dims = [obs_dim - act_dim, act_dim] # keypoints, agent_pos
    elif args.env_name == "pusht-v0":
        fm_output_dims = [act_dim, obs_dim - act_dim] # agent_pos, block_pose
    forward_model = ForwardMLP(fm_input_dims, fm_output_dims, ResidualBlock)
    forward_model.to(args.device)
    fm_optimizer = torch.optim.Adam(forward_model.parameters(), lr=5e-4)
    fm_criterion = torch.nn.MSELoss()

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # -------------- Add wandb logging ------------------
        # import wandb
        # from omegaconf import OmegaConf
        # from datetime import datetime

        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # cfg_dict = OmegaConf.to_container(args, resolve=True)
        # run = wandb.init(
        #     #entity="hangle-harvard-univeristy",
        #     project=args.task.env_name,
        #     name=f"{args.task.env_name}_{timestamp}",
        #     config=cfg_dict,
        # )

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
        fm_lr_scheduler = CosineAnnealingLR(fm_optimizer, args.fm_gradient_steps)

        agent.train()
        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.,  "avg_loss_invdyn": 0., "avg_loss_fm": 0.}

        for batch in loop_dataloader(dataloader):
            # TODO: Need to do normalized pre-processing and concatenate `keypoints` and `agent_pos`
            if args.env_name == "pusht-keypoints-v0":
                obs_keypoint = batch["obs"]["keypoint"].to(args.device)
                obs_agent_pos = batch["obs"]["agent_pos"].to(args.device)
                obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
                obs_dict = {"keypoint": obs_keypoint, "agent_pos": obs_agent_pos}
            elif args.env_name == "pusht-v0":
                obs = batch["obs"]["state"].to(args.device)
                obs_dict = {"state": obs}
            act = batch["action"].to(args.device)
            #val = batch["val"].to(args.device) / return_scale

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step < args.fm_gradient_steps:
                fm_optimizer.zero_grad()
                x_fm, y_fm_1, y_fm_2 = format_data_for_forward_model(obs_dict, act)
                out_1, out_2 = forward_model(x_fm)
                loss_fm_1 = fm_criterion(out_1, y_fm_1)
                loss_fm_2 = fm_criterion(out_2, y_fm_2)
                loss_fm = loss_fm_1 + loss_fm_2
                loss_fm.backward()
                fm_optimizer.step()
                fm_lr_scheduler.step()
                log["avg_loss_fm"] += loss_fm.item()

            if n_gradient_step <= args.invdyn_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["step"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                log["avg_loss_fm"] /= args.log_interval
                print(log)
                logger.log(log, category="train")
                log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0., "avg_loss_fm": 0.}

            # ----------- Eval ------------
            if (n_gradient_step + 1) % args.eval_interval == 0:
                agent.eval()
                invdyn.eval()
                eval_log = inference(args, envs, dataset, agent, invdyn, logger, n_gradient_step + 1)
                eval_log_summary = eval_log["summary"]
                eval_log_summary["step"] = n_gradient_step + 1
                print(eval_log)
                logger.log(eval_log_summary, category="inference")

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.fm_save_interval == 0:
                torch.save(forward_model, save_path + f"fm_ckpt_{n_gradient_step + 1}.pt")
                torch.save(forward_model, save_path + f"fm_ckpt_latest.pt")
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
        results = []
        for damping in args.damping:
            agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
            agent.eval()
            invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn.eval()
            params = {"damping": damping}
            envs = gym.vector.SyncVectorEnv(
                [make_env(args, idx, params=params) for idx in range(args.num_envs)],
            )
            res = inference(args, envs, dataset, agent, invdyn, logger, args.diffusion_gradient_steps, params=params)
            res["damping"] = damping
            results.append(res)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_filepath = os.path.join(args.save_ft_res_path, f"damping_result_{timestamp}.pkl")
        with open(save_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saving results to file path {save_filepath}")
    
    elif args.mode == "finetune":
        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{args.save_model_folder}'
        results = []
        for damping in args.damping:
            agent = ContinuousDiffusionSDE(
                nn_diffusion,
                fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
                device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
            forward_model = ForwardMLP(fm_input_dims, fm_output_dims, ResidualBlock)
            forward_model.to(args.device)

            agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
            agent.eval()
            invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn.eval()
            #forward_model = torch.load(save_path + f"fm_ckpt_{args.fm_ckpt}.pth", weights_only=False)
            forward_model = torch.load(save_path + f"fm_ckpt_{args.fm_ckpt}.pt", weights_only=False)
            forward_model.eval()
            fm_optimizer = torch.optim.Adam(forward_model.parameters(), lr=5e-4)
            fm_criterion = torch.nn.MSELoss()
            
            params = {"damping": damping}
            print(f"{params=}")
            envs = gym.vector.SyncVectorEnv(
                [make_env(args, idx, params=params) for idx in range(args.num_envs)],
            )
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            # Start evaluate and fine-tune the inverse dynamics and forward dynamics models
            res, finetune_data = inference(args, envs, dataset, agent, invdyn, 
                        logger, args.diffusion_gradient_steps, params=params, return_data=True)
            res["damping"] = damping
            res["finetune_invdyn"] = False
            res["finetune_planner"] = False
            results.append(res)
            finetune_filename = f"damping_{damping}_{timestamp}.zarr"
            finetune_filepath = os.path.join(args.save_ft_data_path, finetune_filename)
            create_pusht_zarr(finetune_data, finetune_filepath)
            ft_dataset_path = os.path.expanduser(finetune_filepath)
            if args.env_name == 'pusht-v0':
                ft_dataset = PushTStateDataset(ft_dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                        pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
            elif args.env_name == 'pusht-keypoints-v0':
                ft_dataset = PushTKeypointDataset(ft_dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                        pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
            ft_dataset.normalizer = dataset.normalizer
            ft_dataloader = torch.utils.data.DataLoader(
                ft_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True
            )

            # Finetune
            invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps_ft)
            invdyn.train()
            forward_model.train()
            n_gradient_step = 0
            log = {"avg_loss_invdyn": 0., "avg_loss_fm": 0.}

            for batch in loop_dataloader(ft_dataloader):
                fm_optimizer.zero_grad()
                if args.env_name == "pusht-keypoints-v0":
                    obs_keypoint = batch["obs"]["keypoint"].to(args.device)
                    obs_agent_pos = batch["obs"]["agent_pos"].to(args.device)
                    obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
                    obs_dict = {"keypoint": obs_keypoint, "agent_pos": obs_agent_pos}
                elif args.env_name == "pusht-v0":
                    obs = batch["obs"]["state"].to(args.device)
                    obs_dict = {"state": obs}
                act = batch["action"].to(args.device)
                #val = batch["val"].to(args.device) / return_scale

                # ----------- Gradient Step ------------
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()
                if n_gradient_step < args.fm_gradient_steps:
                    x_fm, y_fm_1, y_fm_2 = format_data_for_forward_model(obs_dict, obs_agent_pos, act)
                    out_1, out_2 = forward_model(x_fm)
                    loss_fm_1 = fm_criterion(out_1, y_fm_1)
                    loss_fm_2 = fm_criterion(out_2, y_fm_2)
                    loss_fm = loss_fm_1 + loss_fm_2
                    loss_fm.backward()
                    fm_optimizer.step()
                    log["avg_loss_fm"] += loss_fm.item()

                # ----------- Logging ------------
                if (n_gradient_step + 1) % args.log_interval == 0:
                    log["step"] = n_gradient_step + 1
                    log["avg_loss_invdyn"] /= args.log_interval
                    log["avg_loss_fm"] /=args.log_interval
                    print(log)
                    #logger.log(log, category="train")
                    log = {"avg_loss_invdyn": 0., "avg_loss_fm": 0.,}

                # ----------- Saving ------------
                if (n_gradient_step + 1) % args.ft_save_interval == 0:
                    invdyn.save(save_path + f"damping_{damping}_invdyn_ckpt_{n_gradient_step + 1}.pt")
                    invdyn.save(save_path + f"damping_{damping}_invdyn_ckpt_latest.pt")
                    torch.save(forward_model, save_path + f"damping_{damping}_fm_ckpt_{n_gradient_step + 1}.pth")
                    torch.save(forward_model, save_path + f"damping_{damping}_fm_ckpt_latest.pth")

                n_gradient_step += 1
                if n_gradient_step >= args.invdyn_gradient_steps_ft:
                    break
            # Create the planner fine-tune dataset
            invdyn.eval()
            forward_model.eval()
            agent.eval()
            result = inference(args, envs, dataset, agent, invdyn, 
                        logger, args.diffusion_gradient_steps, params=params, return_data=False)
            result["damping"] = damping
            result["finetune_invdyn"] = True
            result["finetune_planner"] = False
            results.append(result)
            # Fine-tune for the trajectory planner
            finetune_traj_data = create_planner_finetune_dataset(args, dataloader, dataset.normalizer,
            agent, invdyn, forward_model, obs_goal, k_sample=args.k_sample, best_k=args.best_k)
            # TODO: Visualize the imagined rollout trajectory here. 2 criteria: (1) Dynamic plausibility, (2) Moving towards the goal
            ft_traj_filepath = f"temp/ft_traj/ft_traj_{timestamp}.pkl"
            with open(ft_traj_filepath, 'wb') as f:
                pickle.dump(finetune_traj_data, f)
            print(f"Saving finetune_traj_data to file path {ft_traj_filepath}")
        # Start tuning the planner
            planner_ft_dataset = PushTPlannerDataset(finetune_traj_data)
            planner_ft_dataloader = torch.utils.data.DataLoader(
                planner_ft_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )
            agent.train()
            n_gradient_step = 0
            log = {"avg_loss_diffusion": 0.}
            diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps_ft)
            for batch in loop_dataloader(planner_ft_dataloader):
                obs = batch["obs"].to(args.device)
                log["avg_loss_diffusion"] += agent.update(obs)["loss"]
                diffusion_lr_scheduler.step()

                if (n_gradient_step + 1) % args.log_interval == 0:
                    log["step"] = n_gradient_step + 1
                    log["avg_loss_diffusion"] /= args.log_interval
                    print(log)
                    log["avg_loss_diffusion"] = 0.0
                
                if (n_gradient_step + 1) % args.save_interval_planner_ft == 0:
                    agent.save(save_path + f"damping_{damping}_diffusion_ckpt_{n_gradient_step + 1}.pt")

                n_gradient_step += 1
                if n_gradient_step >= args.diffusion_gradient_steps_ft:
                    break
            
            result = inference(args, envs, dataset, agent, invdyn, 
                        logger, args.diffusion_gradient_steps, params=params, return_data=False)
            result["damping"] = damping
            result["finetune_invdyn"] = True
            result["finetune_planner"] = True
            results.append(result)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_filepath = os.path.join(args.save_ft_res_path, f"finetune_result_{timestamp}.pkl")
        with open(save_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saving eval_result to file path {save_filepath}")


    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()