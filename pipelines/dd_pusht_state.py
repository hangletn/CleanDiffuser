import os
import pickle
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

import d4rl
import gym
import pathlib
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

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
from env_utils import create_pusht_zarr, make_info_concat_pusht, PushTPlannerDataset, get_pusht_goal_keypoint, get_pusht_goal_pose, s6_to_s5, s5_to_s6, renorm_cossin, PushTFinetuneDataset
from env_utils import EarlyStopping
from forward_model import ForwardMLP, ResidualBlock, ForwardMLPChunk

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
        env = MultiStepWrapper(env, n_obs_steps=env_step, n_action_steps=env_step)
        #env = MultiStepWrapper(env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps)
        env.seed(args.seed+idx)
        print("Env seed: ", args.seed+idx)
        return env

    return thunk

def inference(args, envs, dataset, agent, invdyn, logger, current_step,
            mode="train", params={}, return_data=False, video_title="ft00", num_eps=None):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    finetune_data = []
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_coverages = []
    seeds = []
    obs_dim, act_dim = args.obs_dim, args.act_dim
    damping = params["damping"] if "damping" in params else None
    gravity = params["gravity"] if "gravity" in params else None
    if mode == "train":
        start_seed = 0
    elif mode == "test":
        start_seed = 10000
    else:
        raise ValueError(f"Mode {mode} must be train or test")
    
    if num_eps is None:
        num_eps = args.eval_episodes
    else:
        num_eps = num_eps

    for i in range(num_eps // args.num_envs):
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
        # seed = np.random.randint(0,25536)
        # print(f"Env {i}: Seed {seed}")
        # obs, t = envs.reset(seed=seed), 0
        seed = start_seed + i
        seeds.append(seed)
        obs, t = envs.reset(seed=seed), 0
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
        warm_start = args.warm_start

        # initialize video stream
        if args.save_video:
            if damping is not None:
                logger.video_init(envs.envs[0], enable=True, video_id=f"{current_step}_{i}_{seed}_{video_title}_damping_{damping}")  # save videos
            else:
                logger.video_init(envs.envs[0], enable=True, video_id=f"{current_step}_{i}_{seed}_{video_title}")  # save videos

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
        ep_reward = np.around(np.array(ep_reward), 2)
        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        max_coverage_idx = np.argmax(np.concatenate(coverage_area_list), axis=0)
        ep_coverage = np.around(np.max(np.concatenate(coverage_area_list), axis=0), 2)
        if ep_coverage >= args.ft_coverage_threshold:
            this_finetune_data_aligned = {k: v[:max_coverage_idx] for (k,v) in this_finetune_data_aligned.items()}
            finetune_data.append(this_finetune_data_aligned)
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {ep_reward} success:{success}")
        episode_rewards.append(ep_reward.item())
        episode_steps.append(t)
        episode_success.append(success.item())
        episode_coverages.append(ep_coverage)
    success_rate = np.nanmean(np.where(np.array(episode_success) == 1.0, 1.0, 0.0))
    mean_coverage = np.nanmean(np.array(episode_coverages))
    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)} Success rate: {success_rate} Mean coverage: {mean_coverage}")
    res = {"step": episode_steps, "reward": episode_rewards, "success": episode_success, "coverage": episode_coverages, "seed": seeds,
            "summary": {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success), 'success_rate': success_rate, 'mean_coverage': mean_coverage}}
    if not return_data:
        return res
    else:
        print(f"Number of fine-tuned episodes: {len(finetune_data)}. Filter threshold: {args.ft_coverage_threshold}")
        return res, finetune_data
    
def compute_reward(obs_kp, obs_kp_goal, rew_weight):
    # TODO: Check whether it would work for pose as well (look like it should also work!)
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
    # TODO: Modify this function to work with chunk-level prediction
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

def actual_env_rollout(obs, act, args, params, normalizer, return_theta=False):
    # obs.shape (batch_size, horizon, obs_dim)
    # act.shape (batch_size, horizon, act_dim)
    # TODO: Use parallel env to rollout
    batch_size, horizon, obs_dim = obs.shape
    damping = params["damping"] if "damping" in params else None
    obs_rollout = []
    for i in range(batch_size):
        this_obs = obs[i]
        this_act = act[i]
        obs_prior = this_obs[:args.obs_steps]
        obs_0 = obs_prior[-1]
        act_sim = this_act[args.obs_steps-1:-1] # not count the last action
        env = gym.make(args.env_name, damping=damping, reset_to_state=obs_0)
        _ = env.reset()
        # Run the env rollout
        this_obs_rollout = []
        for act_sim_i in act_sim:
            obs_sim_i, reward, done, info = env.step(act_sim_i)
            this_obs_rollout.append(obs_sim_i)
        this_obs_rollout = np.concatenate([obs_prior, this_obs_rollout])
        obs_rollout.append(this_obs_rollout)
    obs_rollout = np.stack(obs_rollout)
    obs_rollout = normalizer["obs"]["state"].normalize(obs_rollout)
    if return_theta:
        return obs_rollout
    else:
        obs_rollout = s5_to_s6(torch.from_numpy(obs_rollout)).numpy()
        return obs_rollout

def make_env_rollout(args, params={}):
    def thunk():
        damping = params["damping"] if "damping" in params else None
        obs_0 = params["obs_0"] if "obs_0" in params else None
        env = gym.make(args.env_name, damping=damping, reset_to_state=obs_0)
        return env
    return thunk
def actual_env_rollout_async(obs, act, args, params, normalizer, return_theta=False, do_normalize=True):
    # Use async is actually longer...
    batch_size, horizon, obs_dim = obs.shape
    damping = params["damping"] if "damping" in params else None
    obs_rollout = []
    obs_prior = obs[:, :args.obs_steps]
    obs_0 = obs_prior[0, -1]
    env_params = {"damping": damping, "obs_0": obs_0}
    act_sim = act[:, args.obs_steps-1:-1]
    envs = gym.vector.AsyncVectorEnv(
        [make_env_rollout(args, params=env_params) for i in range(batch_size)]
    )
    _ = envs.reset()
    for step in range(act_sim.shape[1]):
        act_sim_step = act_sim[:, step, :]
        obs_sim_step, reward, done, info = envs.step(act_sim_step)
        obs_rollout.append(obs_sim_step)
    obs_rollout = np.stack(obs_rollout).transpose([1, 0, 2]) # (batch_size, horizon, obs_dim)
    obs_rollout = np.concatenate([obs_prior, obs_rollout], axis=1) # concat horizon
    if do_normalize:
        obs_rollout = normalizer["obs"]["state"].normalize(obs_rollout)
    if return_theta:
        return obs_rollout
    else:
        obs_rollout = s5_to_s6(torch.from_numpy(obs_rollout)).numpy()
        return obs_rollout

def create_planner_finetune_dataset(args, dataloader, obs_goal, 
    agent, invdyn, forward_model, normalizer, params, k_sample=16, best_k=4, use_actual_env=False):
    """ Similar to `inference` but have 2 difference
    1. Each data point in `dataloader` is used as the starting point for each episode, rollout can be done by either the forward model or the actual environment
    2. Sample k trajectory instead of 1 trajectory from the planner, rollout and choose the best-k
    (may consider filter out bad episode)
    """
    # ---------------- Start Rollout ----------------
    obs_dim, act_dim = args.obs_dim, args.act_dim
    finetune_traj = []
    reward_traj = []
    plan_traj = []
    action_traj = []
    count = 0
    if use_actual_env:
        print("USING ACTUAL ENV TO ROLL OUT!!!")
    for batch in tqdm(dataloader):
        # Extract obs and act from batch dataloader
        if args.env_name == "pusht-keypoints-v0":
            obs_keypoint = batch["obs"]["keypoint"].to(args.device)
            obs_agent_pos = batch["obs"]["agent_pos"].to(args.device)
            obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
        elif args.env_name == "pusht-v0":
            obs = batch["obs"]["state"].to(args.device)
        act = batch["action"].to(args.device)
        contacts = batch["n_contacts"].type(torch.float).to(args.device)
        batch_size = obs.size()[0]
        # Have a k sample loop here, use agent.sample() and invdyn.predict() to get obs_hat, act_hat
        # TODO: Try to make minibatch size here -> instead of 1, do (batch_size/k_sample) instead
        # TODO: Do a performance run to see at which `batch_size` using `AsyncVectorEnv` would be faster? This may also be a RAM problem
        for i in range(batch_size):
            prior = torch.zeros((k_sample, args.task.horizon, obs_dim), device=args.device)
            this_obs, this_act = obs[i], act[i]
            # Repeat k-sample
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
            # TODO: Update this `forward_model_rollout()` function
            #traj_fm = forward_model_rollout(forward_model, obs_repeat[:, :args.obs_steps], action_pred, args)
            # Rollout using forward model or actual env
            if use_actual_env:
                action_pred_raw = normalizer["action"].unnormalize(action_pred.detach().cpu().numpy())
                obs_repeat_raw = normalizer["obs"]["state"].unnormalize(obs_repeat.detach().cpu().numpy())
                traj_fm = actual_env_rollout(obs_repeat_raw, action_pred_raw, args, params, normalizer, return_theta=False)
            else:
                fm_pred = forward_model.predict(obs_repeat, action_pred, args, return_theta=False)
                traj_fm = fm_pred["state_pred"]
                traj_fm = traj_fm.cpu().numpy()
            # Compute rewards and filter
            rew_mask = np.ones((args.task.horizon - args.obs_steps))
            rew_mask[:args.action_steps] = args.next_obs_loss_weight # weight the action horizon higher
            if args.env_name == "pusht-keypoints-v0":
                traj_fm_kp = traj_fm[:, args.obs_steps-1:, :18]
            elif args.env_name == "pusht-v0":
                traj_fm_kp = traj_fm[:, args.obs_steps-1:, 2:]
            traj_rew = np.array([compute_reward(i, obs_goal, rew_mask) for i in traj_fm_kp])
            traj_fm_best_k_idx = np.argsort(traj_rew)[-best_k:]
            traj_rew_best_k = traj_rew[traj_fm_best_k_idx]
            traj_fm_best_k = traj_fm[traj_fm_best_k_idx, :]
            action_best_k = action_pred[traj_fm_best_k_idx, :]
            #traj_fm_best_k_append = np.concatenate([obs_repeat[:best_k, :args.obs_steps], traj_fm_best_k], axis=1) # concat along horizon
            traj_fm_best_k = renorm_cossin(torch.from_numpy(traj_fm_best_k), eps=0.0)
            traj_fm_best_k = s6_to_s5(traj_fm_best_k).numpy()
            finetune_traj.append(traj_fm_best_k)
            reward_traj.append(traj_rew_best_k)
            action_traj.append(action_best_k.detach().cpu().numpy())
        #break
            
        count += 1
    finetune_traj_data = np.concatenate(finetune_traj, axis=0) # (num_sample, horizon, obs_dim)
    reward_traj_data = np.concatenate(reward_traj, axis=0)
    plan_traj_data = np.concatenate(plan_traj, axis=0)
    action_traj_data = np.concatenate(action_traj, axis=0)
    return {"obs": finetune_traj_data,
            "rew": reward_traj_data,
            "plan_traj": plan_traj_data,
            "action": action_traj_data}

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

# TODO: Make this work for the chunk-level prediction as well
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

def compute_val_loss(val_dataloader, model, args):
    val_loss = 0.0
    num_sample = 0
    model.eval()
    for batch in val_dataloader:
        obs = batch["obs"].to(args.device)
        act = batch["action"].to(args.device)
        batch_size = len(obs)
        act_pred = model.forward(obs[:, :-1], obs[:, 1:])
        act_true = act[:,:-1]
        loss = ((act_pred - act_true) ** 2).mean()
        val_loss += loss.item()*batch_size
        num_sample += batch_size
    return val_loss / num_sample


@hydra.main(config_path="../configs/dd/pusht", config_name="pusht_state", version_base=None)
#@hydra.main(config_path="../configs/dd/pusht", config_name="pusht_state_test", version_base=None)
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
        # TODO: This may not work for the `MLPChunk`
        fm_output_dims = [obs_dim - act_dim, act_dim] # keypoints, agent_pos
    elif args.env_name == "pusht-v0":
        obs_in_dim = obs_dim + 1 # s5 to s6
        fm_input_dims = (obs_in_dim + act_dim) * 2 + act_dim * (args.task.horizon - args.obs_steps - 1)
        fm_output_dims = obs_in_dim * (args.task.horizon - args.obs_steps)
        contact_dims = args.task.horizon - args.obs_steps
    forward_model = ForwardMLPChunk(fm_input_dims, fm_output_dims, contact_dims, ResidualBlock, device=args.device)

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
        logger = Logger(pathlib.Path(args.work_dir), args)

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
        fm_lr_scheduler = CosineAnnealingLR(forward_model.optim, args.fm_gradient_steps)

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
            elif args.env_name == "pusht-v0":
                obs = batch["obs"]["state"].to(args.device)
            act = batch["action"].to(args.device)
            contacts = batch["n_contacts"].type(torch.float).to(args.device)
            #val = batch["val"].to(args.device) / return_scale

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step < args.fm_gradient_steps:
                log["avg_loss_fm"] += forward_model.update(obs, act, contacts, args)["loss"]
                fm_lr_scheduler.step()

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
            # TODO: Add an `AND` here so forward_model does not get saved every 1000 steps
            if (n_gradient_step + 1) % args.fm_save_interval == 0:
                forward_model.save(save_path + f"fm_ckpt_{n_gradient_step + 1}.pt")
                forward_model.save(save_path + f"fm_ckpt_latest.pt")
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
        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{args.save_model_folder}'
        results = []
        for damping in args.damping:
            logger = Logger(pathlib.Path(args.work_dir), args)
            print(f"Evaluating damping value {damping}")
            agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
            agent.eval()
            invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn.eval()
            params = {"damping": damping}
            envs = gym.vector.SyncVectorEnv(
                [make_env(args, idx, params=params) for idx in range(args.num_envs)],
            )
            for mode in ["train", "test"]:
                res = inference(args, envs, dataset, agent, invdyn, 
                logger, args.diffusion_gradient_steps, params=params, mode=mode)
                res["damping"] = damping
                res["finetune_invdyn"] = False
                res["finetune_planner"] = False
                res["eval"] = mode
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
            this_args = deepcopy(args)
            this_args.damping = damping
            this_args.work_dir = os.path.join(args.work_dir, f"damping_{damping}")
            this_args.group = f"{this_args.group}_damping_{damping}"
            this_args.exp_name = f"{this_args.exp_name}_damping_{damping}"
            logger = Logger(pathlib.Path(this_args.work_dir), this_args)
            agent = ContinuousDiffusionSDE(
                nn_diffusion,
                fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
                device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
            forward_model = ForwardMLPChunk(fm_input_dims, fm_output_dims, contact_dims, ResidualBlock, device=args.device)

            agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
            agent.eval()
            invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn.eval()
            forward_model.load(save_path + f"fm_ckpt_{args.fm_ckpt}.pt")
            forward_model.eval()
            
            params = {"damping": damping}
            print(f"{params=}")
            envs = gym.vector.SyncVectorEnv(
                [make_env(args, idx, params=params) for idx in range(args.num_envs)],
            )
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            # Start evaluate and fine-tune the inverse dynamics and forward dynamics models
            print("____Evaluating invdyn_False_planner_False!____")
            for mode in ["train", "test"]:
                if mode == "train":
                    res, rollout_data = inference(args, envs, dataset, agent, invdyn, logger, 
                    args.diffusion_gradient_steps, params=params, return_data=True, mode=mode)
                if mode == "test":
                    res = inference(args, envs, dataset, agent, invdyn, 
                                logger, args.diffusion_gradient_steps, params=params, return_data=False, mode=mode)
                res["damping"] = damping
                res["finetune_invdyn"] = False
                res["finetune_planner"] = False
                res["eval"] = mode
                results.append(res)
            rollout_filename = f"damping_{damping}_{timestamp}.zarr"
            rollout_filepath = os.path.join(args.save_ft_data_path, rollout_filename)
            create_pusht_zarr(rollout_data, rollout_filepath, use_keypoint=False)
            rollout_dataset_path = os.path.expanduser(rollout_filepath)
            if args.env_name == 'pusht-v0':
                rollout_dataset = PushTStateDataset(rollout_dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                        pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
            elif args.env_name == 'pusht-keypoints-v0':
                rollout_dataset = PushTKeypointDataset(rollout_dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                        pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
            rollout_dataset.normalizer = dataset.normalizer
            # ft_dataloader = torch.utils.data.DataLoader(
            #     ft_dataset,
            #     batch_size=args.batch_size,
            #     num_workers=4,
            #     shuffle=True,
            #     pin_memory=True,
            #     persistent_workers=True,
            #     drop_last=True
            # )

            # # Finetune
            # invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps_ft)
            # fm_lr_scheduler = CosineAnnealingLR(forward_model.optim, args.fm_gradient_steps_ft)
            # save_path_ft = "ft_models/"
            # invdyn.train()
            # forward_model.train()
            # n_gradient_step = 0
            # log = {"avg_loss_invdyn": 0., "avg_loss_fm": 0.}

            # for batch in loop_dataloader(ft_dataloader):
            #     if args.env_name == "pusht-keypoints-v0":
            #         obs_keypoint = batch["obs"]["keypoint"].to(args.device)
            #         obs_agent_pos = batch["obs"]["agent_pos"].to(args.device)
            #         obs = torch.cat([obs_keypoint, obs_agent_pos], dim=-1)
            #         obs_dict = {"keypoint": obs_keypoint, "agent_pos": obs_agent_pos}
            #     elif args.env_name == "pusht-v0":
            #         obs = batch["obs"]["state"].to(args.device)
            #         obs_dict = {"state": obs}
            #     act = batch["action"].to(args.device)
            #     contacts = batch["n_contacts"].type(torch.float).to(args.device)
            #     #val = batch["val"].to(args.device) / return_scale

            #     # ----------- Gradient Step ------------
            #     log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
            #     invdyn_lr_scheduler.step()
            #     if n_gradient_step < args.fm_gradient_steps_ft:
            #         log["avg_loss_fm"] += forward_model.update(obs, act, contacts, args)["loss"]
            #         fm_lr_scheduler.step()

            #     # ----------- Logging ------------
            #     if (n_gradient_step + 1) % args.log_interval == 0:
            #         log["step"] = n_gradient_step + 1
            #         log["avg_loss_invdyn"] /= args.log_interval
            #         log["avg_loss_fm"] /=args.log_interval
            #         print(log)
            #         #logger.log(log, category="train")
            #         log = {"avg_loss_invdyn": 0., "avg_loss_fm": 0.,}

            #     # ----------- Saving ------------
            #     if (n_gradient_step + 1) % args.save_interval_invdyn_ft == 0:
            #         invdyn.save(save_path_ft + f"damping_{damping}_invdyn_ckpt_{n_gradient_step + 1}.pt")
            #         invdyn.save(save_path_ft + f"damping_{damping}_invdyn_ckpt_latest.pt")
            #         forward_model.save(save_path_ft + f"damping_{damping}_fm_ckpt_{n_gradient_step + 1}.pt")
            #         forward_model.save(save_path_ft + f"damping_{damping}_fm_ckpt_latest.pt")

            #     n_gradient_step += 1
            #     if n_gradient_step >= args.invdyn_gradient_steps_ft:
            #         break
            # Evaluate fine-tuned invdyn model
            # invdyn.eval()
            # forward_model.eval()
            # agent.eval()
            # Create the planner fine-tune dataset
            if args.ft_using_cache: # use cache planner fine-tune data
                ft_traj_filepath = os.path.join(args.ft_cache_path, f"ft_traj_damping_{damping}.pkl")
                finetune_traj_data = pickle.load(open(ft_traj_filepath, "rb"))
            else:
                ft_traj_filepath = f"temp/ft_traj/ft_traj_damping_{damping}_{timestamp}.pkl"
                finetune_traj_data = create_planner_finetune_dataset(args, dataloader, obs_goal,
                agent, invdyn, forward_model, dataset.normalizer, params, k_sample=args.k_sample, best_k=args.best_k,
                use_actual_env=args.use_actual_env)
                # TODO: Visualize the imagined rollout trajectory here. 2 criteria: (1) Dynamic plausibility, (2) Moving towards the goal
                with open(ft_traj_filepath, 'wb') as f:
                    pickle.dump(finetune_traj_data, f)
                print(f"Saving finetune_traj_data to file path {ft_traj_filepath}")
            # Start fine-tuning the planner and inverse dynamics model
            planner_ft_dataset = PushTPlannerDataset(finetune_traj_data)
            if args.ft_using_combined:
                planner_ft_dataset = PushTFinetuneDataset(planner_ft_dataset, rollout_dataset)
            torch_seed = torch.Generator().manual_seed(args.seed)
            planner_ft_train_dataset, planner_ft_val_dataset = random_split(planner_ft_dataset, [0.8, 0.2], generator=torch_seed)
            planner_ft_dataloader = torch.utils.data.DataLoader(
                planner_ft_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )
            planner_ft_train_dataloader = torch.utils.data.DataLoader(
                planner_ft_train_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )
            planner_ft_val_dataloader = torch.utils.data.DataLoader(
                planner_ft_val_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )
            # Fine-tune `invdyn` first
            n_gradient_step = 0
            log = {"avg_loss_invdyn": 0.}
            invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps_ft)
            save_path_ft = f"ft_models/{timestamp}/"
            if not os.path.exists(save_path_ft):
                os.makedirs(save_path_ft, exist_ok=True)
            #best_val_loss = float("inf")
            # TODO: Add `EarlyStop` here
            early_stop_invdyn = EarlyStopping(patience=3,
                                verbose=True, mode="min",
                                path=save_path_ft + f"damping_{damping}_invdyn_ckpt_best.pt")
            for batch in loop_dataloader(planner_ft_train_dataloader):
                invdyn.train()
                obs = batch["obs"].to(args.device)
                act = batch["action"].to(args.device)
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()
                
                if (n_gradient_step + 1) % args.eval_interval_ft == 0:
                    val_loss = compute_val_loss(planner_ft_val_dataloader, invdyn, args)
                    log["val_loss_invdyn"] = val_loss
                    early_stop_invdyn(val_loss, invdyn)
                    # if val_loss < best_val_loss:
                    #     best_val_loss = val_loss
                    #     print(f"Step {n_gradient_step+1}, {best_val_loss=}")
                    #     invdyn.save(save_path_ft + f"damping_{damping}_invdyn_ckpt_best.pt")
                
                if (n_gradient_step + 1) % args.log_interval == 0:
                    log["step"] = n_gradient_step + 1
                    log["avg_loss_invdyn"] /= args.log_interval
                    print(log)
                    logger.log(log, category="train")
                    log = {"avg_loss_invdyn": 0., }

                if (n_gradient_step + 1) % args.save_interval_invdyn_ft == 0:
                    invdyn.save(save_path_ft + f"damping_{damping}_invdyn_ckpt_{n_gradient_step + 1}.pt")
                    invdyn.save(save_path_ft + f"damping_{damping}_invdyn_ckpt_latest.pt")
                
                if early_stop_invdyn.early_stop:
                    break
                
                n_gradient_step += 1
                if n_gradient_step >= args.invdyn_gradient_steps_ft:
                    break
            
            # Fine-tune `planner``
            invdyn_steps = n_gradient_step
            n_gradient_step = 0
            log = {"avg_loss_diffusion": 0.,}
            diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps_ft)
            early_stop_planner = EarlyStopping(patience=3,
                                verbose=True, mode="max",
                                path=save_path_ft + f"damping_{damping}_diffusion_ckpt_best.pt")
            #best_score = float("-inf")
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
            invdyn.load(save_path_ft + f"damping_{damping}_invdyn_ckpt_best.pt")
            invdyn.eval()
            for batch in loop_dataloader(planner_ft_dataloader):
                agent.train()
                obs = batch["obs"].to(args.device)
                act = batch["action"].to(args.device)
                log["avg_loss_diffusion"] += agent.update(obs)["loss"]
                diffusion_lr_scheduler.step()

                if (n_gradient_step + 1) % args.log_interval == 0:
                    log["step"] = n_gradient_step + 1 + invdyn_steps # quick hack for `wandb`
                    log["avg_loss_diffusion"] /= args.log_interval
                    print(log)
                    logger.log(log, category="train")
                    log = {"avg_loss_diffusion": 0.,}
                
                if (n_gradient_step + 1) % args.eval_interval_ft == 0:
                    agent.eval()
                    eval_log = inference(args, envs, dataset, agent, invdyn, logger, n_gradient_step + 1, 
                    params=params, return_data=False, mode="train", video_title="ft_eval")
                    eval_log_summary = eval_log["summary"]
                    eval_log_summary["step"] = n_gradient_step + 1 + invdyn_steps
                    logger.log(eval_log_summary, category="inference")
                    current_score = eval_log_summary["mean_coverage"]
                    early_stop_planner(current_score, agent)
                    # if current_score > best_score:
                    #     print(f"Step {n_gradient_step + 1}: {best_score=}, {current_score=}")
                    #     best_score = current_score
                    #     # Save the current model
                    #     agent.save(save_path_ft + f"damping_{damping}_diffusion_ckpt_best.pt")
                
                if (n_gradient_step + 1) % args.save_interval_planner_ft == 0:
                    agent.save(save_path_ft + f"damping_{damping}_diffusion_ckpt_{n_gradient_step + 1}.pt")
                    agent.save(save_path_ft + f"damping_{damping}_diffusion_ckpt_latest.pt")
                
                if early_stop_planner.early_stop:
                    break

                n_gradient_step += 1
                if n_gradient_step >= args.diffusion_gradient_steps_ft:
                    break
            
            agent = ContinuousDiffusionSDE(
                nn_diffusion,
                fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
                device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
            agent.load(save_path_ft + f"damping_{damping}_diffusion_ckpt_best.pt")
            agent.eval()
            invdyn.eval()
            print("____Evaluating invdyn_True_planner_True!____")
            for mode in ["train", "test"]:
                res = inference(args, envs, dataset, agent, invdyn, logger, 
                args.diffusion_gradient_steps, params=params, return_data=False, mode=mode, video_title="ft11")
                res["damping"] = damping
                res["finetune_invdyn"] = True
                res["finetune_planner"] = True
                res["eval"] = mode
                results.append(res)
            
            # Add evaluation for fine-tuning the planner but not the invdyn
            # Load the original invdyn
            invdyn_0 = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
            invdyn_0.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn_0.eval()
            print("____Evaluating invdyn_False_planner_True!____")
            for mode in ["train", "test"]:
                res = inference(args, envs, dataset, agent, invdyn_0, logger, args.diffusion_gradient_steps, 
                params=params, return_data=False, mode=mode, video_title="ft01")
                res["damping"] = damping
                res["finetune_invdyn"] = False
                res["finetune_planner"] = True
                res["eval"] = mode
                results.append(res)
            
            # Add evaluation for fine-tuning the invdyn but not the planner
            # Load the original planner
            agent_0 = ContinuousDiffusionSDE(
                nn_diffusion,
                fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
                device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")
            agent_0.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
            agent_0.eval()
            print("____Evaluating invdyn_True_planner_False!____")
            for mode in ["train", "test"]:
                res = inference(args, envs, dataset, agent_0, invdyn, logger, 
                args.diffusion_gradient_steps, params=params, return_data=False, mode=mode, video_title="ft10")
                res["damping"] = damping
                res["finetune_invdyn"] = True
                res["finetune_planner"] = False
                res["eval"] = mode
                results.append(res)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_filepath = os.path.join(args.save_ft_res_path, f"finetune_result_{timestamp}.pkl")
        with open(save_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saving eval_result to file path {save_filepath}")


    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()