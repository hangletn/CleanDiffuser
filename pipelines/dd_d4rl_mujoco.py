import os
import pickle
from datetime import datetime

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed
from env_utils import modify_env
from cleandiffuser.env.wrapper import VideoRecordingWrapper
from cleandiffuser.env.utils import VideoRecorder

def generate_random_hex_string(size=5):
    hex_array = np.array([i for i in "0123456789abcdef"])
    return "".join(list(np.random.choice(hex_array, size=size)))

def make_env(args, idx, params):
    def thunk():
        env = modify_env(args, mass_scale=params["mass_scale"], friction_scale=params["friction_scale"])
        if args.save_video:
            video_recorder = VideoRecorder.create_h264(
                                fps=20,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=22,
                                thread_type='FRAME',
                                thread_count=1
                            )
            hex_str = generate_random_hex_string()
            file_name = f"mass_{params['mass_scale']}_friction_{params['friction_scale']}_{idx}_{hex_str}.mp4"
            file_folder = os.path.join("videos", f"{args.task.env_name}")
            if not os.path.exists(file_folder):
                os.makedirs(file_folder, exist_ok=True)
            file_path = os.path.join(file_folder, file_name)
            env = VideoRecordingWrapper(env, video_recorder, file_path=file_path, steps_per_render=1)
        #env.seed(args.seed+idx)
        return env
    return thunk

def save_video(frame_list, args, params):
    import imageio
    from PIL import Image
    img_list = [Image.fromarray(img) for img in frame_list]
    hex_str = generate_random_hex_string()
    file_name = f"mass_{params['mass_scale']}_friction_{params['friction_scale']}_{hex_str}.mp4"
    file_folder = os.path.join("videos", f"{args.task.env_name}")
    if not os.path.exists(file_folder):
        os.makedirs(file_folder, exist_ok=True)
    file_path = os.path.join(file_folder, file_name)
    imageio.mimsave(file_path, img_list)
    print(f"save video at {file_path}")

def run_inference(args, initial_env, env, dataset, agent, invdyn, save_path, params, return_data=False):
    finetune_data = {
        "actions": [],
        "next_observations": [],
        "observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": []
    }
    agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
    agent.eval()
    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
    invdyn.eval()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    mass_scale, friction_scale = params["mass_scale"], params["friction_scale"]

    # Use `SynVectorEnv` to record videos and `AsyncVectorEnv` for evaluation
    if args.save_video:
        env_eval = gym.vector.AsyncVectorEnv([make_env(args, idx, params) for idx in range(args.num_envs)])
    else:
        env_eval = gym.vector.AsyncVectorEnv([lambda: env for _ in range(args.num_envs)])

    normalizer = dataset.get_normalizer()
    episode_rewards = []

    prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
    condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return
    for i in range(args.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            finetune_data["observations"].append(obs)
            obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

            # sample trajectories
            prior[:, 0] = obs
            traj, log = agent.sample(
                prior, solver=args.solver,
                n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)

            # inverse dynamic
            with torch.no_grad():
                act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

            # step
            obs, rew, done, info = env_eval.step(act)

            # add data to `finetune_data`
            finetune_data["actions"].append(act)
            finetune_data["next_observations"].append(obs)
            finetune_data["rewards"].append(rew)
            finetune_data["terminals"].append(done)
            if t == 1000:
                finetune_data["timeouts"].append(np.array([True for _ in range(args.num_envs)]))
            else:
                finetune_data["timeouts"].append(np.array([False for _ in range(args.num_envs)]))

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
            if (t+1) % args.eval_print_freq == 0:
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

        episode_rewards.append(ep_reward)

    episode_rewards = [list(map(lambda x: initial_env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)
    result = {
        "mass_scale": mass_scale,
        "friction_scale": friction_scale,
        "episode_rewards": episode_rewards
    }
    print(f"mass scale: {mass_scale}, friction scale: {friction_scale}")
    print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
    if return_data:
        new_finetune_data = {}
        for key, val in finetune_data.items():
            new_val = {env_idx: [] for env_idx in range(args.num_envs)}
            for i in range(len(val)): # num_step
                for j in range(len(val[i])): # num_envs
                    new_val[j].append(val[i][j])
            new_val = np.concatenate([np.stack(new_val[env_idx]) for env_idx in range(args.num_envs)])
            print(f"key: {key} val: {new_val.shape}")
            new_finetune_data[key] = new_val
    else:
        new_finetune_data = {}

    return result, new_finetune_data

@hydra.main(config_path="../configs/dd/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

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
    fix_mask[0] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    loss_weight[1] = args.next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    agent = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic -------------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # # -------------- Add wandb logging ------------------
    # import wandb
    # from omegaconf import OmegaConf

    # cfg_dict = OmegaConf.to_container(args, resolve=True)
    # run = wandb.init(
    #     #entity="hangle-harvard-univeristy",
    #     project=args.task.env_name,
    #     config=cfg_dict,
    # )

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        agent.train()
        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.,  "avg_loss_invdyn": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs, val)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.invdyn_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                print(log)
                #run.log(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}

            # ----------- Saving ------------
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
        agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
        agent.eval()
        invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
        invdyn.eval()

        results = []
        for mass_scale in args.body_mass:
            for friction_scale in args.geom_friction:
                initial_env = gym.make(args.task.env_name)
                env = modify_env(args, mass_scale=mass_scale, friction_scale=friction_scale)
                params = {"mass_scale": mass_scale, "friction_scale": friction_scale}
                result, _ = run_inference(args, initial_env, env, dataset, agent, invdyn, save_path, params, return_data=False)
                results.append(result)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_filepath = f"mass_result_{timestamp}.pkl"
        with open(save_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saving training states (acc, loss) to file path {save_filepath}")
    
    elif args.mode == "finetune":
        results = []
        for mass_scale in args.body_mass:
            for friction_scale in args.geom_friction:
                agent = ContinuousDiffusionSDE(
                    nn_diffusion, nn_condition,
                    fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
                    device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

                # --------------- Inverse Dynamic -------------------
                invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
                agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
                agent.eval()
                invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                invdyn.eval()
                initial_env = gym.make(args.task.env_name)
                env = modify_env(args, mass_scale=mass_scale, friction_scale=friction_scale)
                params = {"mass_scale": mass_scale, "friction_scale": friction_scale}
                result, finetune_data = run_inference(args, initial_env, env, dataset, agent, invdyn, save_path, params, return_data=True)
                result["finetune"] = False
                results.append(result)

                invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps_ft)
                ft_dataset = D4RLMuJoCoDataset(
                    finetune_data, horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
                ft_dataloader = DataLoader(
                    ft_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
                invdyn.train()
                n_gradient_step = 0.0
                log = {"avg_loss_invdyn": 0.0}
                for batch in loop_dataloader(ft_dataloader):
                    obs = batch["obs"]["state"].to(args.device)
                    act = batch["act"].to(args.device)
                    val = batch["val"].to(args.device) / return_scale
                    log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                    invdyn_lr_scheduler.step()
                    # ----------- Logging ------------
                    if (n_gradient_step + 1) % args.log_interval == 0:
                        log["gradient_steps"] = n_gradient_step + 1
                        log["avg_loss_invdyn"] /= args.log_interval
                        print(log)
                        #run.log(log)
                        log = {"avg_loss_invdyn": 0.0}

                    # ----------- Saving ------------
                    if (n_gradient_step + 1) % args.ft_save_interval == 0:
                        invdyn.save(save_path + f"mass_{mass_scale}_friction_{friction_scale}_invdyn_ckpt_{n_gradient_step + 1}.pt")
                        invdyn.save(save_path + f"mass_{mass_scale}_friction_{friction_scale}_invdyn_ckpt_latest.pt")

                    n_gradient_step += 1
                    if n_gradient_step >= args.invdyn_gradient_steps_ft:
                        break
                invdyn.eval()
                result, _ = run_inference(args, initial_env, env, dataset, agent, invdyn, save_path, params, return_data=False)
                result["finetune"] = True
                results.append(result)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_filepath = f"finetune_result_{timestamp}.pkl"
        with open(save_filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saving eval_result to file path {save_filepath}")

        pass

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
