import gym
import mujoco
import mujoco_py
import d4rl
from gym import Wrapper
import zarr
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', mode="min"):
        """
        Args:
            patience (int): How long to wait after last validation loss improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, val_loss, model):
        if self.mode == "min":
            score = -val_loss
        elif self.mode == "max":
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        model.save(self.path)
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def detect_data_duplication(data_sample, eps=1e-5):
    """
    Docstring for detect_data_duplication
    
    :param data_sample (np.darray): (n_sample, horizon, obs_dims). Assumed data is normalized at range [-1, 1]
    :param eps: Filter threshold. Minimum L2-distance average
    """
    # Compute intra-distance
    num_sample, horizon, obs_dim = data_sample.shape
    dist_list = []
    # iterate and compute L2 distance of each pair

    # filter out the data sample if average absolute distance of each units < eps (assumed inputs data is normalized)
    # return the filtered idx
    raise NotImplementedError

def s5_to_s6(s5: torch.Tensor, only_block=False) -> torch.Tensor:
    """
    s5: [B, T, 5] -> s6: [B, T, 6]
    last dim: [xa, ya, xb, yb, theta] -> [xa, ya, xb, yb, cos(theta), sin(theta)]
    """
    if only_block:
        xb, yb, th = s5.unbind(dim=-1)
        th = ((th + 1.0) / 2.0) * 2*np.pi # unnormalize to have range (0, 2*pi)
        return torch.stack([xb, yb, torch.cos(th), torch.sin(th)], dim=-1)
    else:
        xa, ya, xb, yb, th = s5.unbind(dim=-1)
        th = ((th + 1.0) / 2.0) * 2*np.pi # unnormalize to have range (0, 2*pi)
        return torch.stack([xa, ya, xb, yb, torch.cos(th), torch.sin(th)], dim=-1)

def s6_to_s5(s6, only_block=False):
    if only_block:
        xb, yb, cos_th, sin_th = s6.unbind(dim=-1)
        th = torch.atan2(sin_th, cos_th) # [-pi, pi]
        th = torch.where(th < 0, th + 2 * torch.pi, th)
        th = (th / (2*np.pi)) * 2.0 - 1.0 # normalize back to range [-1, 1]
        return torch.stack([xb, yb, th], dim=-1)
    else:
        xa, ya, xb, yb, cos_th, sin_th = s6.unbind(dim=-1)
        th = torch.atan2(sin_th, cos_th)
        th = torch.where(th < 0, th + 2 * torch.pi, th)
        th = (th / (2*np.pi)) * 2.0 - 1.0 # normalize back to range [-1, 1]
        return torch.stack([xa, ya, xb, yb, th], dim=-1)

def renorm_cossin(s6: torch.Tensor, eps: float = 1e-8, only_block=False) -> torch.Tensor:
    """
    s6: [..., 6], renormalize last two dims to unit norm
    """
    if only_block:
        cos_idx, sin_idx = 2, 3
    else:
        cos_idx, sin_idx = 4, 5
    cos_th = s6[..., cos_idx]
    sin_th = s6[..., sin_idx]
    norm = torch.sqrt(cos_th**2 + sin_th**2 + eps)
    cos_th = cos_th / norm
    sin_th = sin_th / norm
    return torch.cat([s6[..., :cos_idx], cos_th.unsqueeze(-1), sin_th.unsqueeze(-1)], dim=-1)

def weighted_loss(y_hat, y, w, loss_func, reduction="mean"):
    batch_size, obs_dim = y.size()
    #w_rep = w.view(1, obs_dim).repeat(batch_size, 1)
    loss = w * loss_func(y_hat, y)
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"{reduction=} is invalid!")
    return loss

# TODO: Make the normalization consistent between `PushTPlannerDataset` and `PushTStateDataset`
class PushTFinetuneDataset(Dataset):
    def __init__(self, planner_dataset, state_dataset):
        """
        :param planner_dataset: Instance of PushTPlannerDataset
        :param state_dataset: Instance of PushTStateDataset
        """
        super().__init__()
        self.planner_ds = planner_dataset
        self.state_ds = state_dataset
        
        # Calculate lengths for indexing logic
        self.planner_len = len(planner_dataset)
        self.state_len = len(state_dataset)

    def __len__(self):
        return self.planner_len + self.state_len

    def __getitem__(self, idx):
        if idx < self.planner_len:
            # Sample from Planner Dataset
            data = self.planner_ds[idx]
            
            # Standardization: Nest 'obs' to match the StateDataset format
            # and add placeholder for 'n_contacts' if necessary
            return {
                "obs": data["obs"],
                "action": data["action"]
            }
        else:
            # Sample from State Dataset (adjusting index)
            data = self.state_ds[idx - self.planner_len]
            
            # Ensure consistency: state_ds already returns {'obs': {'state': ...}}
            return {
                "obs": data["obs"]["state"],
                "action": data["action"],
            }

class PushTPlannerDataset(Dataset):
    """
    Wrapper dataset for `PushT` trajectory data
    """
    def __init__(self, data_dict):
        """
        :param data_array: 
        Format {"obs": (num_sample, horizon, kp_dim)}
        """
        super().__init__()
        self.data_dict = data_dict
    
    def __len__(self):
        return len(self.data_dict["obs"])
    
    def __getitem__(self, idx):
        
        this_obs = self.data_dict["obs"][idx]
        this_act = self.data_dict["action"][idx]
        sample = {
            "obs": torch.Tensor(this_obs),
            "action": torch.Tensor(this_act),
        }
        return sample

def min_max_norm(x, x_min, x_max):
    x_norm = (x - x_min) / x_max # [0,1]
    x_norm = x_norm*2 - 1 # [-1, 1]
    return x_norm

def get_pusht_goal_pose(dataset, do_normalize=True):
    env = gym.make("pusht-keypoints-v0")
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    goal_pose = info["goal_pose"]
    goal_pose = np.concatenate([np.array([0.0, 0.0]), goal_pose]) # append dummy agent pos
    if do_normalize:
        goal_pose = dataset.normalizer["obs"]["state"].normalize(goal_pose)
    goal_pose = torch.from_numpy(goal_pose)
    goal_pose = s5_to_s6(goal_pose).numpy()[2:] # only take block pose
    return goal_pose

def get_pusht_goal_keypoint(do_normalize=True):
    env = gym.make("pusht-keypoints-v0")
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    goal_pose = info["goal_pose"]
    goal_body = env.unwrapped._get_goal_pose_body(goal_pose)
    kp_map = env.kp_manager.get_keypoints_global(
        pose_map={"block": goal_body}, is_obj=True)
    # python dict guerentee order of keys and values
    kps = np.concatenate(list(kp_map.values()), axis=0)
    if do_normalize:
        kps = min_max_norm(kps, 0, 512)
    return kps

def make_info_concat_pusht(info):
    info_concat = {key: [] for key in info[0].keys()}
    for per_info in info:
        for key, val in per_info.items():
            info_concat[key].append(val)
    info_concat = {key: np.concatenate(val) for key, val in info_concat.items()}
    return info_concat

def create_pusht_zarr(episodes, file_path, use_keypoint=True):
    file_path = Path(file_path)
    root = zarr.open(file_path, mode="w")

    # Concatenate all episodes
    actions, imgs, keypoints, states, n_contacts = [], [], [], [], []
    episode_ends = []
    t_offset = 0

    for ep in episodes:
        actions.append(np.asarray(ep["action"], dtype=np.float32))
        states.append(np.asarray(ep["state"], dtype=np.float32))
        if use_keypoint:
            keypoints.append(np.asarray(ep["keypoint"], dtype=np.float32))
        n_contacts.append(np.asarray(ep["n_contacts"], dtype=np.float32))
        # If you have image observations:
        if "img" in ep:
            imgs.append(np.asarray(ep["img"], dtype=np.float32))

        t_offset += len(ep["action"])
        episode_ends.append(t_offset)

    # Stack along time
    actions = np.concatenate(actions, axis=0)
    states = np.concatenate(states, axis=0)
    if use_keypoint:
        keypoints = np.concatenate(keypoints, axis=0)
    n_contacts = np.concatenate(n_contacts, axis=0)
    if imgs:
        imgs = np.concatenate(imgs, axis=0)
    
    # Write to zarr
    data_grp = root.create_group("data")
    data_grp.create_dataset("action", data=actions, dtype="f4", compressor=None)
    if imgs:
        data_grp.create_dataset("img", data=imgs, dtype="f4", compressor=None)
    if use_keypoint:
        data_grp.create_dataset("keypoint", data=keypoints, dtype="f4", compressor=None)
    data_grp.create_dataset("state", data=states, dtype="f4", compressor=None)
    data_grp.create_dataset("n_contacts", data=n_contacts, dtype="f4", compressor=None)

    meta_grp = root.create_group("meta")
    meta_grp.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64))

    print(f"Created dataset at {file_path}")
    print(f"Total transitions: {len(actions)}, episodes: {len(episode_ends)}")


def modify_env(args, mass_scale=1.0, friction_scale=1.0):
    import tempfile
    #env = gym.make(args.task.env_name)
    env = gym.make("HalfCheetah-v3")
    this_xml = env.model.get_xml()
    this_model = mujoco_py.load_model_from_xml(this_xml)
    this_model.body_mass[1:] *= mass_scale
    this_model.body_inertia[1:, :] *= mass_scale
    this_model.geom_friction[:, :] *= friction_scale
    modified_xml = this_model.get_xml()
    temp_file = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    filename = temp_file.name
    with open(filename, "w") as file:
        file.write(modified_xml)
    file.close()
    modified_env = gym.make("HalfCheetah-v3", xml_file=filename)
    return modified_env


def create_modify_map(input_map):
    import numpy as np
    """
    Example of `input_map`
    input_map = {
    "body_mass": [0.0, 4.0, 0.5],
    "geom_friction": [0.0, 4.0, 0.5]
    }
    """
    modify_map = {}
    for (key, val) in input_map.items():
        low, high, step = val
        vals = np.arange(low, high+step, step)
        if vals[0] == 0:
            vals[0] += step*0.2
        modify_map[key] = vals
    return modify_map

def get_scaled_attr(env, attr, scale):
    current_val = getattr(env.model, attr)
    scaled_val = current_val * scale
    return scaled_val
