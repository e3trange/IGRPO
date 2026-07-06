from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from igpo.core_igpo import compute_answer_block_avg_log_prob, metric_update, get_step_rewards
from verl import DataProto
import pickle
import torch
import numpy as np
import os
import re
import glob
from collections import Counter

def check_all_batch():
    dir_path = "e.g., ~/IGRPO/tmp/debug_batches/wo_think/igpo-3B"
    pattern = os.path.join(dir_path, "batch_debug_after_adv_*.pkl")

    files = glob.glob(pattern)

    def extract_idx(path):
        name = os.path.basename(path)
        m = re.match(r"batch_debug_after_adv_(\d+)\.pkl$", name)
        return int(m.group(1)) if m else float("inf")

    files = sorted(files, key=extract_idx)
    for pkl_path in files:
        idx = extract_idx(pkl_path)
        if idx > 100:
            continue
        with open(pkl_path, "rb") as f:
            batch = pickle.load(f)
        print(metric_update(get_step_rewards(batch, prob_diff_mode=True), reward_mask_mode="ones", traj_step_mask_mode=0))
        
        


def sort_batch(batch: DataProto) -> list:
    """
    batch: dict of lists / tensors
    return: batch, sorted_indices
    """
    uids = batch.non_tensor_batch["uid"]
    traj_uids = batch.non_tensor_batch["traj_uid"]
    traj_steps = batch.non_tensor_batch["traj_step"]
    bs = len(uids)

    sorted_indices = sorted(
        range(bs),
        key=lambda i: (uids[i], traj_uids[i], traj_steps[i])
    )
    sorted_indices_np = np.array(sorted_indices, dtype=np.int64)

    for k, v in batch.batch.items():
        if isinstance(v, torch.Tensor):
            batch.batch[k] = v[sorted_indices]
        else:
            try:
                batch.batch[k] = [v[i] for i in sorted_indices]
            except Exception:
                raise TypeError(f"batch.batch[{k}] is not sortable, type={type(v)}")
            
    for k, v in batch.non_tensor_batch.items():
        if isinstance(v, torch.Tensor):
            batch.non_tensor_batch[k] = v[sorted_indices]
        elif isinstance(v, list):
            batch.non_tensor_batch[k] = [v[i] for i in sorted_indices]
        elif isinstance(v, np.ndarray):
            batch.non_tensor_batch[k] = v[sorted_indices_np]
        else:
            try:
                batch.non_tensor_batch[k] = [v[i] for i in sorted_indices]
            except Exception:
                raise TypeError(f"batch.non_tensor_batch[{k}] is not sortable, type={type(v)}")

    return sorted_indices


def get_target_trajectory(batch: DataProto, win: float  = 1.0, len: int = 4, select_index: int = 0) -> list:
    """
    batch: dict of lists / tensors
    return: target_indices for debug
    """
    won_indices = [
        i for i, x in enumerate(batch.non_tensor_batch["episode_rewards"])
        if x == win
    ]

    max_len = max(batch.non_tensor_batch["episode_lengths"][i] for i in won_indices)
    assert len <= max_len, f"Selected length must less than or equal to max_len{max_len}."
    len_indices = [
        i for i in won_indices
        if batch.non_tensor_batch["episode_lengths"][i] == len
    ]

    target_traj_uid = batch.non_tensor_batch["traj_uid"][len_indices[select_index]]
    target_indices = [
        i for i, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"])
        if traj_uid == target_traj_uid
    ]
    return target_indices

def get_target_group(batch: DataProto, uid: str):
    return [
        i for i, x in enumerate(batch.non_tensor_batch["uid"])
        if x == uid
    ]
    
    
def check_one_batch(select_index : int = 10):
    local_path = copy_to_local("e.g., ~/data/Base_models/Qwen2.5-3B-Instruct", use_shm=False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)

    filename = f"batch_debug_after_adv_{select_index}.pkl"
    pkl_path = "e.g., ~/IGRPO/tmp/debug_batches/wo_think/igrpo-3B/" + filename
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    print(batch.batch.keys())
    print(batch.non_tensor_batch.keys())

    if "avg_ans_log_probs" in batch.batch.keys():
        get_step_rewards(batch=batch, prob_diff_mode=True)

    if "traj_uid" in batch.non_tensor_batch.keys() and "episode_lengths" in batch.non_tensor_batch.keys():
        unique_index = np.unique(batch.non_tensor_batch["traj_uid"], return_index=True)[1]
        print("Unique traj num:", len(unique_index))
        length_counter = Counter(batch.non_tensor_batch['episode_lengths'][unique_index])
        print(f"Trajectory length distribution: {length_counter}")
        mean_length = sum(int(k) * v for k, v in length_counter.items()) / sum(length_counter.values())
        print(f"Mean trajectory length: {mean_length}")

    sort_batch(batch=batch)

    target_indices = get_target_trajectory(batch=batch, win=1.0, len=4, select_index=0)
    target_indices = get_target_group(batch, batch.non_tensor_batch['uid'][target_indices[0]])
    for i in target_indices:
        traj_step = batch.non_tensor_batch["traj_step"][i]
        traj_uid = batch.non_tensor_batch["traj_uid"][i]
        traj_reward = batch.non_tensor_batch["episode_rewards"][i]
        prompt_ids = batch.batch["prompts"][i]
        response_ids = batch.batch["responses"][i]
        avg_ans_log_probs = batch.batch["avg_ans_log_probs"][i]
        step_rewards = batch.batch["step_rewards"][i]

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        print("-" * 100)
        print(f"traj_step: {traj_step}, traj_uid: {traj_uid}, traj_reward: {traj_reward}")
        print(f"avg_ans_log_probs: {avg_ans_log_probs}")
        print(f"step_rewards: {step_rewards}")
        print("\n[PROMPT]")
        print(prompt_text)
        print("\n[RESPONSE]")
        print(response_text)
        print("-" * 100)    

    compute_answer_block_avg_log_prob(batch, tokenizer, None, debug=5)

    print(metric_update(get_step_rewards(batch, prob_diff_mode=True)))
    

check_all_batch()