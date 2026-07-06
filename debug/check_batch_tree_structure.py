from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl import DataProto
import pickle
import numpy as np
import os
import re
from difflib import SequenceMatcher
import copy
from agent_system.environments.env_package.search import search_projection 
from collections import defaultdict
import random

def reassign_node_uid(batch: DataProto):
    new_node_uid = np.zeros_like(batch.non_tensor_batch['node_uid'], dtype=np.int32)
    cur_node_uid = 1
    old2new = {}
    for i in range(len(batch)):
        if old2new.get(batch.non_tensor_batch['node_uid'][i], None) is None:
            old2new[batch.non_tensor_batch['node_uid'][i]] = cur_node_uid
            cur_node_uid += 1
        new_node_uid[i] = old2new[batch.non_tensor_batch['node_uid'][i]]

    new_parent_node_uid = np.zeros_like(batch.non_tensor_batch['parent_node_uid'], dtype=np.int32)
    for i in range(len(batch)):
        new_parent_node_uid[i] = old2new.get(batch.non_tensor_batch['parent_node_uid'][i], 0)
    
    batch.non_tensor_batch['node_uid'] = new_node_uid
    batch.non_tensor_batch['parent_node_uid'] = new_parent_node_uid


def check_particuler_tree(batch: DataProto, select_uid_mode: str = "random") -> DataProto:
    if select_uid_mode is None:
        return None
    if select_uid_mode == "all" or select_uid_mode == "random" or select_uid_mode == "reward" or select_uid_mode == "info_gain":
        unique_uid = np.unique(batch.non_tensor_batch['uid'])
        if select_uid_mode == "all":
            print(f"Unique uid: {unique_uid}")
            return None
        max_group_rewards_mean = -float('inf')
        max_group_info_gain_sum_mean = -float('inf')
        select_uid_info_gain = None
        select_uid_reward = None
        for uid in unique_uid:
            group_mask = batch.non_tensor_batch['uid'] == uid
            group_rewards_mean = batch.non_tensor_batch['rewards'][group_mask].mean()
            group_info_gain_sum_mean = batch.non_tensor_batch['info_gain_sum'][group_mask].mean()
            if group_rewards_mean > max_group_rewards_mean:
                max_group_rewards_mean = group_rewards_mean
                select_uid_reward = uid
            if group_info_gain_sum_mean > max_group_info_gain_sum_mean:
                max_group_info_gain_sum_mean = group_info_gain_sum_mean
                select_uid_info_gain = uid

        select_uid = None
        if select_uid_mode == "random":
            select_uid = np.random.choice(unique_uid)
        elif select_uid_mode == "reward":
            select_uid = select_uid_reward
        elif select_uid_mode == "info_gain":
            select_uid = select_uid_info_gain
    else:
        select_uid = select_uid_mode
    print(f"Selected uid: {select_uid}")

    group_mask = batch.non_tensor_batch['uid'] == select_uid
    group_index = np.where(group_mask)[0]
    batch = batch[group_index]
    sort_index = np.argsort(batch.non_tensor_batch['traj_step'])
    batch = batch[sort_index]
    reassign_node_uid(batch)
    return batch


def change_node_reward(batch: DataProto, mask_node: np.ndarray, sim_threshold: float = 0.8) -> DataProto:
    print(f"Number of nodes to change rewards: {mask_node.astype(bool).sum()}")

    def extract_search_content(text):
        if text is None:
            return ""
        text = str(text)
        blocks = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
        if len(blocks) == 0:
            return ""
        return " ".join(b.strip() for b in blocks)

    def sim(a, b):
        return SequenceMatcher(None, a, b).ratio()

    traj_step = batch.non_tensor_batch["traj_step"]
    uid = batch.non_tensor_batch['uid']
    text_actions = batch.non_tensor_batch["text_actions"]
    search_texts = np.array([extract_search_content(x) for x in text_actions], dtype=object)

    candidate_global_mask = ~mask_node

    matched_idx = np.full(len(traj_step), -1, dtype=int)
    matched_score = np.full(len(traj_step), -1.0, dtype=float)

    target_indices = np.where(mask_node)[0]

    for i in target_indices:
        cur_step = traj_step[i]
        cur_uid = uid[i]
        cur_search = search_texts[i]

        candidate_mask = (uid == cur_uid) & (traj_step == cur_step) & candidate_global_mask
        assert not candidate_mask[i]
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            continue

        best_j = -1
        best_s = -1.0
        for j in candidate_indices:
            s = sim(cur_search, search_texts[j])
            if s > best_s:
                best_s = s
                best_j = j

        assert best_j != -1
        matched_idx[i] = best_j
        matched_score[i] = best_s

    new_batch = copy.deepcopy(batch)
    write_mask = (mask_node & (matched_idx != -1) & (matched_score > sim_threshold)).astype(bool)
    new_batch.non_tensor_batch["rewards"][write_mask] = new_batch.non_tensor_batch["rewards"][matched_idx[write_mask]]
    print(f"Number of nodes that change rewards: {write_mask.astype(bool).sum()}")
    return new_batch


def check_one_batch(select_index : int = None, select_uid_mode: str = "random"):
    local_path = copy_to_local("e.g., ~/data/Base_models/Qwen2.5-3B-Instruct", use_shm=False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)
    debug_dir = "e.g., ~/IGRPO/tmp/debug_batches/wo_think/IGRPO-3B"

    if select_index is None:
        indices = [
            int(m.group(1))
            for fname in os.listdir(debug_dir)
            if (m := re.match(r"batch_debug_after_adv_(\d+)\.pkl$", fname))
        ]
        if not indices:
            raise FileNotFoundError(f"No matching pkl files found in: {debug_dir}")
        select_index = max(indices)

    filename = f"batch_debug_after_adv_{select_index}.pkl"
    pkl_path = os.path.join(debug_dir, filename)
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    print(batch.batch.keys())
    print(batch.non_tensor_batch.keys())

    print(f"Batch size: {len(batch)}")

    mask_first_step = batch.non_tensor_batch['traj_step'] == 0
    mask_last_step = (np.asarray(batch.non_tensor_batch['traj_step']) == batch.non_tensor_batch['traj_step'].max())

    mask_answer_node = np.array([('<answer>' in x and '</answer>' in x) for x in search_projection(batch.non_tensor_batch['text_actions'])[0]], dtype=bool)
    mask_true_terminal = mask_answer_node | mask_last_step
    assert np.all(mask_true_terminal <= batch.non_tensor_batch["is_terminal"])
    print(f"mean rewards of true terminal nodes: {batch.non_tensor_batch['rewards'][mask_true_terminal].mean()}, average true terminal per group: {mask_true_terminal.sum() / len(set(batch.non_tensor_batch['uid']))}, step:{select_index}")

    mask_fake_terminal = batch.non_tensor_batch["is_terminal"] ^ mask_true_terminal
    mask_fake_terminal_first_step = mask_fake_terminal & mask_first_step
    batch_change = change_node_reward(batch, mask_fake_terminal_first_step, sim_threshold=0.8)
    print(f"mean rewards of first step nodes after change: {batch_change.non_tensor_batch["rewards"][mask_first_step].mean()}")
    
    # we find those uids: the rewards have both 0 and 1 in mask_true_terminal, which is meaningful for debug
    uid_to_rewards = defaultdict(list)
    true_terminal = np.where(mask_true_terminal)[0]
    for i in true_terminal:
        uid = batch.non_tensor_batch['uid'][i]
        reward = batch.non_tensor_batch['rewards'][i]
        uid_to_rewards[uid].append(reward)
    target_uids = []
    for uid, rewards in uid_to_rewards.items():
        rewards = np.array(rewards)
        if rewards.max() - rewards.min() > 0.5:
            target_uids.append(uid)
    if select_uid_mode == "diversity":
        if len(target_uids) > 0:
            select_uid_mode = random.choice(target_uids)
        else:
            select_uid_mode = None

    batch = check_particuler_tree(batch, select_uid_mode=select_uid_mode)
    if batch is None:
        return

    for i in range(len(batch)):
        node_uid = batch.non_tensor_batch['node_uid'][i]
        parent_node_uid = batch.non_tensor_batch['parent_node_uid'][i]
        reward = batch.non_tensor_batch['rewards'][i]
        info_gain_sum = batch.non_tensor_batch['info_gain_sum'][i]
        info_gain = batch.non_tensor_batch['info_gain'][i]
        is_terminal = batch.non_tensor_batch['is_terminal'][i]
        prompt = tokenizer.decode(batch.batch['prompts'][i], skip_special_tokens=True)
        response = batch.non_tensor_batch['text_actions'][i]
        subtree_traj_num = batch.non_tensor_batch['subtree_traj_num'][i]
        subtree_traj_num = 0
        print(
            f"Node UID: {node_uid}, Parent Node UID: {parent_node_uid}\n"
            f"Is Terminal: {is_terminal}, subtree_traj_num: {subtree_traj_num}\n"
            f"Reward: {reward}, Info Gain Sum: {info_gain_sum}, Info Gain: {info_gain}\n"
            f"Prompt: \n{prompt}\n"
            f"Response: \n{response}\n"
            f"{'-' * 100}"
        )
    print(f"ground truth: {batch[0].non_tensor_batch['ground_truth']['target'][0]}")


for i in range(5, 200, 5):
    check_one_batch(select_index=i, select_uid_mode=None)