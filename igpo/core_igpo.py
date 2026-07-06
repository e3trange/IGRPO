from collections import defaultdict
import copy
from typing import Any
import torch

import numpy as np
from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from agent_system.environments.env_package.search.projection import think_prefix_projection


BEGIN_ANS = "<answer>\n"
END_ANS = "\n</answer>"
ANSWER_TEMPLATE = """\nNow there's enough information to answer\n</think>
{begin_ans}{ground_truth}{end_ans}"""


def find_answer_interval(text: str, start_tag: str, end_tag: str) -> tuple[int, int]:

    start_pos = text.rfind(start_tag)
    if start_pos == -1:
        raise ValueError(f"No {start_tag} tag found.")

    x = start_pos + len(start_tag) - 1

    end_pos = text.find(end_tag, x + 1)
    if end_pos == -1:
        raise ValueError(f"No matching {end_tag} tag found after the last {start_tag}.")

    y = end_pos
    return x + 1, y


def compute_answer_block_avg_log_prob(batch: DataProto, tokenizer, actor_rollout_wg, 
                                      debug: int = -1, 
                                      think: bool = True, 
                                      response_length: int = None) -> DataProto:
    """
        Combine the prompts in batch with responses before </think> + ANSWER_TEMPLATE, to get avg ans log_prob
        of the <answer></answer> block
    """
    
    select_keys = ["prompts"] # left padding
    new_batch = copy.deepcopy(batch.select(batch_keys=select_keys))
    device = batch.batch["prompts"].device

    if response_length is None:
        response_length = batch.batch["responses"].shape[1]
    

    prompts = new_batch.batch["prompts"]
    bs, prompt_length = prompts.shape

    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id are both None")

    ground_truth_targets = [x["target"] for x in batch.non_tensor_batch["ground_truth"]]
    # WITHOUT </think>
    if think:
        text_think = think_prefix_projection(batch.non_tensor_batch["text_actions"])[0]
    else:
        text_think = ["" for _ in range(bs)]

    responses_list = []
    response_attention_mask_list = []
    gt_token_indices_list = []

    for i in range(bs):
        if i < debug:
            print("=" * 100)

        gt = ground_truth_targets[i][0]
        text_ans = ANSWER_TEMPLATE.format(ground_truth=gt, begin_ans=BEGIN_ANS, end_ans=END_ANS)
        text = text_think[i] + text_ans if think else "<think>" + text_ans
        if i < debug:
            print(f"text[{i}]:\n{text}")

        check_response_length_flag = False
        while True:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            responses = encoded["input_ids"][0]
            # add eos_id to the end of responses
            responses = torch.cat([responses, torch.tensor([eos_id], device=responses.device, dtype=responses.dtype)], dim=0)
            if responses.shape[0] > response_length:
                text = "<think>" + text_ans
                assert not check_response_length_flag, f"<think> + text_ans{text_ans} exceeds response_length!"
                check_response_length_flag = True
            else:
                # pad responses to response_length
                if responses.shape[0] < response_length:
                    pad_len = response_length - responses.shape[0]
                    responses = torch.cat(
                        [
                            responses,
                            torch.full((pad_len,), pad_id, device=responses.device, dtype=responses.dtype),
                        ],
                        dim=0,
                    )
                break

        offsets = encoded["offset_mapping"][0].tolist()
        gt_interval = find_answer_interval(
            text=text,
            start_tag=BEGIN_ANS,
            end_tag=END_ANS,
        )
        if i < debug:
            print(f"gt_interval[{i}]: {gt_interval}, ground_truth: {text[gt_interval[0]: gt_interval[1]]}")

        gt_token_indices = []
        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if max(s, gt_interval[0]) < min(e, gt_interval[1]):
                gt_token_indices.append(idx)
        if len(gt_token_indices) == 0:
            raise ValueError(
                f"No token fully contained in ground_truth span: "
                f"char_span=({gt_interval[0]}, {gt_interval[1]}), ground_truth={gt!r}"
            )
        gt_token_indices_list.append(gt_token_indices)

        if i < debug:
            print(f"gt_token_indices[{i}]: {gt_token_indices} ground_truth: {tokenizer.decode(responses[gt_token_indices], skip_special_tokens=True)}")

        response_attention_mask = get_response_mask(response_id=responses.unsqueeze(0), eos_token=eos_id, dtype=batch.batch['attention_mask'].dtype)[0]
        responses_list.append(responses)
        response_attention_mask_list.append(response_attention_mask)
                    
    responses = torch.stack(responses_list, dim=0).to(device=device) # [bs, response_length]
    response_attention_mask = torch.stack(response_attention_mask_list, dim=0).to(device=device)

    input_ids = torch.cat([prompts, responses], dim=1)   # [bs, prompt_length + response_length]
    prompt_attention_mask = batch.batch["attention_mask"][:, :prompt_length].to(
        device=response_attention_mask.device, dtype=response_attention_mask.dtype
    )

    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1) # [bs, prompt_length + response_length]
    
    prompt_position_ids = batch.batch["position_ids"][:, :prompt_length]
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device, dtype=prompt_position_ids.dtype,
    ).unsqueeze(0).expand(bs, -1)
    response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    
    new_batch.batch["responses"] = responses
    new_batch.batch["response_mask"] = response_attention_mask
    new_batch.batch["input_ids"] = input_ids
    new_batch.batch["attention_mask"] = attention_mask
    new_batch.batch["position_ids"] = position_ids

    if debug >= 0:
        print(f"=" * 100)
        print(position_ids[0][-5-response_length: prompt_length])
        print(position_ids[0][-response_length: -response_length+5])
        print(response_attention_mask[0][gt_token_indices_list[0]])
        return

    old_log_prob: DataProto = actor_rollout_wg.compute_log_prob(new_batch)
    del new_batch
    response_log_prob: torch.Tensor = old_log_prob.batch.pop("old_log_probs")
    del old_log_prob

    avg_log_probs = []
    for i in range(bs):
        gt_token_indices = gt_token_indices_list[i]
        gt_log_probs = response_log_prob[i, gt_token_indices]
        avg_log_probs.append(gt_log_probs.mean())

    response_log_prob_device, response_log_prob_dtype = response_log_prob.device, response_log_prob.dtype
    del response_log_prob
    avg_log_probs = torch.stack(avg_log_probs, dim=0).to(device=response_log_prob_device, dtype=response_log_prob_dtype)
    batch.batch["avg_ans_log_probs"] = avg_log_probs
    return batch


def metric_update(batch: DataProto, reward_mask_mode: str = "all", traj_step_mask_mode: int = None) -> dict[str, Any]:
    step_rewards : torch.Tensor = batch.batch["step_rewards"]
    is_terminal : np.ndarray = batch.non_tensor_batch["is_terminal"]
    is_terminal_tensor = torch.as_tensor(is_terminal, dtype=torch.bool, device=step_rewards.device)
    non_terminal_mask = ~is_terminal_tensor
    terminal_mask = is_terminal_tensor

    episode_rewards = batch.non_tensor_batch["episode_rewards"]
    episode_rewards_tensor = torch.as_tensor(np.asarray(episode_rewards, dtype=np.float32), device=step_rewards.device)
    if reward_mask_mode == "ones":
        mask = (episode_rewards_tensor == 1.0)
    elif reward_mask_mode == "zeros":
        mask = (episode_rewards_tensor == 0.0)
    elif reward_mask_mode == "all":
        mask = torch.ones_like(step_rewards, dtype=torch.bool)
    else:
        raise ValueError(
            f"Unsupported reward_mask_mode: {reward_mask_mode}. "
            f"Expected one of ['ones', 'zeros', 'all']."
        )

    if traj_step_mask_mode is not None:
        traj_steps = batch.non_tensor_batch["traj_step"]
        traj_steps_tensor = torch.as_tensor(np.asarray(traj_steps, dtype=np.int64), device=step_rewards.device)
        traj_step_mask = (traj_steps_tensor <= traj_step_mask_mode)
        mask = mask & traj_step_mask
        step_mask_mode = f"step_{traj_step_mask_mode}"
    else:
        step_mask_mode = "step_all"

    traj_uids = batch.non_tensor_batch["traj_uid"]
    traj_uids_np = np.asarray(traj_uids)

    if mask.any():
        num_unique_traj = len(np.unique(traj_uids_np[mask.cpu().numpy()]))
        mean_step_rewards_igpo = step_rewards[mask].sum().item() / num_unique_traj
    else:
        num_unique_traj = 0
        mean_step_rewards_igpo = 0.0

    non_terminal_mask = non_terminal_mask & mask
    if non_terminal_mask.any():
        mean_info_gains_igpo = step_rewards[non_terminal_mask].sum().item() / num_unique_traj
    else:
        mean_info_gains_igpo = 0.0

    terminal_mask = terminal_mask & mask
    if terminal_mask.any():
        mean_episode_rewards_igpo = step_rewards[terminal_mask].sum().item() / num_unique_traj
    else:
        mean_episode_rewards_igpo = 0.0

    metrics = {
        f"episode/mean_info_gains_igpo_{reward_mask_mode}_{step_mask_mode}": mean_info_gains_igpo,
        # f"episode/mean_step_rewards_igpo_{reward_mask_mode}_{step_mask_mode}": mean_step_rewards_igpo,
        # f"episode/mean_episode_rewards_igpo_{reward_mask_mode}_{step_mask_mode}": mean_episode_rewards_igpo,
    }

    return metrics


def get_step_rewards(batch: DataProto, prob_diff_mode : bool = False) -> DataProto:
    avg_log_probs : torch.Tensor = batch.batch["avg_ans_log_probs"]
    device, dtype = avg_log_probs.device, avg_log_probs.dtype
    bs = avg_log_probs.shape[0]
    traj_uids = batch.non_tensor_batch["traj_uid"]
    traj_steps = batch.non_tensor_batch["traj_step"]

    # (traj_uid, traj_step) -> index
    traj_to_index = {}
    for i in range(bs):
        traj_uid = traj_uids[i]
        traj_step = int(traj_steps[i])
        key = (traj_uid, traj_step) # due to adjust_batch, there may be same trajs
        traj_to_index[key] = i

    episode_rewards = batch.batch["token_level_scores"].sum(dim=-1).to(device=device, dtype=dtype)
    step_rewards = torch.zeros(bs, device=device, dtype=dtype)
    is_terminal = np.zeros(bs, dtype=bool)
    for i in range(bs):
        traj_uid = traj_uids[i]
        traj_step = int(traj_steps[i])
        next_key = (traj_uid, traj_step + 1)
        if next_key in traj_to_index:
            j = traj_to_index[next_key]
            if prob_diff_mode:
                step_rewards[i] = torch.exp(avg_log_probs[j]) - torch.exp(avg_log_probs[i])
            else:
                step_rewards[i] = avg_log_probs[j] - avg_log_probs[i]
        else:
            step_rewards[i] = episode_rewards[i]
            is_terminal[i] = True

    batch.batch["step_rewards"] = step_rewards
    batch.non_tensor_batch["is_terminal"] = is_terminal
    
    return batch


def compute_igpo_step_rewards(batch: DataProto, tokenizer, actor_rollout_wg, prob_diff_mode: bool = False, think: bool = True) -> tuple[DataProto, dict[str, Any]]:
    batch = compute_answer_block_avg_log_prob(batch=batch, tokenizer=tokenizer, actor_rollout_wg=actor_rollout_wg, think=think) # [bs]

    batch = get_step_rewards(batch, prob_diff_mode=prob_diff_mode)
    
    return batch


def grpo_style_group_computation(uid2score, uid2mean, uid2std):
    for uid in uid2score:
        if len(uid2score[uid]) == 1:
            uid2mean[uid] = torch.tensor(0.0)
            uid2std[uid] = torch.tensor(1.0)
        elif len(uid2score[uid]) > 1:
            uid2mean[uid] = torch.mean(torch.tensor(uid2score[uid]))
            uid2std[uid] = torch.std(torch.tensor([uid2score[uid]]))
        else:
            raise ValueError(f"no score in prompt index: {uid}")


def compute_turn_level_discounted_rewards(step_rewards: torch.Tensor,
                                          traj_uids: np.ndarray,
                                          traj_steps: np.ndarray,
                                          gamma: float):
    bs = step_rewards.shape[0]
    traj_uid_to_steps = defaultdict(list) # traj_uid -> list of (traj_step, index)
    for i in range(bs):
        traj_uid = traj_uids[i]
        traj_step = int(traj_steps[i])
        traj_uid_to_steps[traj_uid].append((traj_step, i))
    
    for traj_uid in traj_uid_to_steps:
        traj_uid_to_steps[traj_uid].sort(reverse=True)

        running = 0.0
        last_traj_step = None
        current_step_value = None

        for traj_step, idx in traj_uid_to_steps[traj_uid]:
            if traj_step != last_traj_step:
                current_step_value = step_rewards[idx] + gamma * running
                running = current_step_value
                last_traj_step = traj_step

            step_rewards[idx] = current_step_value


def compute_igpo_outcome_advantage(step_rewards: torch.Tensor, 
                                   response_mask: torch.Tensor, 
                                   uid: np.ndarray,
                                   traj_uid: np.ndarray,
                                   traj_step: np.ndarray,
                                   is_terminal: np.ndarray,
                                   gamma: float,
                                   eps: float = 1e-6):
    uid2score = defaultdict(list)
    uid2mean = {}
    uid2std = {}
    uid2score_terminal = defaultdict(list)
    uid2mean_terminal = {}
    uid2std_terminal = {}
    bs = step_rewards.shape[0]

    with torch.no_grad():
        # GRPO style advantage computation first
        for i in range(bs):
            if is_terminal[i]:
                uid2score_terminal[uid[i]].append(step_rewards[i])
            else:
                uid2score[uid[i]].append(step_rewards[i])
        grpo_style_group_computation(uid2score, uid2mean, uid2std)
        grpo_style_group_computation(uid2score_terminal, uid2mean_terminal, uid2std_terminal)
        for i in range(bs):
            if is_terminal[i]:
                step_rewards[i] = (step_rewards[i] - uid2mean_terminal[uid[i]]) / (uid2std_terminal[uid[i]] + eps)
            else:
                step_rewards[i] = (step_rewards[i] - uid2mean[uid[i]]) / (uid2std[uid[i]] + eps)

        # turn-level discounted
        compute_turn_level_discounted_rewards(
            step_rewards=step_rewards,
            traj_uids=traj_uid,
            traj_steps=traj_step,
            gamma=gamma
        )

        advantages = step_rewards.unsqueeze(-1) * response_mask
        
    return advantages, advantages
