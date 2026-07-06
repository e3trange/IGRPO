from verl import DataProto
import torch
import numpy as np
from collections import defaultdict

class TreeStructureRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        self.config = config

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        
        # since we already adjusted batch, its not a tree but a DAG
        node_uid2index_list = defaultdict(list)
        for index, data_item in enumerate(data):
            node_uid = data_item.non_tensor_batch['node_uid']
            node_uid2index_list[node_uid].append(index)

        # the reward of a node equals to the average rewards of the leaf nodes in the corresponding subtree. (since the reward of a non-leaf node is 0(search datasets))        
        data.non_tensor_batch['subtree_traj_num'] = np.zeros_like(data.non_tensor_batch['is_terminal'], dtype=np.int32)
        data.non_tensor_batch['subtree_traj_depths'] = np.zeros_like(data.non_tensor_batch['is_terminal'], dtype=np.float32)
        data.non_tensor_batch['subtree_tool_callings'] = np.zeros_like(data.non_tensor_batch['is_terminal'], dtype=np.float32)
        # WARNING!: we can not modify data_item. We use data.non_tensor_batch[xx][i] to update.
        for i in range(len(data)):
            if data.non_tensor_batch['is_terminal'][i]:
                data.non_tensor_batch['subtree_traj_num'][i] = 1
                data.non_tensor_batch['subtree_traj_depths'][i] = data.non_tensor_batch['traj_step'][i] + 1
                data.non_tensor_batch['subtree_tool_callings'][i] = data.non_tensor_batch['current_tool_callings'][i]
                cur = node_uid2index_list.get(data.non_tensor_batch['parent_node_uid'][i], [])
                # go up along the (tree)DAG edges
                while len(cur) > 0:
                    nex = []
                    for index in cur:
                        data.non_tensor_batch['subtree_traj_num'][index] += 1
                        if self.config.algorithm.igrpo.reward_mode == "avg":
                            data.non_tensor_batch['rewards'][index] += data.non_tensor_batch['rewards'][i]
                        elif self.config.algorithm.igrpo.reward_mode == "max":
                            data.non_tensor_batch['rewards'][index] = max(data.non_tensor_batch['rewards'][index], data.non_tensor_batch['rewards'][i])
                        else:
                            raise NotImplementedError
                        data.non_tensor_batch['subtree_traj_depths'][index] += data.non_tensor_batch['subtree_traj_depths'][i]
                        data.non_tensor_batch['subtree_tool_callings'][index] += data.non_tensor_batch['subtree_tool_callings'][i]
                        for nex_index in node_uid2index_list.get(data.non_tensor_batch['parent_node_uid'][index], []):
                            nex.append(nex_index)
                    cur = nex

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']

            assert data_item.non_tensor_batch['subtree_traj_num'] > 0, f"traj_step: {data_item.non_tensor_batch['traj_step']} & node_uid: {data_item.non_tensor_batch['node_uid']}"
            if self.config.algorithm.igrpo.reward_mode == "avg":
                data.non_tensor_batch['rewards'][i] /= data_item.non_tensor_batch['subtree_traj_num']
            data.non_tensor_batch['subtree_traj_depths'][i] /= data_item.non_tensor_batch['subtree_traj_num']
            data.non_tensor_batch['subtree_tool_callings'][i] /= data_item.non_tensor_batch['subtree_traj_num']
            data_item = data[i] # retake

            reward = data_item.non_tensor_batch['rewards']
            length = data_item.non_tensor_batch['subtree_traj_depths']
            if self.normalize_by_length:
                score = reward / length
            else:
                score = reward
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
