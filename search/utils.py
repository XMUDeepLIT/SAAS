import re
from typing import Tuple, List
import numpy as np
from slime.utils.types import Sample
from collections import defaultdict
from typing import Any
import logging
import math

logger = logging.getLogger(__name__)

def calculate_score_smooth_label_V1_linear(
    current_query_num: float, 
    query_num_thre: float, 
    score_i: float, 
    beta: float = 0.05, 
    label: str = "easy"
) -> float:
    # === 1. Easy mode: zero tolerance ===
    if label == "easy":
        if score_i == 1:
            return score_i - beta * current_query_num
        else:
            return score_i

    # === 2. Middle mode: only require efficiency for successful cases ===
    elif label == "middle":
        if score_i == 1:
            # Logic: penalize for searching more than thre, reward for searching less (advantage is negative, subtracting negative = adding)
            advantage = current_query_num - query_num_thre
            advantage = max(0, advantage)
            efficiency_adjustment = beta * advantage
            return score_i - efficiency_adjustment
        else:
            return score_i

    return score_i

def calculate_score_smooth_label_V1_middle_linear(
    current_query_num: float, 
    query_num_thre: float, 
    score_i: float, 
    beta: float = 0.05, 
    label: str = "easy"
) -> float:
    # === 1. Easy mode: zero tolerance ===
    if label == "easy":
        return score_i

    # === 2. Middle mode: only require efficiency for successful cases ===
    elif label == "middle":
        if score_i == 1:
            # Logic: penalize for searching more than thre, reward for searching less (advantage is negative, subtracting negative = adding)
            advantage = current_query_num - query_num_thre
            advantage = max(0, advantage)
            efficiency_adjustment = beta * advantage
            return score_i - efficiency_adjustment
        else:
            return score_i

    return score_i


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def format_reward(response: str) -> float:
    """
    1. All tags must conform to rules (within predefined tags)
    2. Content within all tags must not be empty
    3. First two tags must be think, last two tags must be answer
    4. After tool_call, the next two tags must be tool_response and /tool_response
    5. Number of tool_call tags must equal number of tool_response tags
    6. Number of think tags must equal number of tool_call tags + 1
    """

    response = response.strip()

    # Check if any tag content contains disallowed tags
    allowed_tags = {'think', 'tool_call', 'tool_response', 'answer', '/think', '/tool_call', '/tool_response', '/answer'}
    all_tags = re.findall(r'<([^>]+)>', response)
    for tag in all_tags:
        if tag not in allowed_tags:
            return False

    # Must start with <think> and end with </answer>
    # if not response.startswith('<think>') or not response.endswith('</answer>'):
    #     return False

    # Extract all tags in order
    tags = re.findall(r'<(/?(?:think|tool_call|tool_response|answer))>', response)

    # At least 4 tags required: <think></think><answer></answer>
    if len(tags) < 4:
        return False

    # Check if any tag content is empty
    tag_contents = {
        'think': re.findall(r'<think>(.*?)</think>', response, re.DOTALL),
        'tool_call': re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL),
        'tool_response': re.findall(r'<tool_response>(.*?)</tool_response>', response, re.DOTALL),
        'answer': re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL),
    }


    # Return 0 if any tag has empty content
    for tag_type, contents in tag_contents.items():
        for content in contents:
            if not content.strip():
                return False
        
    # Check structure
    if tags[0] != 'think' or tags[1] != '/think':
        return False
    
    if tags[-2] != 'answer' or tags[-1] != '/answer':
        return False

    # Check search-information pairing in the middle
    middle_tags = tags[2:-2] # Exclude initial think and final answer

    i = 0
    while i < len(middle_tags):
        if middle_tags[i] == 'tool_call':
            # Must be followed by /search, information, /information
            if (i + 3 >= len(middle_tags) or middle_tags[i + 1] != '/tool_call' or middle_tags[i + 2] != 'tool_response' or middle_tags[i + 3] != '/tool_response'):
                return False
            i += 4
        else:
            i += 1
    think_num = response.count('<think>')
    search_num = response.count('<tool_call>')
    information_num = response.count('<tool_response>')
    if search_num != information_num:
        return False

    if think_num != search_num + 1:
        return False
    
    return True

def _kmp_prefix_function_fast(s: str) -> np.ndarray:
    """KMP prefix function (NumPy implementation, suitable for length <= 8192)"""
    arr = np.frombuffer(s.encode('utf-8', 'ignore'), dtype=np.uint8)
    n = len(arr)
    pi = np.zeros(n, dtype=np.int32)
    j = 0
    for i in range(1, n):
        while j > 0 and arr[i] != arr[j]:
            j = pi[j - 1]
        if arr[i] == arr[j]:
            j += 1
        pi[i] = j
    return pi


def _find_tandem_repeats_fast(text: str, min_repeat_len: int = 4) -> List[Tuple[int, int]]:
    """Detect all consecutive repeated segments (based on KMP prefix function)"""
    n = len(text)
    pi = _kmp_prefix_function_fast(text)
    spans = []
    for i in range(n):
        L = i + 1
        p = L - pi[i]
        if pi[i] > 0 and p >= min_repeat_len and L % p == 0:
            k = L // p
            if k >= 2:
                spans.append((L - p * k, L))
    # Merge adjacent intervals
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def calc_repetition_rate(text: str, min_repeat_len: int = 4) -> Tuple[float, List[Tuple[int, int, str]]]:
    """
    Fast version: suitable for text length <= 8192.
    Global detection + merge repeated segments + longest block ratio.
    """
    text = _normalize_text(text)
    n = len(text)
    if n < 2 * min_repeat_len:
        return 0.0, []

    spans = _find_tandem_repeats_fast(text, min_repeat_len)
    if not spans:
        return 0.0, []

    # Merge and extract samples
    merged = []
    s, e = spans[0]
    for s2, e2 in spans[1:]:
        if s2 <= e:
            e = max(e, e2)
        else:
            merged.append((s, e))
            s, e = s2, e2
    merged.append((s, e))

    total_cov = sum(e - s for s, e in merged)
    max_block = max(e - s for s, e in merged)
    rate = 1 - (1 - total_cov / n) * (1 - max_block / n)

    samples = []
    for s, e in sorted(merged, key=lambda x: x[1] - x[0], reverse=True)[:5]:
        snippet = text[s:e]
        if len(snippet) > 200:
            snippet = snippet[:100] + " … " + snippet[-100:]
        samples.append((int(s), int(e), str(snippet)))

    return float(round(rate, 6)), samples


def  get_metrics(args, samples, evaluation: bool):
    reward_metrics={}
    keys = ["exact_match", "f1", "query_num", "is_format", "is_entail"]
    data_source_lst = [sample.metadata['data_source'] for sample in samples]
    exact_match_lst = [sample.exact_match for sample in samples]
    uid_lst = [sample.group_index for sample in samples]
    for key in keys:
        ds_to_value = defaultdict(list)
        if key in ["turn_num", "query_num", "response_length", "lap_ratio", "is_format"]:
            if key == "is_format":
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]

            ds_to_correct_value = defaultdict(list)
            ds_to_failed_value = defaultdict(list)

            assert len(data_source_lst) == len(val_lst) == len(exact_match_lst), f"len(data_source_lst): {len(data_source_lst)}, len(val_lst): {len(val_lst)}, len(exact_match_lst): {len(exact_match_lst)}"
            for ds, val, exact_match in zip(data_source_lst, val_lst, exact_match_lst):
                ds_to_value[ds].append(val)
                if exact_match:
                    ds_to_correct_value[ds].append(val)
                else:
                    ds_to_failed_value[ds].append(val)
            for ds, value_lst in ds_to_correct_value.items():
                reward_metrics[f'{ds}/{key}/correct'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 
            for ds, value_lst in ds_to_failed_value.items():
                reward_metrics[f'{ds}/{key}/failed'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 
            for ds, value_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}/all'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 

        elif key in ["exact_match"]:
            val_lst = [int(getattr(sample, key)) for sample in samples]
            ds_uid_to_acc = defaultdict(lambda: defaultdict(list))
            for uid, val, data_source in zip(uid_lst, val_lst, data_source_lst):
                ds_uid_to_acc[data_source][uid].append(val)
                ds_to_value[data_source].append(val)
            for ds, val_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0 # reward metrics 
                if evaluation:
                    reward_metrics[f'exact_match/{ds}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0
            for ds, uid_to_acc_lst in ds_uid_to_acc.items():
                all_EM = 0
                all_not_EM = 0
                mid_EM = 0
                for uid, acc_lst in uid_to_acc_lst.items():
                    if sum(acc_lst) == args.n_samples_per_prompt:
                        all_EM += 1
                    elif sum(acc_lst) == 0:
                        all_not_EM += 1
                    else:
                        mid_EM += 1
                reward_metrics[f'{ds}/all_EM'] = all_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 
                reward_metrics[f'{ds}/all_not_EM'] = all_not_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 
                reward_metrics[f'{ds}/mid_EM'] = mid_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 

        else:
            if key in ["invalid_analysis", "invaild_action_cnt", "is_entail"]:
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]
            for ds, val in zip(data_source_lst, val_lst):
                ds_to_value[ds].append(val)
            for ds, val_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0 # reward metrics 
                if key == "f1" and evaluation:
                    reward_metrics[f'f1/{ds}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0

    
    if not evaluation:
        acc_lst = [int(getattr(sample, "exact_match")) for sample in samples]
        all_EM = []
        all_not_EM = []
        mid_EM = []

        uid_to_EM = defaultdict(int)
        for uid, acc in zip(uid_lst, acc_lst):
            uid_to_EM[uid] += acc

        for item in uid_to_EM.items():
            if item[1] == args.n_samples_per_prompt:
                all_EM.append(1)
                all_not_EM.append(0)
                mid_EM.append(0)
            elif item[1] == 0:
                all_not_EM.append(1)
                all_EM.append(0)
                mid_EM.append(0)
            else:
                all_EM.append(0)
                all_not_EM.append(0)
                mid_EM.append(1)
        reward_metrics['tool_reward_info/all_EM'] = sum(all_EM) / len(all_EM) if len(all_EM) > 0 else 0
        reward_metrics['tool_reward_info/all_not_EM'] = sum(all_not_EM) / len(all_not_EM) if len(all_not_EM) > 0 else 0
        reward_metrics['tool_reward_info/mid_EM'] = sum(mid_EM) / len(mid_EM) if len(mid_EM) > 0 else 0
    
        keys = ["exact_match", "f1", "query_num", "is_format", "is_entail"]
        for key in keys:
            if key in ["exact_match", "is_format", "invalid_analysis", "is_entail"]:
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]
            reward_metrics[f'tool_reward_info/{key}'] = sum(val_lst) / len(val_lst) # reward metrics 

    return reward_metrics

def get_metrics_ggpo(args, samples, evaluation: bool):
    reward_metrics={}
    keys = ["exact_match", "f1", "query_num", "is_format", "is_entail"]
    data_source_lst = [sample.metadata['data_source'] for sample in samples]
    exact_match_lst = [sample.exact_match for sample in samples]
    uid_lst = [sample.group_index for sample in samples]
    
    # Get half point for separating first half and second half
    n_samples_per_prompt = getattr(args, 'n_samples_per_prompt', 8)
    half_point = n_samples_per_prompt // 2
    
    # Determine which samples belong to first half vs second half based on index
    # First n_samples_per_prompt samples belong to first group, next n_samples_per_prompt to second group, etc.
    # Within each group, first half_point samples are "first half", remaining are "second half"
    first_half_mask = [(sample.in_group_idx % n_samples_per_prompt) < half_point for sample in samples]
    
    for key in keys:
        ds_to_value = defaultdict(list)
        if key in ["turn_num", "query_num", "response_length", "lap_ratio", "is_format"]:
            if key == "is_format":
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]

            ds_to_correct_value = defaultdict(list)
            ds_to_failed_value = defaultdict(list)
            ds_to_value_first_half = defaultdict(list)
            ds_to_value_second_half = defaultdict(list)
            ds_to_correct_value_first_half = defaultdict(list)
            ds_to_failed_value_first_half = defaultdict(list)
            ds_to_correct_value_second_half = defaultdict(list)
            ds_to_failed_value_second_half = defaultdict(list)

            assert len(data_source_lst) == len(val_lst) == len(exact_match_lst), f"len(data_source_lst): {len(data_source_lst)}, len(val_lst): {len(val_lst)}, len(exact_match_lst): {len(exact_match_lst)}"
            for i, (ds, val, exact_match) in enumerate(zip(data_source_lst, val_lst, exact_match_lst)):
                ds_to_value[ds].append(val)
                if exact_match:
                    ds_to_correct_value[ds].append(val)
                else:
                    ds_to_failed_value[ds].append(val)
                # Separate by first half and second half
                if first_half_mask[i]:
                    ds_to_value_first_half[ds].append(val)
                    if exact_match:
                        ds_to_correct_value_first_half[ds].append(val)
                    else:
                        ds_to_failed_value_first_half[ds].append(val)
                else:
                    ds_to_value_second_half[ds].append(val)
                    if exact_match:
                        ds_to_correct_value_second_half[ds].append(val)
                    else:
                        ds_to_failed_value_second_half[ds].append(val)
            for ds, value_lst in ds_to_correct_value.items():
                reward_metrics[f'{ds}/{key}/correct'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 
            for ds, value_lst in ds_to_failed_value.items():
                reward_metrics[f'{ds}/{key}/failed'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 
            for ds, value_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}/all'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0 # reward metrics 
            # First half metrics
            for ds, value_lst in ds_to_correct_value_first_half.items():
                reward_metrics[f'{ds}/{key}/first_half_correct'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0
            for ds, value_lst in ds_to_failed_value_first_half.items():
                reward_metrics[f'{ds}/{key}/first_half_failed'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0
            for ds, value_lst in ds_to_value_first_half.items():
                reward_metrics[f'{ds}/{key}/first_half_all'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0
            # Second half metrics
            for ds, value_lst in ds_to_correct_value_second_half.items():
                reward_metrics[f'{ds}/{key}/second_half_correct'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0
            for ds, value_lst in ds_to_failed_value_second_half.items():
                reward_metrics[f'{ds}/{key}/second_half_failed'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0
            for ds, value_lst in ds_to_value_second_half.items():
                reward_metrics[f'{ds}/{key}/second_half_all'] = sum(value_lst) / len(value_lst) if len(value_lst) > 0 else 0

        elif key in ["exact_match"]:
            val_lst = [int(getattr(sample, key)) for sample in samples]
            ds_uid_to_acc = defaultdict(lambda: defaultdict(list))
            ds_uid_to_acc_first_half = defaultdict(lambda: defaultdict(list))
            ds_uid_to_acc_second_half = defaultdict(lambda: defaultdict(list))
            for i, (uid, val, data_source) in enumerate(zip(uid_lst, val_lst, data_source_lst)):
                ds_uid_to_acc[data_source][uid].append(val)
                ds_to_value[data_source].append(val)
                # Separate by first half and second half
                if first_half_mask[i]:
                    ds_uid_to_acc_first_half[data_source][uid].append(val)
                else:
                    ds_uid_to_acc_second_half[data_source][uid].append(val)
            for ds, val_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0 # reward metrics 
                if evaluation:
                    reward_metrics[f'exact_match/{ds}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0
            for ds, uid_to_acc_lst in ds_uid_to_acc.items():
                all_EM = 0
                all_not_EM = 0
                mid_EM = 0
                for uid, acc_lst in uid_to_acc_lst.items():
                    if sum(acc_lst) == n_samples_per_prompt:
                        all_EM += 1
                    elif sum(acc_lst) == 0:
                        all_not_EM += 1
                    else:
                        mid_EM += 1
                reward_metrics[f'{ds}/all_EM'] = all_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 
                reward_metrics[f'{ds}/all_not_EM'] = all_not_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 
                reward_metrics[f'{ds}/mid_EM'] = mid_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0 # reward metrics 
            # First half EM metrics
            for ds, uid_to_acc_lst in ds_uid_to_acc_first_half.items():
                all_EM = 0
                all_not_EM = 0
                mid_EM = 0
                for uid, acc_lst in uid_to_acc_lst.items():
                    if sum(acc_lst) == half_point:
                        all_EM += 1
                    elif sum(acc_lst) == 0:
                        all_not_EM += 1
                    else:
                        mid_EM += 1
                reward_metrics[f'{ds}/first_half_all_EM'] = all_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0
                reward_metrics[f'{ds}/first_half_all_not_EM'] = all_not_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0
                reward_metrics[f'{ds}/first_half_mid_EM'] = mid_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0
            # Second half EM metrics
            for ds, uid_to_acc_lst in ds_uid_to_acc_second_half.items():
                all_EM = 0
                all_not_EM = 0
                mid_EM = 0
                for uid, acc_lst in uid_to_acc_lst.items():
                    if sum(acc_lst) == half_point:
                        all_EM += 1
                    elif sum(acc_lst) == 0:
                        all_not_EM += 1
                    else:
                        mid_EM += 1
                reward_metrics[f'{ds}/second_half_all_EM'] = all_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0
                reward_metrics[f'{ds}/second_half_all_not_EM'] = all_not_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0
                reward_metrics[f'{ds}/second_half_mid_EM'] = mid_EM / len(uid_to_acc_lst) if len(uid_to_acc_lst) > 0 else 0

        else:
            if key in ["invalid_analysis", "invaild_action_cnt", "is_entail"]:
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]
            for ds, val in zip(data_source_lst, val_lst):
                ds_to_value[ds].append(val)
            for ds, val_lst in ds_to_value.items():
                reward_metrics[f'{ds}/{key}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0 # reward metrics 
                if key == "f1" and evaluation:
                    reward_metrics[f'f1/{ds}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0

    
    if not evaluation:
        acc_lst = [int(getattr(sample, "exact_match")) for sample in samples]
        
        # Overall metrics
        all_EM = []
        all_not_EM = []
        mid_EM = []

        uid_to_EM = defaultdict(int)
        for uid, acc in zip(uid_lst, acc_lst):
            uid_to_EM[uid] += acc

        for item in uid_to_EM.items():
            if item[1] == n_samples_per_prompt:
                all_EM.append(1)
                all_not_EM.append(0)
                mid_EM.append(0)
            elif item[1] == 0:
                all_not_EM.append(1)
                all_EM.append(0)
                mid_EM.append(0)
            else:
                all_EM.append(0)
                all_not_EM.append(0)
                mid_EM.append(1)
        reward_metrics['tool_reward_info/all_EM'] = sum(all_EM) / len(all_EM) if len(all_EM) > 0 else 0
        reward_metrics['tool_reward_info/all_not_EM'] = sum(all_not_EM) / len(all_not_EM) if len(all_not_EM) > 0 else 0
        reward_metrics['tool_reward_info/mid_EM'] = sum(mid_EM) / len(mid_EM) if len(mid_EM) > 0 else 0
        
        # First half metrics
        first_half_EM = []
        first_half_not_EM = []
        first_half_mid_EM = []
        
        uid_to_EM_first_half = defaultdict(int)
        uid_to_EM_second_half = defaultdict(int)
        
        for i, (uid, acc) in enumerate(zip(uid_lst, acc_lst)):
            if first_half_mask[i]:
                uid_to_EM_first_half[uid] += acc
            else:
                uid_to_EM_second_half[uid] += acc
        
        for item in uid_to_EM_first_half.items():
            if item[1] == half_point:
                first_half_EM.append(1)
                first_half_not_EM.append(0)
                first_half_mid_EM.append(0)
            elif item[1] == 0:
                first_half_not_EM.append(1)
                first_half_EM.append(0)
                first_half_mid_EM.append(0)
            else:
                first_half_EM.append(0)
                first_half_not_EM.append(0)
                first_half_mid_EM.append(1)
        
        reward_metrics['tool_reward_info/first_half_EM'] = sum(first_half_EM) / len(first_half_EM) if len(first_half_EM) > 0 else 0
        reward_metrics['tool_reward_info/first_half_not_EM'] = sum(first_half_not_EM) / len(first_half_not_EM) if len(first_half_not_EM) > 0 else 0
        reward_metrics['tool_reward_info/first_half_mid_EM'] = sum(first_half_mid_EM) / len(first_half_mid_EM) if len(first_half_mid_EM) > 0 else 0
        
        # Second half metrics
        second_half_EM = []
        second_half_not_EM = []
        second_half_mid_EM = []
        
        for item in uid_to_EM_second_half.items():
            if item[1] == half_point:
                second_half_EM.append(1)
                second_half_not_EM.append(0)
                second_half_mid_EM.append(0)
            elif item[1] == 0:
                second_half_not_EM.append(1)
                second_half_EM.append(0)
                second_half_mid_EM.append(0)
            else:
                second_half_EM.append(0)
                second_half_not_EM.append(0)
                second_half_mid_EM.append(1)
        
        reward_metrics['tool_reward_info/second_half_EM'] = sum(second_half_EM) / len(second_half_EM) if len(second_half_EM) > 0 else 0
        reward_metrics['tool_reward_info/second_half_not_EM'] = sum(second_half_not_EM) / len(second_half_not_EM) if len(second_half_not_EM) > 0 else 0
        reward_metrics['tool_reward_info/second_half_mid_EM'] = sum(second_half_mid_EM) / len(second_half_mid_EM) if len(second_half_mid_EM) > 0 else 0
    
        keys = ["exact_match", "f1", "query_num", "is_format", "is_entail"]
        for key in keys:
            if key in ["exact_match", "is_format", "invalid_analysis", "is_entail"]:
                val_lst = [int(getattr(sample, key)) for sample in samples]
            else:
                val_lst = [getattr(sample, key) for sample in samples]
            reward_metrics[f'tool_reward_info/{key}'] = sum(val_lst) / len(val_lst) if len(val_lst) > 0 else 0
            
            # First half
            val_lst_first_half = [val_lst[i] for i in range(len(val_lst)) if first_half_mask[i]]
            reward_metrics[f'tool_reward_info/first_half_{key}'] = sum(val_lst_first_half) / len(val_lst_first_half) if len(val_lst_first_half) > 0 else 0
            
            # Second half
            val_lst_second_half = [val_lst[i] for i in range(len(val_lst)) if not first_half_mask[i]]
            reward_metrics[f'tool_reward_info/second_half_{key}'] = sum(val_lst_second_half) / len(val_lst_second_half) if len(val_lst_second_half) > 0 else 0

    return reward_metrics
