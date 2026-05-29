# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import re

from search.qa_em_format import compute_em, compute_f1, is_valid_sequence, compute_score_f1, compute_entail, extract_solution, is_retrieval_correct

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from collections import defaultdict
from search.utils import calculate_score_smooth_label_V1_middle_linear, calc_repetition_rate


# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 5,
    "topk": 3,
    "search_concurrency": 32,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.1,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    title_to_doc = defaultdict(list)
    title_to_doc_id = defaultdict(list)
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        doc_id = doc_item["document"]["id"]
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        title_to_doc[title].append(text)
        title_to_doc_id[title].append(doc_id)
    return format_reference, title_to_doc, title_to_doc_id


async def search(query: str) -> str:
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from search.local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from search.google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
    )


def postprocess_predictions(prediction: str):
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content

async def execute_predictions(prediction: str, no_retrieval: bool) -> str:

    action, content = postprocess_predictions(prediction)

    search_query = ""
    title_to_doc_id = {}
    title_to_doc = {}
    if no_retrieval:
        if action == "answer":
            next_obs = ""
            done = True
        else:
            next_obs = ""
            done = False
    else:
        if action == "search":
            search_query = content
            async with SEMAPHORE:
                search_results, title_to_doc, title_to_doc_id = await search(search_query)
            next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
            done = False
        elif action == "answer":
            next_obs = ""
            done = True
        else:
            next_obs = "\nMy previous action is invalid. \
    If I want to search, I should put the query between <search> and </search>. \
    If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
            done = False

    return next_obs, done, search_query, title_to_doc, title_to_doc_id


async def generate(args, sample: Sample, sampling_params, group_idx: int = None) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Handle partial rollout samples: continue generation from existing response
    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None

    no_retrieval = False

    if group_idx is not None:
        sample.in_group_idx = group_idx
        no_retrieval = args.disable_search_tokens_for_first_half and sample.in_group_idx < args.disable_search_first_n and sample.metadata['data_source'] == 'train_searchR1_nq'

    if no_retrieval:
        prompt_template = """Answer the given question. You must conduct reasoning inside <think> and </think> first. After reasoning, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}"""
        prompt = {"role": "user", "content": prompt_template.format(question=sample.metadata['question'])}
        sample.prompt = state.tokenizer.apply_chat_template(
                    [prompt],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        # Use the new prompt for SGLang server as well
        prompt_text = sample.prompt

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done, search_query, title_to_doc, title_to_doc_id = await execute_predictions(cur_response, no_retrieval)
        if done:
            sample.is_query_rep = (len(sample.query_list) != len(set(sample.query_list)))
            
            doc_count = 0
            lap_doc_count = 0
            lap_ratio = 0
            for title, id_list in sample.doc_id_list.items():
                doc_count += len(id_list)
                lap_doc_count += len(id_list) - len(set(id_list))

            if doc_count > 0:
                lap_ratio = lap_doc_count / doc_count
            sample.lap_ratio = lap_ratio

            break
        if not no_retrieval:
            assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        if search_query != "":
            sample.query_num += 1
            sample.query_list.append(search_query)
            for title, doc_list in title_to_doc.items():
                sample.doc_list[title].extend(doc_list)

            for title, id_list in title_to_doc_id.items():
                sample.doc_id_list[title].extend(id_list)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    # Store statistics for wandb logging
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    sample.exact_match = compute_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label,
    )
    sample.f1 = compute_f1(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label,
    )
    # sample.is_format, format_reason = is_valid_sequence(sample.prompt + sample.response)

    sample.is_format, format_reason = is_valid_sequence(sample.prompt + sample.response)
    score = compute_score_f1(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label,
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    line_obj = {
        "id": sample.index,
        "prompt": sample.prompt,
        "response": sample.response,
        "exact_match": sample.exact_match,
        "f1": sample.f1,
        "is_format": sample.is_format,
        "format_reason": format_reason,
        "score": score,
    }

    return score, line_obj

async def mq_group_reward_func(args, samples, **kwargs):
    """
    The group reward function for multi-hop retrieval-based question answering.

    Args:
        args: the arguments
        samples: group samples to evaluate
    """
    correct_query_num_list = []
    exist_em_head = False
    head_correct_count = 0
    exist_em_tail = False
    score_func = calculate_score_smooth_label_V1_linear

    for sample in samples:
        exact_match = compute_em(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        score = compute_score_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label, format_score=SEARCH_R1_CONFIGS["format_score"])
        f1 = compute_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        sample.exact_match = exact_match
        sample.score = score
        sample.f1 = f1

        sample.is_entail = compute_entail(solution_str=sample.prompt + sample.response, ground_truth=sample.label)

        # Only check group_index if it's not None (training mode)
        group_index = getattr(sample, 'in_group_idx', None)
        if group_index is not None:
            disable_search_first_n = args.disable_search_first_n
            if sample.exact_match and group_index < disable_search_first_n:
                exist_em_head = True
                head_correct_count += 1
            elif sample.exact_match and group_index >= disable_search_first_n:
                exist_em_tail = True
                correct_query_num_list.append(sample.query_num)

    correct_num = len(correct_query_num_list)
    rollout_num = args.n_samples_per_prompt
    percentage = float(correct_num) / float(rollout_num)
    correct_query_mean = float(sum(correct_query_num_list)) / float(correct_num) if correct_num > 0 else 0

    label = ""

    for sample in samples:
        reward = sample.score
        group_index = getattr(sample, 'in_group_idx', None)
        # if group_index is not None and group_index >= args.disable_search_first_n:
        if True:
            if args.efficient_query:
                if exist_em_head and head_correct_count >= 2:
                    query_num_thre = 0
                    label = "easy"
                elif exist_em_tail:
                    query_num_thre = 0x3f3f3f3f
                    for query_num in correct_query_num_list:
                        query_num_thre = min(query_num_thre, query_num)
                    if query_num_thre == 0x3f3f3f3f and exist_em_head:
                        query_num_thre = 0
                    label = "middle"
                else:
                    label = "hard"
                    query_num_thre = args.max_query_nums
                reward  = score_func(current_query_num=sample.query_num, query_num_thre=query_num_thre, score_i=reward, label=label)
                sample.retrieval_penalty = reward - sample.score
                # sample.retrieval_penalty = score_func(current_query_num=sample.query_num, query_num_thre=query_num_thre, score_i=reward, label=label) - sample.score

        if args.query_rep_penalty and sample.is_query_rep:
            reward = reward - 0.1

        sample.is_format, _ = is_valid_sequence(sample.prompt + sample.response)
        sample.is_retrieval = is_retrieval_correct(sample.prompt + sample.response, sample.label["target"])
        repetition_ratio, _ = calc_repetition_rate(sample.response)
        is_rsp_repeat = repetition_ratio >= 0.1
        sample.rsp_rep_ratio = repetition_ratio
        sample.is_rsp_rep = is_rsp_repeat

        if args.repetition_detect and sample.is_rsp_rep:
            reward = reward - 0.5
 
        sample.reward = reward

    group_result = []
    for sample in samples:
        result = {
            "group_index": getattr(sample, 'in_group_idx', None),
            "data_source": sample.metadata['data_source'],
            "question": sample.metadata['question'],
            "pred": extract_solution(sample.prompt + sample.response),
            "ground_truth": sample.label,
            "query_num": sample.query_num if sample.query_num is not None else None,
            "query_list": sample.query_list,
            "doc_list": sample.doc_list,
            "doc_id_list": sample.doc_id_list,
            "lap_ratio": sample.lap_ratio,
            "is_query_rep": sample.is_query_rep,
            "retrieval_check": sample.is_retrieval,
            "score": sample.score,
            "reward": sample.reward,
            "retrieval_penalty": sample.retrieval_penalty,
            "exact_match": sample.exact_match,
            "is_entail": sample.is_entail,
            "f1": sample.f1,
            "rsp_token_len": sample.response_length,
            "prompt": sample.prompt,
            "response": sample.response,
        }

        group_result.append(result)

    line_obj = {
        "data_source": samples[0].metadata['data_source'],
        "exist_em_head": exist_em_head,
        "exist_em_tail": exist_em_tail,
        "rollout_info": group_result,
        "group_preds": [result["pred"] for result in group_result],
        "ground_truth": group_result[0]["ground_truth"],
        "rollout_num": rollout_num,
        "correct_query_threshold": correct_query_mean,
        "correct_query_num_list": correct_query_num_list,
        "correct_query_num": len(correct_query_num_list),
        "correct_query_percentage": len(correct_query_num_list) / rollout_num,
        "max_query_nums": args.max_query_nums,
        "group_index": [x["group_index"] for x in group_result],
        "group_scores": [x["score"] for x in group_result],
        "group_retrieval_check": [int(x['retrieval_check']) for x in group_result],
        "group_rewards": [x["reward"] for x in group_result],
        "group_retrieval_penalties": [x["retrieval_penalty"] for x in group_result],
        "group_EMs": [int(x["exact_match"]) for x in group_result],
        "group_entails": [int(x["is_entail"]) for x in group_result],
        "group_f1": [x["f1"] for x in group_result],
        "group_query_num": [x["query_num"] for x in group_result],
        "group_query_rep": [x["is_query_rep"] for x in group_result],
        "group_doc_list": [x["doc_list"] for x in group_result],
        "group_doc_id_list": [x["doc_id_list"] for x in group_result],
        "group_lap_ratio": [x["lap_ratio"] for x in group_result]
    }

    final_rewards = []
    for sample in samples:
        final_rewards.append(sample.reward)

    return final_rewards, line_obj


async def mq_group_reward_func2(args, samples, **kwargs):
    """
    The group reward function for multi-hop retrieval-based question answering.

    Args:
        args: the arguments
        samples: group samples to evaluate
    """
    correct_query_num_list = []
    exist_em = False
    score_func = calculate_score_smooth_label_V1_middle_linear

    for sample in samples:
        exact_match = compute_em(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        score = compute_score_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label, format_score=SEARCH_R1_CONFIGS["format_score"])
        f1 = compute_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        sample.exact_match = exact_match
        sample.score = score
        sample.f1 = f1

        sample.is_entail = compute_entail(solution_str=sample.prompt + sample.response, ground_truth=sample.label)

        # Only check group_index if it's not None (training mode)
        group_index = getattr(sample, 'in_group_idx', None)
        if group_index is not None:
            if sample.exact_match:
                exist_em = True
                correct_query_num_list.append(sample.query_num)

    correct_num = len(correct_query_num_list)
    rollout_num = args.n_samples_per_prompt
    percentage = float(correct_num) / float(rollout_num)
    correct_query_mean = float(sum(correct_query_num_list)) / float(correct_num) if correct_num > 0 else 0

    label = ""

    for sample in samples:
        reward = sample.score
        group_index = getattr(sample, 'in_group_idx', None)
        # if group_index is not None and group_index >= args.disable_search_first_n:
        if True:
            if args.efficient_query:
                if exist_em:
                    query_num_thre = 0x3f3f3f3f
                    for query_num in correct_query_num_list:
                        query_num_thre = min(query_num_thre, query_num)
                    label = "middle"
                else:
                    label = "hard"
                    query_num_thre = args.max_query_nums
                reward  = score_func(current_query_num=sample.query_num, query_num_thre=query_num_thre, score_i=reward, label=label)
                sample.retrieval_penalty = reward - sample.score
                # sample.retrieval_penalty = score_func(current_query_num=sample.query_num, query_num_thre=query_num_thre, score_i=reward, label=label) - sample.score


        if args.query_rep_penalty and sample.is_query_rep:
            reward = reward - 0.1

        sample.is_format, _ = is_valid_sequence(sample.prompt + sample.response)
        sample.is_retrieval = is_retrieval_correct(sample.prompt + sample.response, sample.label["target"])
        repetition_ratio, _ = calc_repetition_rate(sample.response)
        is_rsp_repeat = repetition_ratio >= 0.1
        sample.rsp_rep_ratio = repetition_ratio
        sample.is_rsp_rep = is_rsp_repeat

        if args.repetition_detect and sample.is_rsp_rep:
            reward = reward - 0.5
 
        sample.reward = reward

    group_result = []
    for sample in samples:
        result = {
            "group_index": getattr(sample, 'in_group_idx', None),
            "data_source": sample.metadata['data_source'],
            "question": sample.metadata['question'],
            "pred": extract_solution(sample.prompt + sample.response),
            "ground_truth": sample.label,
            "query_num": sample.query_num if sample.query_num is not None else None,
            "query_list": sample.query_list,
            "doc_list": sample.doc_list,
            "doc_id_list": sample.doc_id_list,
            "lap_ratio": sample.lap_ratio,
            "is_query_rep": sample.is_query_rep,
            "retrieval_check": sample.is_retrieval,
            "score": sample.score,
            "reward": sample.reward,
            "retrieval_penalty": sample.retrieval_penalty,
            "exact_match": sample.exact_match,
            "is_entail": sample.is_entail,
            "f1": sample.f1,
            "rsp_token_len": sample.response_length,
            "prompt": sample.prompt,
            "response": sample.response,
        }

        group_result.append(result)

    line_obj = {
        "data_source": samples[0].metadata['data_source'],
        "exist_em": exist_em,
        "rollout_info": group_result,
        "group_preds": [result["pred"] for result in group_result],
        "ground_truth": group_result[0]["ground_truth"],
        "rollout_num": rollout_num,
        "correct_query_threshold": correct_query_mean,
        "correct_query_num_list": correct_query_num_list,
        "correct_query_num": len(correct_query_num_list),
        "correct_query_percentage": len(correct_query_num_list) / rollout_num,
        "max_query_nums": args.max_query_nums,
        "group_index": [x["group_index"] for x in group_result],
        "group_scores": [x["score"] for x in group_result],
        "group_retrieval_check": [int(x['retrieval_check']) for x in group_result],
        "group_rewards": [x["reward"] for x in group_result],
        "group_retrieval_penalties": [x["retrieval_penalty"] for x in group_result],
        "group_EMs": [int(x["exact_match"]) for x in group_result],
        "group_entails": [int(x["is_entail"]) for x in group_result],
        "group_f1": [x["f1"] for x in group_result],
        "group_query_num": [x["query_num"] for x in group_result],
        "group_query_rep": [x["is_query_rep"] for x in group_result],
        "group_doc_list": [x["doc_list"] for x in group_result],
        "group_doc_id_list": [x["doc_id_list"] for x in group_result],
        "group_lap_ratio": [x["lap_ratio"] for x in group_result]
    }

    final_rewards = []
    for sample in samples:
        final_rewards.append(sample.reward)

    return final_rewards, line_obj


async def nq_group_reward_func(args, samples, **kwargs):
    """
    The group reward function for single-hop retrieval-based question answering.

    Args:
        args: the arguments
        samples: group samples to evaluate
    """
    correct_query_num_list = []

    for sample in samples:
        exact_match = compute_em(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        score = compute_score_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label, format_score=SEARCH_R1_CONFIGS["format_score"])
        f1 = compute_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
        sample.exact_match = exact_match
        sample.score = score
        sample.f1 = f1
        sample.is_entail = compute_entail(solution_str=sample.prompt + sample.response, ground_truth=sample.label)

        reward = score

        if args.query_rep_penalty and sample.is_query_rep:
            reward = reward - 0.1

        sample.is_format, _ = is_valid_sequence(sample.prompt + sample.response)
        sample.is_retrieval = is_retrieval_correct(sample.prompt + sample.response, sample.label["target"])
        repetition_ratio, _ = calc_repetition_rate(sample.response)
        is_rsp_repeat = repetition_ratio >= 0.1
        sample.rsp_rep_ratio = repetition_ratio
        sample.is_rsp_rep = is_rsp_repeat

        if args.repetition_detect and sample.is_rsp_rep:
            reward = reward - 0.5
        
        sample.reward = reward
        if exact_match:
            correct_query_num_list.append(sample.query_num)

    group_result = []
    for sample in samples:
        result = {
            "group_index": getattr(sample, 'in_group_idx', None),
            "data_source": sample.metadata['data_source'],
            "question": sample.metadata['question'],
            "pred": extract_solution(sample.prompt + sample.response),
            "ground_truth": sample.label,
            "query_num": sample.query_num if sample.query_num is not None else None,
            "query_list": sample.query_list,
            "doc_list": sample.doc_list,
            "doc_id_list": sample.doc_id_list,
            "lap_ratio": sample.lap_ratio,
            "is_query_rep": sample.is_query_rep,
            "retrieval_check": sample.is_retrieval,
            "score": sample.score,
            "reward": sample.reward,
            "exact_match": sample.exact_match,
            "is_entail": sample.is_entail,
            "f1": sample.f1,
            "rsp_token_len": sample.response_length,
            "prompt": sample.prompt,
            "response": sample.response,
        }

        group_result.append(result)

    line_obj = {
        "data_source": samples[0].metadata['data_source'],
        "rollout_info": group_result,
        "group_preds": [result["pred"] for result in group_result],
        "ground_truth": group_result[0]["ground_truth"],
        "rollout_num": args.n_samples_per_prompt,
        "correct_query_num_list": correct_query_num_list,
        "correct_query_num": len(correct_query_num_list),
        "correct_query_percentage": len(correct_query_num_list) / args.n_samples_per_prompt,
        "max_query_nums": args.max_query_nums,
        "group_index": [x["group_index"] for x in group_result],
        "group_scores": [x["score"] for x in group_result],
        "group_retrieval_check": [int(x['retrieval_check']) for x in group_result],
        "group_rewards": [x["reward"] for x in group_result],
        "group_EMs": [x["exact_match"] for x in group_result],
        "group_entails": [int(x["is_entail"]) for x in group_result],
        "group_f1": [x["f1"] for x in group_result],
        "group_query_num": [x["query_num"] for x in group_result],
        "group_query_rep": [x["is_query_rep"] for x in group_result],
        "group_doc_list": [x["doc_list"] for x in group_result],
        "group_doc_id_list": [x["doc_id_list"] for x in group_result],
        "group_lap_ratio": [x["lap_ratio"] for x in group_result]
    }

    final_rewards = []
    for sample in samples:
        final_rewards.append(sample.reward)

    return final_rewards, line_obj

async def eval_reward_func(args, sample: Sample, **kwargs):
    assert isinstance(sample.prompt, str), f"prompt: {sample.prompt}"
    exact_match = compute_em(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
    score = compute_score_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label, format_score=SEARCH_R1_CONFIGS["format_score"])
    f1 = compute_f1(solution_str=sample.prompt + sample.response, ground_truth=sample.label)
    sample.exact_match = exact_match
    sample.score = score
    sample.f1 = f1
    sample.is_entail = compute_entail(solution_str=sample.prompt + sample.response, ground_truth=sample.label)

    reward = score

    if args.query_rep_penalty and sample.is_query_rep:
        reward = reward - 0.1

    sample.is_format, _ = is_valid_sequence(sample.prompt + sample.response)
    sample.is_retrieval = is_retrieval_correct(sample.prompt + sample.response, sample.label["target"])
    repetition_ratio, _ = calc_repetition_rate(sample.response)
    is_rsp_repeat = repetition_ratio >= 0.1
    sample.rsp_rep_ratio = repetition_ratio
    sample.is_rsp_rep = is_rsp_repeat

    if args.repetition_detect and sample.is_rsp_rep:
        reward = reward - 0.5

    sample.reward = reward
    
    result = {
        "id": sample.index,
        "data_source": sample.metadata['data_source'],
        "question": sample.metadata['question'],
        "pred": extract_solution(sample.prompt + sample.response),
        "ground_truth": sample.label,
        "query_num": sample.query_num if sample.query_num is not None else None,
        "query_list": sample.query_list,
        "doc_list": sample.doc_list,
        "doc_id_list": sample.doc_id_list,
        "lap_ratio": sample.lap_ratio,
        "is_query_rep": sample.is_query_rep,
        "format_check": sample.is_format,
        "retrieval_check": sample.is_retrieval,
        "score": sample.score,
        "reward": sample.reward,
        "exact_match": sample.exact_match,
        "is_entail": sample.is_entail,
        "f1": sample.f1,
        "rsp_token_len": sample.response_length,
        "prompt": sample.prompt,
        "response": sample.response,
    }

    return reward, result