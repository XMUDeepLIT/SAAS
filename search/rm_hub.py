import asyncio
import random

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

async def batched_async_rm_search(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    custom_mq_rm_path = args.custom_mq_rm_path
    custom_nq_rm_path = args.custom_nq_rm_path
    
    reward_fn_dict = {
            "searchR1_hotpotqa": custom_mq_rm_path,
            "searchR1_nq": custom_nq_rm_path,
            "searchR1_bamboogle": custom_mq_rm_path,
            "searchR1_popqa": custom_nq_rm_path,
            "searchR1_musique": custom_mq_rm_path,
            "searchR1_triviaqa": custom_nq_rm_path,
            "searchR1_2wikimultihopqa": custom_mq_rm_path,
            "train_searchR1_hotpotqa": custom_mq_rm_path,
            "train_searchR1_nq": custom_nq_rm_path,
        }

    rm_function = load_function(reward_fn_dict[samples[0].metadata["data_source"]])
    return await rm_function(args, samples, **kwargs)

async def async_eval_rm_search(
    args,
    sample: Sample,
    **kwargs,
):
    rm_function = load_function(args.custom_eval_rm_path)
    return await rm_function(args, sample, **kwargs)