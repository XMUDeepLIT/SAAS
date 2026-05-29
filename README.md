# SAAS

SAAS is an implementation of **Self-Aware Reinforcement Learning for Over-Search Mitigation in Agentic Search**. SAAS trains a search-augmented LLM to recognize its own knowledge boundary, so it searches only when its parametric knowledge is insufficient and stops once enough evidence has been collected.

1. For each training question, draw two grouped rollouts under the current policy: a search-disabled group and a search-enabled group.
2. Compare the number of correct trajectories in the two groups to classify the question as `NO_SEARCH`, `NEED_SEARCH`, or `UNDETERMINED`.
3. Apply a boundary-aware reward: penalize every search on `NO_SEARCH` questions, penalize only searches beyond the minimum sufficient count on `NEED_SEARCH` questions, and skip the search penalty on `UNDETERMINED` questions.
4. Optimize with a two-stage curriculum — Stage I (capability acquisition) trains with the outcome-only reward, Stage II (efficiency refinement) activates the boundary-aware reward.

## Core structure

```text
SAAS/
├── train.py                                              # Synchronous RL training entry
├── train_async.py                                        # Asynchronous RL training entry
├── requirements.txt                                      # Python dependencies
├── setup.py                                              # Package installer
├── README.md                                             # Usage and project overview
├── search/                                               # SAAS task implementation
│   ├── generate_with_search_group_mqnq_differentprompt.py  # Multi-turn rollout + boundary-aware group reward
│   ├── generate_with_search_group_f1.py                  # Outcome-only F1 reward (Stage I / baseline)
│   ├── sglang_rollout.py                                 # SGLang rollout driver
│   ├── local_search_server.py                            # Async client for the local dense retriever
│   ├── qa_em_format.py                                   # EM / F1 / entailment scoring + tag-format check
│   ├── utils.py                                          # Boundary-aware penalty curves and metrics
│   ├── rm_hub.py                                         # Reward-model registry helpers
│   └── shell/                                            # Launch scripts (training + evaluation)
├── slime/                                                # Vendored slime RL framework (rollout, ray, backends)
├── slime_plugins/                                        # Optional plugins
├── tools/                                                # Checkpoint conversion utilities
└── tests/                                                # Unit and integration tests
```

## Environment Setup

Use the `slimerl/slime:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e . --no-deps
# for Search R1
pip install chardet
```

Download and prepare the training data:

```bash
cd /root/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
pip install -e . --no-deps
pip install tensordict

# Set your working directory
WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# Process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# (Optional) Process multiple dataset search format test file
# Note: the final file is not shuffled
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

## Retrieval backend

`SEARCH_R1_CONFIGS` at the top of `search/generate_with_search_group.py` selects the backend:

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 5,
    "topk": 3,
    "search_concurrency": 32,
    "search_backend": "local",                          # "local" or "google"
    "local":  {"search_url": "http://127.0.0.1:8000/retrieve", "proxy": None},
    "google": {"api_key": "<your-serper-dev-key>", "snippet_only": True, "proxy": None},
    "return_logprob": True,
    "format_score": 0.1,
}
```

For the `local` backend, run a FAISS retrieval server over the 2018 Wikipedia dump with E5-base-v2 as the encoder.

## Run

The local retriever requires a **separate conda environment** (it uses GPU for efficient retrieval, and we keep its dependencies isolated from the training environment). Steps 1–4 set up and start the retrieval server; Step 5 launches training in the base Python environment.

### Step 1: Install conda (skip if already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Step 2: Create the retriever environment

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

# PyTorch with CUDA
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

# Retrieval stack
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

### Step 3: Download the index and corpus

The retrieval files are large: roughly **60–70 GB** for download and **132 GB** after extraction. Make sure you have enough disk space.

```bash
save_path=/path/to/Index

# Download the split index parts and the corpus
python search/local_dense_retriever/download.py --save_path $save_path

# Combine split index files into a single FAISS index
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress the corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 4: Start the local retrieval server

```bash
# If "conda: command not found", first run:
# source ~/miniconda3/etc/profile.d/conda.sh && conda init && source ~/.bashrc

conda activate retriever

save_path=/path/to/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search/local_dense_retriever/retrieval_server.py \
    --index_path      $index_file \
    --corpus_path     $corpus_file \
    --topk            3 \
    --retriever_name  $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

Notes:

- First startup downloads the encoder and loads the index, which can take several minutes; subsequent starts take 1–2 minutes.
- GPU memory usage: approximately 5–7 GB per GPU.
- The retrieval server keeps running after the shell exits. To restart it, find the PID via `lsof -i :8000`, kill it, and rerun the command above.

### Step 5: Install SAAS, convert the checkpoint, and launch training

Make sure you are **NOT** in the retriever conda environment. If you are, run `conda deactivate`.

```bash
# Install SAAS in the base / training environment
pip install -e . --no-deps
pip install -r requirements.txt

# Convert the HF checkpoint to Megatron torch_dist
source slime/scripts/models/qwen2.5-3B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint   /path/to/Qwen2.5-3B-Instruct \
    --save            /path/to/Qwen2.5-3B-Instruct_torch_dist

# If a previous Ray run got stuck, clear its cache before relaunching:
# rm -rf ~/.cache && rm -rf ~/.ray*

# Launch training (edit placeholders at the top of the script first)
bash search/shell/train.sh
```

Placeholders to fill in inside the launch script:

```bash
data_dir=/path/to/data                       # train.jsonl + test.jsonl
model_dir=/path/to/hf/models                 # original HF checkpoint
train_model_dir=/path/to/megatron/models     # converted torch_dist checkpoint
model_name=<your-model-name>
experiment_name=<your-experiment-name>
WANDB_KEY=<your-wandb-key>
data_saved_path_dir=/path/to/save
```

The `CUSTOM_ARGS` block already wires the SAAS rollout and reward into slime.

## Main outputs

A run materializes the following directories under `data_saved_path`:

```text
ckpts/                       # Megatron checkpoints saved every save_freq rollouts
reward_info/                 # Per-rollout JSON dumps of boundary decisions, query counts, EM/F1, rewards
log/logs-<timestamp>.log     # stdout/stderr tee
tb_logs/<experiment>/        # TensorBoard event files
```

The per-rollout dump in `reward_info/` is enough to recompute the reported metrics (Accuracy, Search Count, QOR, SOR) offline.

## Notes

All environment-specific absolute paths, model names, and API keys have been replaced with command-line arguments or placeholders. The code does not assume a specific user name, machine name, or GPU model.
