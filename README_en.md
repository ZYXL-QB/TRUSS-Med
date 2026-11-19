# TRUSS-Med: Medical Question Answering via Transformers, Retrieval, and Unified State Space Models

TRUSS-Med provides Jamba 1.6 inference, LoRA fine-tuning, and model ablation for medical multiple-choice question answering. [中文 README](README.md)

Run the commands from the cloned repository root in the same Bash terminal. Section 2 provides prepared datasets for a first run. Follow Sections 4, 6, 7, and 8 to check the Mini baseline workflow. Sections 3 and 5 cover dataset reconstruction and ablation.

## 1. Environment Setup

The reference machine runs Ubuntu 22.04 LTS with eight NVIDIA A800 GPUs. The table records this project's resource allocation per instance.

| Task | CPU cores | System RAM | GPUs |
| --- | ---: | ---: | ---: |
| Jamba Mini 1.6 inference and LoRA | 16 | 120 GB | 2 A800s |
| Jamba Large 1.6 inference | 64 | 480 GB | 8 A800s |

AI21's reference uses two GPUs with 80 GB of VRAM each for Mini BF16 inference, about 130 GB of aggregate VRAM for its BF16 LoRA example, and eight 80 GB GPUs with `experts_int8` for Large inference. Check the visible GPU count and memory on the target host. [Mini model card](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6) · [Mini LoRA example](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6/blob/14ea3d665a1f68c74f3a7cadaa32ebc84f352308/README.md#lora-example) · [Large model card](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6)

Set the environment names once. Use `conda env list` to check the names if you are using existing environments.

```bash
TRUSS_DATA_ENV="${TRUSS_DATA_ENV:-truss_data_env}"
TRUSS_JAMBA_ENV="${TRUSS_JAMBA_ENV:-jamba_env}"
```

Install Jamba prerequisites on a Linux x86_64 CUDA host. The example uses Python 3.10, PyTorch 2.5.1 with CUDA 12.4, and vLLM 0.6.5. The reference driver for CUDA 12.4 Update 1 is 550.54.15 or newer. [CUDA driver compatibility](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html) · [vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.6.5/requirements-cuda.txt)

```bash
conda create -n "$TRUSS_JAMBA_ENV" python=3.10 -y
conda activate "$TRUSS_JAMBA_ENV"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm==0.6.5 transformers==4.46.3 trl==0.12.2 \
    datasets==3.1.0 accelerate==1.1.1 peft==0.13.2 huggingface-hub==0.26.5 \
    pandas==2.2.2 openpyxl==3.1.2 tqdm==4.67.1 openai==1.58.1 httpx==0.27.2
python -m pip check
python -c 'import torch, vllm; from trl import SFTConfig, SFTTrainer; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

Use the following checks with a new or existing Jamba environment.

```bash
conda activate "$TRUSS_JAMBA_ENV"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -m pip show torch vllm transformers trl peft
python -m pip check
```

If the model configuration enables `use_mamba_kernels`, install the matching wheels before running LoRA. These wheels target the Python 3.10 and PyTorch 2.5 CUDA 12 configuration above.

```bash
python -m pip install \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
    'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -c 'import causal_conv1d, mamba_ssm'
```

To rebuild data in Section 3, create the separate preprocessing environment below. FAISS runs on CPU and the Stella embedding models use CUDA. The original experiment environment is recorded in [requirements.txt](requirements.txt).

```bash
conda create -n "$TRUSS_DATA_ENV" python=3.10 -y
conda activate "$TRUSS_DATA_ENV"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.46.3 huggingface-hub==0.26.5 \
    langchain==0.2.0 langchain-community==0.2.0 sentence-transformers==2.7.0 \
    faiss-cpu==1.8.0.post1 numpy==1.26.4 pydantic==2.7.1 tqdm==4.66.0 \
    openai==1.58.1 httpx==0.27.2
python -m pip check
```

## 2. Datasets

| Dataset | Prepared LoRA input | Examples |
| --- | --- | ---: |
| MedQA | [RAG_MedQA_Mainland_train_500(example).json](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json) | 500 |
| MedMCQA | [Med_MCQA_knowledge_test_train.json](data/MedMCQA/result/Med_MCQA_knowledge_test_train.json) | 300 |

Every training example has `system`, `user`, and `assistant` messages, including an explanation in the assistant message. The JSON files contain no attribution for the model that generated the existing explanations.

For a workflow check, use the MedQA training example and the [MedMCQA evaluation sample](data/MedMCQA/result/Med_MCQA_test.json). The scripts resolve dataset arguments relative to `data/`. Inference filters out assistant messages and scores predictions using `answer_idx`.

```bash
TRAIN_NAME='RAG_MedQA_Mainland_train_500(example)'
TRAIN_DATA="MedQA/$TRAIN_NAME.json"
EVAL_DATA='MedMCQA/result/Med_MCQA_test.json'
```

For a MedMCQA training run, set `TRAIN_NAME=Med_MCQA_knowledge_test_train` and `TRAIN_DATA=MedMCQA/result/Med_MCQA_knowledge_test_train.json`. Use a separate held-out evaluation file. Both bundled 300-question MedMCQA evaluation variants overlap with this training file.

### Small-sample verification

With the variables above set to the MedQA training and MedMCQA evaluation examples, create a four-example training set and a three-example evaluation set. Then follow the Mini baseline commands in Sections 4, 6, 7, and 8. Loading the model still requires the standard GPU configuration.

```bash
python - "$TRAIN_DATA" "$EVAL_DATA" <<'PY'
import json
import sys
from pathlib import Path

data = Path("data")
output = data / "local"
output.mkdir(exist_ok=True)
for source, target, count in zip(sys.argv[1:], ("smoke_train.json", "smoke_eval.json"), (4, 3)):
    samples = json.loads((data / source).read_text(encoding="utf-8"))
    (output / target).write_text(json.dumps(samples[:count], ensure_ascii=False), encoding="utf-8")
PY
TRAIN_NAME=smoke_train
TRAIN_DATA=local/smoke_train.json
EVAL_DATA=local/smoke_eval.json
```

The raw datasets are described by their upstream projects. The two large raw MedMCQA training files, `data/MedMCQA/data/MedMCQA_train.json` and `data/MedMCQA/data/train.json`, use Git LFS. If you need them, run `git lfs install` and `git lfs pull` in the clone. The prepared examples above are standard Git files. [MedQA](https://github.com/jind11/MedQA) · [MedMCQA](https://github.com/medmcqa/medmcqa)

## 3. Dataset Preprocessing

Set up the training and held-out splits before retrieval. MedQA includes UTF-8 textbook files and their converted JSON versions. Use `txt2json.py` for new text, `vector_store.py` to build indexes, and `generate_question_with_knowledges.py` to retrieve passages for each split. [MedQA utilities](data/MedQA/utils)

Activate the data environment before running this section.

```bash
conda activate "$TRUSS_DATA_ENV"
```

### Text conversion and vector indexes

The following commands convert the bundled textbook files to JSON. If the converted files are already available, start with the Stella downloads.

```bash
(
    cd data/MedQA/utils || exit
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_paragraph' --output_json_dir '../data_clean/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_sentence' --output_json_dir '../data_clean/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/en' --output_json_dir '../data_clean/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
)
```

The embedding configuration resolves both model directories under `models/` in the clone. The indexing script builds indexes for both Stella models. [Embedding configuration](data/MedQA/utils/config.py)

```bash
huggingface-cli download infgrad/stella-base-zh-v2 --local-dir models/stella-base-zh-v2
huggingface-cli download infgrad/stella-base-en-v2 --local-dir models/stella-base-en-v2
```

```bash
(
    cd data/MedQA/utils || exit
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_sentence_json' --store_top_path '../data_clean/vector_stores/zh_sentence'
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_paragraph_json' --store_top_path '../data_clean/vector_stores/zh_paragraph'
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/en_json' --store_top_path '../data_clean/vector_stores/en'
)
```

### Knowledge retrieval

The examples below use the question and answer options for retrieval in each Mainland split. With one visible GPU, set `--device_ids=0` and `--num_process=1`. See the [Chinese README retrieval examples](README.md#知识检索与增强) for the US dataset and question-only queries.

```bash
(
    cd data/MedQA/utils || exit
    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/Mainland/train.jsonl" \
        --embedding_model_name="stella-base-zh-v2" \
        --store_path="../data_clean/vector_stores/zh_paragraph" \
        --output_json_dir="../data_clean/questions_with_knowledge/Mainland/train" \
        --output_json_file="train.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"

    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/Mainland/dev.jsonl" \
        --embedding_model_name="stella-base-zh-v2" \
        --store_path="../data_clean/vector_stores/zh_paragraph" \
        --output_json_dir="../data_clean/questions_with_knowledge/Mainland/dev" \
        --output_json_file="dev.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"

    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/Mainland/test.jsonl" \
        --embedding_model_name="stella-base-zh-v2" \
        --store_path="../data_clean/vector_stores/zh_paragraph" \
        --output_json_dir="../data_clean/questions_with_knowledge/Mainland/test" \
        --output_json_file="test.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"
)
```

### Prepare retrieved knowledge

For both training and evaluation, populate `knowledges_for_llm` before formatting the records into `user` messages. The command below follows the existing `load_raw_json_data_with_process` function by joining the passages in `retrieve_info.knowledges_info.knowledges` with two newline characters. Run it from the repository root to create the two local JSON files. [Original field preparation logic](data/MedQA/utils/MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py#L191)

```bash
python - <<'PY'
import json
from pathlib import Path

retrieval_dir = Path("data/MedQA/data_clean/questions_with_knowledge/Mainland")
output_dir = Path("data/local")
output_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    source = retrieval_dir / split / "retrive_use_question_with_options/stella-base-zh-v2" / f"{split}.json"
    examples = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(examples, dict):
        examples = examples["data"]

    for example in examples:
        knowledge_list = example["retrieve_info"]["knowledges_info"]["knowledges"]
        example["knowledges_for_llm"] = "\n\n".join(knowledge_list)

    destination = output_dir / f"Mainland_{split}_with_knowledge.json"
    destination.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(examples)} examples: {destination}")
PY
```

Use `data/local/Mainland_train_with_knowledge.json` to generate training explanations. Use `data/local/Mainland_test_with_knowledge.json` to format the evaluation split.

### Training explanations and dataset formatting

Enable a Doubao text model in the Volcengine Ark console and obtain its model or endpoint ID and API key. Set the two environment variables in the same terminal and run a one-message connectivity check. [Ark quick start](https://docs.volcengine.com/docs/ark/quick-start?lang=en) · [OpenAI-compatible API](https://docs.volcengine.com/docs/ark/compatible-with-openai-sdk?lang=en)

```bash
read -rsp 'Ark API key: ' ARK_API_KEY
printf '\n'
export ARK_API_KEY
export ARK_MODEL_ID='replace with a Doubao model ID or ep- endpoint ID'
```

```bash
python - <<'PY'
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ["ARK_API_KEY"],
)
response = client.chat.completions.create(
    model=os.environ["ARK_MODEL_ID"],
    messages=[{"role": "user", "content": "Reply with a short confirmation."}],
)
print(response.choices[0].message.content)
PY
```

Use the prepared `data/local/Mainland_train_with_knowledge.json` as input to your Ark batch workflow. Join the training explanations to these examples by `id` and put each explanation in `chatgpt_explain`. Preserve `knowledges_for_llm`, the metadata, retrieval passages, question and answer fields. The repository's existing batch client contains integrations such as Kimi. [Batch client](data/MedQA/utils/MultiProcessingLLM/openai_api_llm.py)

Save the combined JSON array as `data/local/Mainland_train_with_doubao.json`. Format it for training and format `data/local/Mainland_test_with_knowledge.json` with `--do_test`. That flag leaves the assistant response out of the evaluation conversation and retains the reference answer.

```bash
(
    cd data/MedQA/utils || exit
    python format_json_dataset_for_training_llm.py \
        --input_json_data_path ../../local/Mainland_train_with_doubao.json \
        --dataset_name RAG_MedQA_Mainland_train_doubao \
        --output_json_data_dir ../../local
    python format_json_dataset_for_training_llm.py \
        --input_json_data_path ../../local/Mainland_test_with_knowledge.json \
        --dataset_name RAG_MedQA_Mainland_eval \
        --output_json_data_dir ../../local \
        --do_test
)
```

```bash
TRAIN_NAME=RAG_MedQA_Mainland_train_doubao
TRAIN_DATA="local/$TRAIN_NAME.json"
EVAL_DATA=local/RAG_MedQA_Mainland_eval.json
```

## 4. Model Download

The ModelScope repository stores Jamba Mini 1.6 under `mini/` and Large 1.6 under `large/`. [Browse the model files on ModelScope](https://modelscope.cn/models/zyxl2026/Jamba/files) · `https://modelscope.cn/models/zyxl2026/Jamba/files`

Run the commands in this section from the repository root to download the weights directly into this repository's `model/` directory. Mini and Large use the subdirectories below. Subsequent inference, fine-tuning, and quick-test commands use these local paths.

| Model | Local download directory |
| --- | --- |
| Jamba Mini 1.6 | `model/Jamba-1.6/mini/` |
| Jamba Large 1.6 | `model/Jamba-1.6/large/` |

Install the ModelScope CLI in the Jamba environment and set the local model paths. Allow disk space for the model weights, an ablation copy, and merged LoRA weights.

```bash
conda activate "$TRUSS_JAMBA_ENV"
python -m pip install modelscope-hub==0.7.1

JAMBA_REPO=zyxl2026/Jamba
JAMBA_REVISION=master
JAMBA_DIR=model/Jamba-1.6
MINI_MODEL="$JAMBA_DIR/mini"
LARGE_MODEL="$JAMBA_DIR/large"
mkdir -p "$JAMBA_DIR"
```

Download `mini/` for Mini inference or fine-tuning. Section 6 provides the Large download command. The scripts use the downloaded local model directories.

```bash
ms-hub download "$JAMBA_REPO" --revision "$JAMBA_REVISION" \
    --local-dir "$JAMBA_DIR" --include 'mini/*'
test -f "$MINI_MODEL/config.json" && test -f "$MINI_MODEL/model.safetensors.index.json"
```

## 5. Model Ablation

Copy the downloaded Mini directory and apply the experiment's `config.json` changes to the copy. Record the original and changed values and use weights compatible with the configuration in each group. Keep training inputs, held-out questions, and inference settings fixed across groups.

```bash
ABLATION_MODEL=model/AI21-Jamba-Mini-1.6-ablation
# Run once to create the copy, then edit its config.json.
test ! -e "$ABLATION_MODEL" && cp -a "$MINI_MODEL" "$ABLATION_MODEL"
```

## 6. Baseline Inference

The bundled Jamba Mini 1.6 `checkpoint-300` is located at `jamba-finetune-data/finetune_result/AI21-Jamba-Mini-1.6/RAG_MedQA_USS_test_train/checkpoint-300/`, relative to the repository root. To run fine-tuned inference with this checkpoint, use the [checkpoint inference command in Section 7](#inference-with-mini-checkpoint-300). [Browse checkpoint-300](jamba-finetune-data/finetune_result/AI21-Jamba-Mini-1.6/RAG_MedQA_USS_test_train/checkpoint-300)

The Mini baseline uses two visible GPUs.

```bash
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p jamba/results

(cd jamba && python jamba16mini_inference.py "../$MINI_MODEL" "$EVAL_DATA")
```

After preparing the alternate model configuration in Section 5, run it against the same evaluation file.

```bash
(cd jamba && python jamba16mini_inference.py "../$ABLATION_MODEL" "$EVAL_DATA")
```

For Large inference, use the variables from Section 4 to download `large/` separately, then run the script with eight visible GPUs and `experts_int8`.

```bash
ms-hub download "$JAMBA_REPO" --revision "$JAMBA_REVISION" \
    --local-dir "$JAMBA_DIR" --include 'large/*'
test -f "$LARGE_MODEL/config.json" && test -f "$LARGE_MODEL/model.safetensors.index.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
(cd jamba && python jamba16large_inference.py "../$LARGE_MODEL" "$EVAL_DATA")
```

## 7. LoRA Fine-tuning and Inference

Run the same training file for the baseline and ablation groups. The adapter is written to `jamba-finetune-data/<model directory name>/<training file stem>/`. Check that the training log completes and that `adapter_config.json` and `adapter_model.safetensors` are present.

```bash
export CUDA_VISIBLE_DEVICES=0,1
(cd jamba && python jamba16mini_finetune.py "../$MINI_MODEL" "$TRAIN_DATA")
```

Train the ablation group with the same input file.

```bash
(cd jamba && python jamba16mini_finetune.py "../$ABLATION_MODEL" "$TRAIN_DATA")
```

Use the relative adapter directory that the training step created. A periodic checkpoint can be selected by appending its actual name. This script creates and reuses a `merged/` directory under the selected adapter.

```bash
BASE_PEFT="$(basename "$MINI_MODEL")/$TRAIN_NAME"
(cd jamba && python jamba16mini_inference_by_finetune.py "../$MINI_MODEL" "$EVAL_DATA" "$BASE_PEFT")
```

Use the ablation group's adapter for its inference run.

```bash
ABLATION_PEFT="$(basename "$ABLATION_MODEL")/$TRAIN_NAME"
(cd jamba && python jamba16mini_inference_by_finetune.py "../$ABLATION_MODEL" "$EVAL_DATA" "$ABLATION_PEFT")
```

### Inference with Mini checkpoint-300

The repository includes a Jamba Mini 1.6 LoRA checkpoint at `jamba-finetune-data/finetune_result/AI21-Jamba-Mini-1.6/RAG_MedQA_USS_test_train/checkpoint-300/`. Load it with the Mini base weights downloaded in Section 4 to run fine-tuned inference directly. [checkpoint-300 files](jamba-finetune-data/finetune_result/AI21-Jamba-Mini-1.6/RAG_MedQA_USS_test_train/checkpoint-300)

Complete the Jamba environment setup in Section 1, download Mini in Section 4, and install Git LFS. Then run the entire block below from the repository root. The checkpoint records PEFT `0.15.2`, so the command installs that version for loading. This example uses two GPUs and the 300 bundled MedMCQA evaluation examples. The third argument, `MINI_CHECKPOINT`, is relative to `jamba-finetune-data/` and must include the `finetune_result/` prefix. [Git LFS installation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

```bash
(
set -e
python -m pip install peft==0.15.2
git lfs install --local

MINI_MODEL="$PWD/model/Jamba-1.6/mini"
MINI_CHECKPOINT="finetune_result/AI21-Jamba-Mini-1.6/RAG_MedQA_USS_test_train/checkpoint-300"
EVAL_DATA="MedMCQA/result/Med_MCQA_test.json"

git lfs pull --include="jamba-finetune-data/$MINI_CHECKPOINT/*" --exclude=""
test -s "$MINI_MODEL/config.json"
test -s "$MINI_MODEL/model.safetensors.index.json"
test -s "jamba-finetune-data/$MINI_CHECKPOINT/adapter_config.json"
test -s "jamba-finetune-data/$MINI_CHECKPOINT/adapter_model.safetensors"

export CUDA_VISIBLE_DEVICES=0,1
mkdir -p jamba/results
(cd jamba && python jamba16mini_inference_by_finetune.py "$MINI_MODEL" "$EVAL_DATA" "$MINI_CHECKPOINT")
)
```

The script creates or reuses `merged/` inside the selected checkpoint, merges the Mini base weights with the LoRA adapter, and runs inference. Workbooks are written to `jamba/results/` with `checkpoint-300` in the filename.

## 8. Evaluation Outputs

Inference writes workbooks under `jamba/results/`. Check for a new workbook and the completion log. A workbook should contain three prediction rows for the small-sample check in Section 2, or 300 for the full bundled MedMCQA evaluation example. Check the generated text and missing predicted options before comparing accuracy.

To rerun option extraction, copy the relevant workbooks to `jamba/results/recheck/`. The utility overwrites the workbooks it processes. [Answer extraction](utils/extract_answer.py)

```bash
mkdir -p jamba/results/recheck
# Copy the workbooks to jamba/results/recheck/ before running the utility.
python utils/extract_answer.py --directory jamba/results/recheck
```

For each evaluation, record the repository commit, model revision, evaluation split, exact changes to `config.json`, and inference and training settings. Keep the original output files so you can verify the accuracy for each group.

## 9. Quick Test

Use the first 300 prepared MedQA Mainland training examples to check Jamba Large 1.6 inference with `experts_int8`. The `user` messages contain retrieved knowledge, and the inference script filters out `assistant` messages. [Prepared MedQA examples](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json)

Complete the [Jamba environment setup](#1-environment-setup), then run the following two blocks in order from the repository root with that environment active. Use eight A800 80GB GPUs. The command defaults to `0,1,2,3,4,5,6,7` and respects an existing `CUDA_VISIBLE_DEVICES` setting. The test command uses the fixed local path `model/Jamba-1.6/large/`.

### 1. Download the model

Download the enterprise ModelScope repository's `large/` weights into `model/Jamba-1.6/large/`. The initial download is approximately 797.12 GB. Rerun the same block to resume an interrupted download. [ModelScope model files](https://modelscope.cn/models/zyxl2026/Jamba/files) · `https://modelscope.cn/models/zyxl2026/Jamba/files`

```bash
(
set -e
python -m pip install modelscope-hub==0.7.1
mkdir -p "model/Jamba-1.6"
ms-hub download zyxl2026/Jamba --revision master \
    --local-dir "$PWD/model/Jamba-1.6" --include 'large/*'
test -s "$PWD/model/Jamba-1.6/large/config.json"
test -s "$PWD/model/Jamba-1.6/large/model.safetensors.index.json"
)
```

### 2. Run the quick test

After the download completes, run the entire block below from the same repository root.

```bash
(
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LARGE_MODEL="$PWD/model/Jamba-1.6/large"
if [ ! -s "$LARGE_MODEL/config.json" ] || [ ! -s "$LARGE_MODEL/model.safetensors.index.json" ]; then
    printf '%s\n' 'Download Jamba Large in Step 1 above before running the quick test.' >&2
    exit 1
fi
mkdir -p data/local jamba/results
RUN_DIR="$(mktemp -d data/local/quick_eval_XXXXXX)"
RUN_NAME="${RUN_DIR##*/}"
EVAL_DATA="local/$RUN_NAME/$RUN_NAME.json"
SAMPLE_COUNT=300

python - "$EVAL_DATA" "$SAMPLE_COUNT" <<'PY'
import json
import sys
from pathlib import Path

import torch

if torch.cuda.device_count() < 8:
    raise SystemExit("Jamba Large needs eight visible GPUs for this command.")

source = Path("data/MedQA/RAG_MedQA_Mainland_train_500(example).json")
sample_count = int(sys.argv[2])
samples = json.loads(source.read_text(encoding="utf-8"))[:sample_count]
if len(samples) != sample_count:
    raise SystemExit(f"The prepared MedQA file needs at least {sample_count} examples.")
Path("data", sys.argv[1]).write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
PY

(cd jamba && python jamba16large_inference.py "$LARGE_MODEL" "$EVAL_DATA")

python - "$RUN_NAME" "$SAMPLE_COUNT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

workbooks = list(Path("jamba/results").glob(f"jamba16large_inference_*_{sys.argv[1]}_*.xlsx"))
if len(workbooks) != 1:
    raise SystemExit("Expected one new workbook for this run. Check the inference log.")
rows = pd.read_excel(workbooks[0], keep_default_na=False)
required = {"Question", "Correct Answer", "Predicted Option", "Generated Text", "Correct"}
sample_count = int(sys.argv[2])
if len(rows) != sample_count or not required.issubset(rows.columns):
    raise SystemExit(f"Expected {sample_count} rows and the evaluation columns in the workbook.")
if rows["Generated Text"].fillna("").astype(str).str.strip().eq("").any():
    raise SystemExit("At least one question has no generated text. Check the inference log.")
print(f"Quick test passed: {workbooks[0]}")
PY
)
```

A successful run prints `Quick test passed` and the path to its new `.xlsx` workbook. The workbook must contain 300 result rows with generated text for each question.
