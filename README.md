# TRUSS-Med: Medical Question Answering via Transformers, Retrieval, and Unified State Space Models

Jamba 1.6 医学选择题推理、LoRA 微调与模型配置消融。

[English README](README_en.md)

> 本文沿用原操作说明。`data/` 目录中的 MedQA 数据、数据处理代码及 `dataset_info.json` 复用自[参考项目的 data 目录](https://github.com/julienamaury/Medical-Answering-Model-202410/tree/2ce178140e0aab1cbe56691f28bbee00c3f09294/data)，MedMCQA 数据来自本项目原实验目录；复制时仅排除了 `.DS_Store`、`__pycache__` 和 `.ipynb_checkpoints` 等缓存文件。

## 实验流程

从原始数据复现实验时，依次配置环境、选择数据并完成预处理，再获取模型、配置消融实验、运行推理和微调，最后核对评测结果。使用仓库里已经预处理的数据时，按[第 2 步](#2-数据集与知识来源)设置训练文件和评测文件，再进入第 4 步。使用自有题库和知识文档时，按第 2、3 步准备。

克隆后进入包含本 README 的仓库根目录，在同一 Bash 终端中执行命令。代码、数据和模型路径都相对于这份克隆。首次验证可选用[小样本验证](#小样本验证)，按第 4、6、7、8 步检查模型下载、基础推理、LoRA 微调和结果输出。

## 1. 实验环境配置

数据预处理和 Jamba 使用两套 Conda 环境。以下命令在同一个终端中执行。已有环境时，用 `conda env list` 查看名称，并将对应变量设为实际名称。

```bash
TRUSS_DATA_ENV="${TRUSS_DATA_ENV:-truss_data_env}"
TRUSS_JAMBA_ENV="${TRUSS_JAMBA_ENV:-jamba_env}"
```

### 数据预处理环境

重建数据集时，在 Ubuntu 22.04 x86_64 上创建数据预处理环境。使用仓库内已有的预处理数据时，直接配置 Jamba 环境。

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

向量索引使用 CPU 版 FAISS，Stella 嵌入和检索脚本使用 CUDA。[requirements.txt](requirements.txt) 记录历史实验环境。

### 硬件与系统级驱动依赖

本项目参考环境为 **Ubuntu 22.04 LTS、8 张 NVIDIA A800**。下表记录本项目的单实例资源分配。

| 任务 | CPU 核心 | 系统内存 | GPU |
| --- | ---: | ---: | ---: |
| Jamba Mini 1.6 推理与微调 | 16 | 120.0 GB | 2 张 A800 |
| Jamba Large 1.6 推理 | 64 | 480.0 GB | 8 张 A800 |

AI21 的参考配置为 Mini 默认 BF16 推理使用 **2 张 80 GB GPU**，Mini BF16 LoRA 示例约需 **130 GB 总显存**，Large 在 **8 张 80 GB GPU** 上使用 `experts_int8`。运行前用 `nvidia-smi` 核对服务器实际可见的卡数、单卡显存和驱动版本。 [Mini 模型卡](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6) · [Mini 1.6 LoRA 示例](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6/blob/14ea3d665a1f68c74f3a7cadaa32ebc84f352308/README.md) · [Large 模型卡](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6)

| 系统依赖 | 参考版本 | 说明 |
| --- | --- | --- |
| CUDA | 12.4 Update 1 | 采用 [vLLM 0.6.5 官方 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.6.5/Dockerfile) 的 CUDA 版本。 |
| NVIDIA Linux 驱动 | `>=550.54.15` | 按 [CUDA 12.4 Update 1 对应的 Linux x86_64 驱动版本](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html) 选取。 |
| cuDNN、NCCL | 随 PyTorch CUDA 构建安装 | 使用与所选 PyTorch CUDA 构建一同分发的库。 |

### 软件包版本依赖

AI21 的 Jamba 1.6 模型卡要求 `vllm>=0.6.5`，并指出 `transformers` 4.44.0 和 4.44.1 存在 Jamba 兼容问题。下表按官方要求和本仓库脚本使用的接口给出参考组合，需在目标服务器上验证。

| 软件包 | 参考版本 | 用途 |
| --- | --- | --- |
| Python | 3.10 | 推理与微调 |
| PyTorch | `2.5.1`（CUDA 12.4 构建） | 推理与微调。参见 [vLLM 0.6.5 CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.6.5/requirements-cuda.txt)和 [PyTorch 发布版本](https://docs.pytorch.org/get-started/previous-versions/#v251)。 |
| vLLM | `0.6.5` | Mini、Large 和合并 LoRA 后的推理。 |
| Transformers | `4.46.3` | 加载模型和分词器，满足 [vLLM 0.6.5](https://github.com/vllm-project/vllm/blob/v0.6.5/requirements-common.txt) 与 [TRL 0.12.2](https://github.com/huggingface/trl/blob/v0.12.2/setup.py) 的版本范围。 |
| TRL | `0.12.2` | Mini 微调，支持脚本使用的 [`SFTConfig` 和 `max_seq_length`](https://huggingface.co/docs/trl/v0.12.2/en/sft_trainer)。 |
| Datasets、Accelerate、PEFT | `datasets>=2.21.0`、`accelerate>=0.34.0`、`peft>=0.8.0` | 按 [TRL 0.12.2 的依赖要求](https://github.com/huggingface/trl/blob/v0.12.2/setup.py) 选择兼容版本，用于读取训练数据、设备分配和 LoRA。 |
| Mamba CUDA 扩展 | `mamba-ssm`、`causal-conv1d>=1.2.0` | Transformers 加载的模型在启用 `use_mamba_kernels` 时需要这两个包，选用与 PyTorch 和 CUDA 匹配的构建。参见 [AI21 的 Mamba 安装说明](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6/blob/14ea3d665a1f68c74f3a7cadaa32ebc84f352308/README.md#prerequisites)。 |
| pandas、openpyxl、tqdm | 随推理环境安装 | 写入 `.xlsx` 和显示进度。 |

如果服务器已经配好 Jamba 环境，直接激活并检查版本。

```bash
conda activate "$TRUSS_JAMBA_ENV"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -m pip show torch vllm transformers trl peft
python -m pip check
```

在 Ubuntu 22.04 LTS 服务器上新建 Jamba 环境时，按下面的命令安装。安装后用上面的命令核对 GPU 和软件包版本。

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

Mini 推理脚本固定用 2 卡，Large 推理脚本固定用 8 卡。运行前用 `CUDA_VISIBLE_DEVICES` 选择分配到的 GPU。

当模型配置启用 `use_mamba_kernels` 时，在 LoRA 微调前安装对应的 CUDA 扩展。下面的 wheel 对应上述 Python 3.10、Linux x86_64、PyTorch 2.5.1 CUDA 12 环境。 [Mamba 发布文件](https://github.com/state-spaces/mamba/releases/tag/v2.2.4) · [causal-conv1d 发布文件](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.0.post8)

```bash
python -m pip install \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
    'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -c 'import causal_conv1d, mamba_ssm'
```

## 2. 数据集与知识来源

### 已经预处理数据集

| 数据集 | LoRA 微调文件 | 样本数 |
| --- | --- | ---: |
| MedQA | [RAG_MedQA_Mainland_train_500(example).json](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json) | 500 |
| MedMCQA | [Med_MCQA_knowledge_test_train.json](data/MedMCQA/result/Med_MCQA_knowledge_test_train.json) | 300 |

两份微调文件的每条样本都在 `messages` 中包含 `system`、`user` 和 `assistant`。`assistant` 保存解析和参考答案，为 LoRA 微调提供监督数据。重新预处理时，按第 3 步用豆包生成解析并写入 `assistant`。仓库现有文件的解析来源待核实。

首次验证可用 MedQA 微调数据和独立于该训练文件的 MedMCQA 样本检查脚本流程。评测文件包含 `answer_idx`，推理脚本过滤 `assistant` 消息后运行推理。 [MedMCQA 评测样本](data/MedMCQA/result/Med_MCQA_test.json)

```bash
TRAIN_NAME='RAG_MedQA_Mainland_train_500(example)'
TRAIN_DATA="MedQA/$TRAIN_NAME.json"
EVAL_DATA='MedMCQA/result/Med_MCQA_test.json'
```

Jamba 脚本的数据参数以仓库的 `data/` 为根，`TRAIN_DATA` 和 `EVAL_DATA` 省略了这个前缀。也可以传入数据文件的绝对路径。选择 MedMCQA 微调时，将 `TRAIN_NAME` 设为 `Med_MCQA_knowledge_test_train`，将 `TRAIN_DATA` 设为 `MedMCQA/result/Med_MCQA_knowledge_test_train.json`，并将 `EVAL_DATA` 指向独立留出的评测文件。正式实验应在同一个数据集上划分训练集和评测集，并固定各组的评测输入。

MedMCQA 微调文件与仓库里的两份 300 题评测样本使用了同一批题目，其中带知识的评测样本连对话内容也与微调文件逐条相同。用这份 MedMCQA 数据微调后，需要另选留出题来评估模型效果。 [带知识评测样本](data/MedMCQA/result/Med_MCQA_knowledge_test.json) · [基础评测样本](data/MedMCQA/result/Med_MCQA_test.json)

### 小样本验证

复制上面的 MedQA 训练与 MedMCQA 评测变量设置后，取前 4 条训练样本和前 3 条评测样本。运行第 4 步下载模型，再执行第 6、7 步的 Mini 对照组命令和第 8 步的结果核验。小样本只减少生成和训练耗时，模型加载仍需第 1 步的 GPU 配置。

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

### 原始数据与知识来源

#### MedQA 数据集

* 数据集来源：[https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)
* 下载链接：[https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing)
* 中文介绍：[https://zhuanlan.zhihu.com/p/679590312](https://zhuanlan.zhihu.com/p/679590312)


#### MedMCQA 数据集

* 数据集来源：[MedMCQA](https://github.com/medmcqa/medmcqa)
* 下载链接：[官方数据下载](https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky)
* 原始训练文件 `data/MedMCQA/data/MedMCQA_train.json` 和 `data/MedMCQA/data/train.json` 使用 Git LFS。需要这些文件时，在克隆目录运行 `git lfs install` 和 `git lfs pull`。上面的预处理样本可以直接读取。

从原始数据构建实验样本时，先划分训练集与评测集，再准备本次实验使用的知识文档。

## 3. 数据集预处理

从原始数据构建训练集和评测集时，先将知识文档转为 JSON，建立向量索引并为题目附加检索知识，再生成训练解析和格式化数据。已有教材 JSON 时可从[向量索引构建](#向量索引构建)开始。已有检索结果时可从[训练解析生成](#训练解析生成)开始。

```bash
conda activate "$TRUSS_DATA_ENV"
```

### 知识文档格式转换

将 UTF-8 的教材或企业知识文档交给现有 [txt2json.py](data/MedQA/utils/txt2json.py) 切分。下面使用仓库里的教材作为示例。处理企业知识时，将 `--input_txt_dir` 和 `--output_json_dir` 换成对应目录。

```bash
(
    cd data/MedQA/utils || exit
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_paragraph' --output_json_dir '../data_clean/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_sentence' --output_json_dir '../data_clean/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
    python -u txt2json.py --input_txt_dir '../data_clean/textbooks/en' --output_json_dir '../data_clean/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
)
```

### 向量索引构建

现有向量化脚本会遍历中英文两套 Stella 模型。[config.py](data/MedQA/utils/config.py) 根据仓库位置定位 `models/` 下的模型。下载两套模型后再建立索引。 [stella-base-zh-v2](https://huggingface.co/infgrad/stella-base-zh-v2) · [stella-base-en-v2](https://huggingface.co/infgrad/stella-base-en-v2)

```bash
huggingface-cli download infgrad/stella-base-zh-v2 --local-dir models/stella-base-zh-v2
huggingface-cli download infgrad/stella-base-en-v2 --local-dir models/stella-base-en-v2
```

使用 `vector_store.py` 为上一步生成的知识 JSON 建立索引。自定义知识目录时，调整 `--input_json_dir` 和 `--store_top_path`。

```bash
(
    cd data/MedQA/utils || exit
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_sentence_json' --store_top_path '../data_clean/vector_stores/zh_sentence'
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_paragraph_json' --store_top_path '../data_clean/vector_stores/zh_paragraph'
    python -u vector_store.py --input_json_dir '../data_clean/textbooks/en_json' --store_top_path '../data_clean/vector_stores/en'
)
```

### 知识检索与增强

使用 `generate_question_with_knowledges.py` 为训练集、验证集和评测集附加检索知识。根据实验使用的题库与检索输入选择对应的命令。原实验采用 `question_with_options`，即用题干和选项检索。

只有一张可用 GPU 时，将 `--device_ids` 设为 `0`，并将 `--num_process` 设为 `1`。使用自定义向量索引时，将 `--store_path` 指向上一环节指定的索引目录。

#### 中文题干检索

```bash
(
    cd data/MedQA/utils || exit
    # 训练集
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
        --query_key_name="question"

    # 验证集
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
        --query_key_name="question"

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
        --query_key_name="question"
)
```

#### 中文题干与选项检索

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

#### 英文题干检索

```bash
(
    cd data/MedQA/utils || exit
    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/train.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/train" \
        --output_json_file="train.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question"

    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/dev.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/dev" \
        --output_json_file="dev.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question"

    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/test.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/test" \
        --output_json_file="test.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question"
)
```

#### 英文题干与选项检索

```bash
(
    cd data/MedQA/utils || exit
    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/train.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/train" \
        --output_json_file="train.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"



    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/dev.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/dev" \
        --output_json_file="dev.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"


    python generate_question_with_knowledges.py \
        --num_process=8 \
        --index_key_name="id" \
        --topk_knowledge=10 \
        --knowledge_threshold=0.65 \
        --input_jsonl_path="../data_clean/questions/US/test.jsonl" \
        --embedding_model_name="stella-base-en-v2" \
        --store_path="../data_clean/vector_stores/en" \
        --output_json_dir="../data_clean/questions_with_knowledge/US/test" \
        --output_json_file="test.json" \
        --device_ids="0,1,2,3,4,5,6,7" \
        --query_key_name="question_with_options"
)
```

以使用题干和选项检索的 Mainland 训练集为例，输出为 `data/MedQA/data_clean/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train.json`。转换前先核对样本数和 `knowledges_for_llm` 中的检索内容。

### 训练解析生成

为数据集生成解析时，可以通过火山方舟的 OpenAI 兼容接口调用豆包文本模型。在 [火山方舟控制台](https://console.volcengine.com/ark) 选择目标项目，开通豆包模型服务，并在「API Key 管理」中创建密钥。复制已开通的模型 ID；如果使用自定义推理接入点，复制其 `ep-` 开头的接入点 ID。参见 [火山方舟快速入门](https://docs.volcengine.com/docs/ark/quick-start?lang=zh)。

在运行 Python 的同一个 Bash 终端中设置环境变量。按提示输入密钥，并将 `ARK_MODEL_ID` 的占位值换成刚复制的 ID。

```bash
read -rsp '火山方舟 API Key: ' ARK_API_KEY
printf '\n'
export ARK_API_KEY
export ARK_MODEL_ID='填入豆包模型 ID 或 ep- 开头的接入点 ID'
```

如果当前环境还没有 OpenAI Python SDK，先运行 `python -m pip install 'openai>=1.0'`。用以下命令发送一条测试消息，终端应打印豆包的回复。参见 [火山方舟 OpenAI SDK 接入文档](https://docs.volcengine.com/docs/ark/compatible-with-openai-sdk?lang=zh)。

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
    messages=[{"role": "user", "content": "请用一句话回复：连接成功。"}],
)
print(response.choices[0].message.content)
PY
```

### 训练集与评测集格式化

上面的豆包示例用于验证一次 API 调用。批量生成豆包解析需要从外部调用流程取得结果，再按题目 ID 写入检索后的训练 JSON 的 `chatgpt_explain` 字段。仓库中的现有批处理脚本提供 Kimi 等接口。 [批处理脚本](data/MedQA/utils/MultiProcessingLLM/openai_api_llm.py) · [原解析提示词](data/MedQA/utils/MultiProcessingLLM/prompt_config_of_generate_explain.json)

保留检索输出里的 `id`、`meta_info`、`question_with_options`、`answer_with_idx`、`answer_idx`、`knowledges_for_llm` 和 `retrieve_info`。以使用题干和选项检索的 Mainland 题库为例，先运行 `mkdir -p data/local`，将带豆包解析的训练 JSON 数组保存为 `data/local/Mainland_train_with_doubao.json`，再生成微调与评测文件。评测输入使用第 3 步中检索后的 `test.json`。

```bash
(
    cd data/MedQA/utils || exit
    python format_json_dataset_for_training_llm.py \
        --input_json_data_path ../../local/Mainland_train_with_doubao.json \
        --dataset_name RAG_MedQA_Mainland_train_doubao \
        --output_json_data_dir ../../local
    python format_json_dataset_for_training_llm.py \
        --input_json_data_path ../data_clean/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test.json \
        --dataset_name RAG_MedQA_Mainland_eval \
        --output_json_data_dir ../../local \
        --do_test
)
```

两份输出位于 `data/local/`。训练文件包含解析和参考答案组成的 `assistant` 消息。`--do_test` 生成含 `system`、`user` 和计分用 `answer_idx` 的评测样本。将下列变量用于后续推理和微调。

```bash
TRAIN_NAME=RAG_MedQA_Mainland_train_doubao
TRAIN_DATA="local/$TRAIN_NAME.json"
EVAL_DATA=local/RAG_MedQA_Mainland_eval.json
```

## 4. 模型获取

Jamba Mini 1.6 和 Large 1.6 权重分别位于魔搭仓库的 `mini/` 和 `large/` 目录。 [打开魔搭模型文件](https://modelscope.cn/models/zyxl2026/Jamba/files) · `https://modelscope.cn/models/zyxl2026/Jamba/files`

在 Jamba 环境安装魔搭 CLI，并设置本地模型路径。为模型权重、消融副本和合并后的 LoRA 权重预留磁盘空间。

```bash
conda activate "$TRUSS_JAMBA_ENV"
python -m pip install modelscope-hub==0.7.1

JAMBA_REPO=zyxl2026/Jamba
JAMBA_REVISION=master
JAMBA_DIR=model/Jamba-1.6
MINI_MODEL="$JAMBA_DIR/mini"
LARGE_MODEL="$JAMBA_DIR/large"
```

运行 Mini 推理或微调时，下载 `mini/`。Large 的下载命令见第 6 步。后续脚本接收下载后的本地模型目录。

```bash
ms-hub download "$JAMBA_REPO" --revision "$JAMBA_REVISION" \
    --local-dir "$JAMBA_DIR" --include 'mini/*'
test -f "$MINI_MODEL/config.json" && test -f "$MINI_MODEL/model.safetensors.index.json"
```

## 5. 消融实验配置

1. 原始模型目录作为对照组。另存一份模型目录用于消融，并使用能区分两组的目录名。
2. 在消融目录的 `config.json` 中填写本次实验实际要测试的 Jamba 内部参数。记录字段名、原值和修改值。需要改变权重张量形状的配置应使用与之匹配的权重。
3. 两组使用相同的评测文件、提示、采样参数和训练数据。推理脚本里的 `temperature`、`top_k`、`top_p` 和上下文长度沿用现有设置。后续分别把各组模型目录传给现有 Jamba 脚本。

```bash
ABLATION_MODEL=model/AI21-Jamba-Mini-1.6-ablation
# 准备消融副本时执行一次，再编辑副本里的 config.json。
test ! -e "$ABLATION_MODEL" && cp -a "$MINI_MODEL" "$ABLATION_MODEL"
```

## 6. 基础模型推理

先用原始配置运行 Mini 推理。 [jamba16mini_inference.py](jamba/jamba16mini_inference.py)

```bash
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p jamba/results

(cd jamba && python jamba16mini_inference.py "../$MINI_MODEL" "$EVAL_DATA")
```

完成第 5 步后，用同一份评测数据运行消融组。

```bash
(cd jamba && python jamba16mini_inference.py "../$ABLATION_MODEL" "$EVAL_DATA")
```

如果运行 Large 推理，使用第 4 步设置的变量，单独下载 `large/`，再选择 8 张可见 GPU 运行 `experts_int8`。 [jamba16large_inference.py](jamba/jamba16large_inference.py)

```bash
ms-hub download "$JAMBA_REPO" --revision "$JAMBA_REVISION" \
    --local-dir "$JAMBA_DIR" --include 'large/*'
test -f "$LARGE_MODEL/config.json" && test -f "$LARGE_MODEL/model.safetensors.index.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
(cd jamba && python jamba16large_inference.py "../$LARGE_MODEL" "$EVAL_DATA")
```

## 7. LoRA 微调与推理

### LoRA 微调

使用同一训练文件分别对各组进行微调。先运行对照组。 [jamba16mini_finetune.py](jamba/jamba16mini_finetune.py)

```bash
export CUDA_VISIBLE_DEVICES=0,1
(cd jamba && python jamba16mini_finetune.py "../$MINI_MODEL" "$TRAIN_DATA")
```

对消融组执行同样的训练。

```bash
(cd jamba && python jamba16mini_finetune.py "../$ABLATION_MODEL" "$TRAIN_DATA")
```

输出目录为 `jamba-finetune-data/<模型目录名>/<训练文件名去掉 .json>/`。确认日志没有 `Training failed`，并在实际输出目录中找到 `adapter_config.json` 和 `adapter_model.safetensors`。如果选择中间检查点，使用实际生成的 `checkpoint-*` 目录。

### 微调后推理

用 [jamba16mini_inference_by_finetune.py](jamba/jamba16mini_inference_by_finetune.py) 加载对应 LoRA。第三个参数相对于 `jamba-finetune-data/`，可以填写最终权重目录，也可以追加实际的 `checkpoint-*`。该脚本会在所选目录下写入或复用 `merged/`，应使用和微调时同组的基础模型及 LoRA 权重。

```bash
BASE_PEFT="$(basename "$MINI_MODEL")/$TRAIN_NAME"
(cd jamba && python jamba16mini_inference_by_finetune.py "../$MINI_MODEL" "$EVAL_DATA" "$BASE_PEFT")
```

使用消融组生成的 LoRA 权重运行推理。

```bash
ABLATION_PEFT="$(basename "$ABLATION_MODEL")/$TRAIN_NAME"
(cd jamba && python jamba16mini_inference_by_finetune.py "../$ABLATION_MODEL" "$EVAL_DATA" "$ABLATION_PEFT")
```

## 8. 评测结果与核验

结果写到 `jamba/results/`，文件名包含模型名、数据集名、采样参数和时间戳。分别记录原始配置和消融配置在基础权重与 LoRA 权重下的样本数、正确数和准确率。

- 终端应出现「推理完成」，并在结果目录生成新的 `.xlsx`。
- Excel 包含 `Question`、`Correct Answer`、`Predicted Option`、`Generated Text` 和 `Correct`。使用仓库里的完整 MedMCQA 评测样本时，每份文件应有 300 条数据。按第 2 步的小样本验证运行时，应有 3 条。检查生成文本和未提取到选项的行，再对比准确率。
- 如需重新提取模型给出的选项，可对结果文件的副本运行 [extract_answer.py](utils/extract_answer.py)。脚本会覆盖所处理目录里的 Excel。

```bash
mkdir -p jamba/results/recheck
# 先把需要重新处理的 .xlsx 复制到 jamba/results/recheck/
python utils/extract_answer.py --directory jamba/results/recheck
```

每次评测完成后，随结果记录仓库提交、模型权重版本、评测数据划分、`config.json` 的具体改动，以及推理和训练参数。保留原始结果文件，以便核对各组的准确率。

## 9. 快速测试

取已预处理的 MedQA Mainland 训练示例前 300 条，验证 Jamba Large 1.6 的 `experts_int8` 推理流程。检索知识已写入 `user` 消息，推理脚本会过滤 `assistant` 消息。 [MedQA 预处理示例](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json)

完成[第 1 步的 Jamba 环境配置](#1-实验环境配置)后，在已激活的 Jamba 环境中，从仓库根目录按顺序执行下面两段命令。使用 8 张 A800 80GB GPU，默认选择 `0,1,2,3,4,5,6,7`，也可以预先设置 `CUDA_VISIBLE_DEVICES`。测试命令已指向 `model/Jamba-1.6/large/`。

### 1. 下载模型

将企业魔搭仓库的 `large/` 权重下载到 `model/Jamba-1.6/large/`，首次下载约 797.12 GB。下载中断后，重新执行同一段命令继续。 [魔搭模型文件](https://modelscope.cn/models/zyxl2026/Jamba/files) · `https://modelscope.cn/models/zyxl2026/Jamba/files`

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

### 2. 运行快速测试

下载完成后，在同一个仓库根目录执行下面整段命令。

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

通过时打印 `Quick test passed` 和本次生成的 `.xlsx` 路径。通过条件为本次生成的工作簿包含 300 条结果，每条都有模型生成文本。
