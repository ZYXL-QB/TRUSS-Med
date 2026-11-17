from pathlib import Path


model_dir = Path(__file__).resolve().parents[3] / "models"


huggingface_embeddings_config = {
    "stella-base-en-v2": {
        "model_name": "infgrad/stella-base-en-v2",
        "model_path": str(model_dir / "stella-base-en-v2"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 1024,
    },
    "stella-base-zh-v2": {
        "model_name": "infgrad/stella-base-zh-v2",
        "model_path": str(model_dir / "stella-base-zh-v2"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 1024,
    }
}

