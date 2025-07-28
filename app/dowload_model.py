from huggingface_hub import snapshot_download

# Model MRC LLM
snapshot_download(
    repo_id="nguyenvulebinh/vi-mrc-large",
    local_dir="models/vi-mrc-large",
    local_dir_use_symlinks=False
)

# Model embedding
snapshot_download(
    repo_id="keepitreal/vietnamese-sbert",
    local_dir="models/vietnamese-sbert",
    local_dir_use_symlinks=False
)

# Model tìm ngữ cảnh
snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir="models/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir_use_symlinks=False
)
