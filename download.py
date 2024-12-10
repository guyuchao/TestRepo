from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(
   repo_id="TencentARC/t2iadapter_sketch_sd14v1",
   local_dir="experiments/pretrained_models/t2iadapter_sketch_sd14v1",
   token="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")
