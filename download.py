from huggingface_hub import HfApi
api = HfApi()


api.upload_folder(
   folder_path="experiments/pretrained_models/paper_pretrained_models/Mix_of_Show_Fused_Models/potter+hermione+thanos+dogA+dogB+catA_chilloutmix/combined_model_base",
   repo_id="guyuchao/MixofShow_Fused_ChilloutMix_SixConcepts",
   repo_type="model",
   token="")