
from PIL import Image, ImageDraw, ImageFont
from roictrl.pipelines.pipeline_stable_diffusion_edlora import StableDiffusionEDLoRAPipeline, bind_concept_prompt
from roictrl.models.unet_2d_condition_model import UNet2DConditionModel
import torch
import math
from copy import deepcopy
from safetensors.torch import load_file
import json
from einops import rearrange


@torch.no_grad()
def encode_roi_input(input_data, pipe, device='cuda', max_rois=30, do_classifier_free_guidance=True, use_instance_cfg=True, negative_prompt=None):
    roi_boxes = input_data["roi_boxes"]
    roi_phrases = input_data["roi_phrases"]

    if len(roi_boxes) > max_rois:
        print(f"use first {max_rois} rois")
        roi_boxes = roi_boxes[: max_rois]
        roi_phrases = roi_phrases[: max_rois]
    assert len(roi_boxes) == len(roi_phrases), 'the roi phrase and position not equal'

    # encode roi prompt
    _roi_phrases = bind_concept_prompt(roi_phrases, pipe.new_concept_cfg)
    
    tokenizer_inputs = pipe.tokenizer(_roi_phrases, padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to(device)
    _instance_embedding = pipe.text_encoder(tokenizer_inputs.input_ids.to(device))[0]
    _instance_embedding = rearrange(_instance_embedding, "(b n) l d -> b n l d", n=16)

    if negative_prompt is None:
        uncond_text = ""
    else:
        uncond_text = negative_prompt
    uncond_tokenizer_inputs = pipe.tokenizer(uncond_text, padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to(device)
    uncond_text_res = pipe.text_encoder(uncond_tokenizer_inputs.input_ids.to(device))
    _instance_uncond_embedding = uncond_text_res[0]

    instance_boxes = torch.zeros(max_rois, 4, device=device, dtype=pipe.text_encoder.dtype)
    instance_embeddings = torch.zeros(
        max_rois, 16, pipe.tokenizer.model_max_length, pipe.unet.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )
    instance_masks = torch.zeros(max_rois, device=device, dtype=pipe.text_encoder.dtype)
    uncond_instance_embeddings = torch.zeros(
        max_rois, 16, pipe.tokenizer.model_max_length, pipe.unet.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )

    n_rois = len(roi_boxes)
    instance_boxes[:n_rois] = torch.tensor(roi_boxes)
    instance_embeddings[:n_rois] = _instance_embedding
    uncond_instance_embeddings[:n_rois] = _instance_uncond_embedding
    instance_masks[:n_rois] = 1
            
    if do_classifier_free_guidance:
        instance_boxes = torch.stack([instance_boxes] * 2)
        instance_embeddings = torch.stack([uncond_instance_embeddings, instance_embeddings])
        instance_masks = torch.stack([instance_masks] * 2)
        
        instance_boxemb_masks = instance_masks.clone()
        instance_boxemb_masks[0] = 0

        if not use_instance_cfg:
            instance_masks[0] = 0

    roictrl = {
        'instance_boxes': instance_boxes,
        'instance_embeddings': instance_embeddings,
        'instance_masks': instance_masks,
        'instance_boxemb_masks': instance_boxemb_masks
    }
    return roictrl


def setup_pipeline(pretrained_model_path, roictrl_path, device='cuda'):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, attention_type='roictrl', subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=False, device_map=None)

    pretrained_state_dict = load_file(roictrl_path)
    #pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'roifuser' in k or 'position_net' in k}
    unet.load_state_dict(pretrained_state_dict, strict=False)

    pipe = StableDiffusionEDLoRAPipeline.from_pretrained(
        pretrained_model_path,
        unet=unet, 
        safety_checker=None, 
        torch_dtype=torch.float16, 
        #variant="fp16"
    )
    pipe.to(device)

    with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)
    
    return pipe

def prepare_object_inference(use_baseline=False):
    input_data = {
        "caption": "Two dogs and a cat on the grass under the sunset, animal photography, 4K, high quality, high resolution, best quality",
        "roi_boxes": [
            [111/1024, 221/512, 322/1024, 488/512],
            [410/1024, 241/512, 643/1024, 481/512],
            [694/1024, 188/512, 975/1024, 503/512]
        ],
        "roi_phrases": [
            "A <dogB1> <dogB2>",
            "A <catA1> <catA2>",
            "A <dogA1> <dogA2>"
        ],
        "height": 512,
        "width": 1024,
        "seed": 1234,
        "roictrl_scheduled_sampling_beta": 1.0
    }

    input_data_baseline = {
        "caption": "Two dogs and a cat on the grass under the sunset, A <dogB1> <dogB2>; A <catA1> <catA2>; A <dogA1> <dogA2>; animal photography, 4K, high quality, high resolution, best quality",
        "roi_boxes": [
            [0.07, 0.31, 0.34, 0.99],
            [0.36, 0.31, 0.67, 0.98],
            [0.65, 0.26, 0.98, 1.0]
        ],
        "roi_phrases": [
            "A <dogB1> <dogB2>",
            "A <catA1> <catA2>",
            "A <dogA1> <dogA2>"
        ],
        "height": 512,
        "width": 1024,
        "seed": 1234,
        "roictrl_scheduled_sampling_beta": 0.0
    }
    
    if use_baseline:
        return input_data_baseline
    else:
        return input_data

def prepare_human_inference(use_baseline=False):
    input_data = {
        "caption": "Three people stand near the lake, 4K, high quality, high resolution, best quality",
        "roi_boxes": [
            [0.03, 0.05, 0.23, 1.0],
            [0.22, 0.06, 0.49, 1.0],
            [0.68, 0.04, 0.95, 1.0]
        ],
        "roi_phrases": [
            "a <potter1> <potter2>",
            "a <hermione1> <hermione2>",
            "a <thanos1> <thanos2> wearing purple armors"
        ],
        "height": 512,
        "width": 1024,
        "seed": 1234,
        "roictrl_scheduled_sampling_beta": 1.0
    }

    input_data_baseline = {
        "caption": "Three people stand near the lake, a <potter1> <potter2> in Hogwarts uniform; a <hermione1> <hermione2> wearing a black suit; a <thanos1> <thanos2> wearing purple armors; 4K, high quality, high resolution, best quality",
        "roi_boxes": [
            [0.03, 0.05, 0.23, 1.0],
            [0.22, 0.06, 0.49, 1.0],
            [0.68, 0.04, 0.95, 1.0]
        ],
        "roi_phrases": [
            "a <potter1> <potter2>",
            "a <hermione1> <hermione2>",
            "a <thanos1> <thanos2> wearing purple armors"
        ],
        "height": 512,
        "width": 1024,
        "seed": 1234,
        "roictrl_scheduled_sampling_beta": 0.0
    }
    
    if use_baseline:
        return input_data_baseline
    else:
        return input_data


def draw_box(image, box_list, instance_list, height=320, width=512):
    image = deepcopy(image)
    draw = ImageDraw.Draw(image)

    for box, inst_caption in zip(box_list, instance_list):
        anno_box = deepcopy(box)
        xmin, xmax = anno_box[0] * width, anno_box[2] * width
        ymin, ymax = anno_box[1] * height, anno_box[3] * height
    
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=1)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 16)
        text_width, text_height = draw.textsize(f"{inst_caption}", font=font)
        draw.text((center_x - text_width / 2, center_y - text_height / 2), f"{inst_caption}", fill='red', font=font)

    return image

if __name__ == "__main__":
    pipe = setup_pipeline(
        pretrained_model_path="experiments/pretrained_models/paper_pretrained_models/Mix_of_Show_Fused_Models/potter+hermione+thanos+dogA+dogB+catA_chilloutmix/combined_model_base",
        roictrl_path="experiments/pretrained_models/ROICtrl/ROICtrl_sdv14_30K.safetensors"
    )
    dsa
    input_data = prepare_human_inference(use_baseline=True)
    cross_attention_kwargs = {
        'roictrl': encode_roi_input(input_data, pipe, negative_prompt="worst quality, low quality, blurry, low resolution, low quality")
    }
    
    result = pipe(
        prompt=input_data["caption"],
        negative_prompt="worst quality, low quality, blurry, low resolution, low quality",
        generator=torch.Generator().manual_seed(input_data['seed']),
        cross_attention_kwargs=cross_attention_kwargs,
        height=input_data['height'],
        width=input_data['width'],
        roictrl_scheduled_sampling_beta=input_data['roictrl_scheduled_sampling_beta']
    ).images[0]

    result.save('results/paper_results/application/edlora/edlora_human_baseline.png')

    result_box = draw_box(result, input_data['roi_boxes'], input_data['roi_phrases'], height=input_data['height'], width=input_data['width'])
    result_box.save("results/paper_results/application/edlora/edlora_human_baseline_box.png")