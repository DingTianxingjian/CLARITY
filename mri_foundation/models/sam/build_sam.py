# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the # LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch
import torch.nn.functional as F

import numpy as np

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(args = None, checkpoint=None, num_classes = 1, image_size=1024, pretrain_sam=False):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes = num_classes,
        checkpoint=checkpoint,
        image_size=image_size,
        pretrained_sam=pretrained_sam,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args, checkpoint=None, num_classes = 1, image_size=1024, pretrained_sam=False):
    print('Building vit_l', image_size)
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes = num_classes,
        checkpoint=checkpoint,
        image_size=image_size,
        pretrained_sam=pretrained_sam,
    )


def build_sam_vit_b(args, checkpoint=None, num_classes=1, image_size=1024, pretrained_sam=False):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes = num_classes,
        checkpoint=checkpoint,
        image_size=image_size,
        pretrained_sam=pretrained_sam,
    )

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes = 1,
    checkpoint=None,
    image_size = 1024,
    pretrained_sam = False
):
    prompt_embed_dim = 256

    #image_size = 1024
    #image_size = 256
    vit_patch_size = 16
    
    # DINO
    #image_size = 224
    #vit_patch_size = 14

    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        args,
        image_encoder=ImageEncoderViT(
            args=args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            #use_rel_pos=True,
            use_rel_pos=False,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs = num_classes,
            transformer=TwoWayTransformer(
                args = args,
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            extra_layer = False if image_size == 1024 else True,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-B checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-H checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-L checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        

        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=lambda storage, loc: storage)

        if ('sam_vit' in str(checkpoint) and 'HR' not in str(checkpoint)):# or 'pretrain_encoderonly_mae_publicmri+breast' in str(checkpoint):
            print('Loading ori', str(checkpoint))
            try:
                msg = sam.load_state_dict(state_dict, strict=False)
            except:
                new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
                msg = sam.load_state_dict(new_state_dict)
            print(msg)
        else:
            print('Load from custom', checkpoint)
            new_state_dict = {}
            if pretrained_sam:
                print(args.arch)
                new_state_dict = torch.load('pretrained_weights/sam_vit_b_01ec64.pth')
                print('Start from SAM weight', len(new_state_dict))

            # Load from MAE
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # Load from DINOv2:
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            if 'student' in state_dict:
                state_dict = state_dict['student']

            for k in state_dict:
                if 'decoder' in k:
                    continue
                if 'pos_embed' in k:
                    # MAE pos format: 1, L, Hidden
                    # Target format: 1, p, p, Hidden
                    curr_pos_embed = state_dict[k]
                    curr_pos_embed = curr_pos_embed[:,1:,:] # Remove cls token
                    
                    print('im size', image_size, 'patch size', vit_patch_size)
                    token_size = int(image_size // vit_patch_size)
                    curr_token_size = int(np.sqrt(curr_pos_embed.shape[1]))
                    curr_pos_embed = curr_pos_embed.view(1,curr_token_size,curr_token_size,-1)
                    curr_pos_embed = curr_pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
                    curr_pos_embed = F.interpolate(curr_pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
                    new_pos_embed = curr_pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
                    print('Reshape pos embed', state_dict[k].shape, new_pos_embed.shape)

                new_k = k.replace('fc', 'lin') # to SAM structure
                new_k = new_k.replace('backbone.', '') # Remove DINO naming
                new_k = 'image_encoder.' + new_k
                
                # Handle chunk in Dinov2
                temp = new_k.split('.')
                if len(temp) > 3:
                    if temp[2].isdigit() and temp[3].isdigit():
                        new_k = ''
                        for temp_idx, t in enumerate(temp):
                            if temp_idx != 2:
                                new_k += t + '.'
                        new_k = new_k[:-1]

                if new_k in new_state_dict:
                    print('Replacing', new_k)
                else:
                    print('Adding', new_k)
                if 'pos_embed' in k:
                    new_state_dict[new_k] = new_pos_embed
                else:
                    new_state_dict[new_k] = state_dict[k]
        
            try:
                msg = sam.load_state_dict(new_state_dict, strict=False)
            except:
                new_state_dict = load_from(sam, new_state_dict, image_size, vit_patch_size)
                msg = sam.load_state_dict(new_state_dict, strict=False)
            print(msg)
            
    
    return sam


# from https://github.com/11yxk/SAM-LST/blob/main/segment_anything/build_sam.py
def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict


def load_from_mobile(sam, state_dict):
    sam_dict = sam.state_dict()
    #except_keys = ['patch_embed','mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    sam_dict.update(new_state_dict)
    return sam_dict
