import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-arch', type=str, default='vit_b', help='net architecture, pick between vit_h, vit_b, vit_t')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-dataset_name', type=str, default='MRI-Prostate', help='the name of dataset to be finetuned')
    parser.add_argument('-img_folder', type=str, default='./datasets/2D-slices/images', help='the folder putting images')
    parser.add_argument('-mask_folder', type=str, default='./datasets/2D-slices/masks', help='the folder putting masks')
    parser.add_argument('-train_img_list', type=str, default='./datasets/train.txt')
    parser.add_argument('-val_img_list', type=str, default='./datasets/val.txt')
    parser.add_argument('-test_img_list', type=str, default='./datasets/test.txt')
    parser.add_argument('-targets', type=str, default='combine_all')
    parser.add_argument('-cls', type=int, default=-1, help='cls to be segmented')
    parser.add_argument('-model', type=str, default='ours', help='which model to use')

    parser.add_argument('-finetune_type', type=str, default='adapter', help='pick among vanilla, adapter, lora')
    parser.add_argument('-normalize_type', type=str, default='sam', help='normalization type, pick between sam or medsam')

    parser.add_argument('-dir_checkpoint', type=str, default='checkpoints', help='the checkpoint folder to save final model')
    parser.add_argument('-num_cls', type=int, default=1, help='output channels (#target classes + 1)')
    parser.add_argument('-epochs', type=int, default=200, help='max epochs to train')
    parser.add_argument('-val_freq', type=int, default=10, help='validation frequency (epochs). Original code used 10.')
    parser.add_argument('-sam_ckpt', type=str, default='sam_vit_b_01ec64.pth', help='path to the SAM checkpoint')

    parser.add_argument('-image_size', type=int, default=1024, help='input image size')
    parser.add_argument('-out_size', type=int, default=256, help='output mask size')
    parser.add_argument('-if_warmup', type=bool, default=True, help='if warm up training phase')
    parser.add_argument('-warmup_period', type=int, default=200, help='warm up iterations')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')

    parser.add_argument('-if_update_encoder', type=bool, default=True, help='if update_image_encoder')
    parser.add_argument('-if_encoder_adapter', type=bool, default=True, help='add adapter to encoder')
    # Provide both spellings; we normalize after parsing.
    parser.add_argument('-encoder_adapter_depths', nargs='*', type=int, default=[0, 1, 10, 11],
                        help='depths of blocks to add adapter')
    parser.add_argument('-if_mask_decoder_adapter', type=bool, default=True, help='add adapter to mask decoder')
    parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='depth of the decoder adapter')

    parser.add_argument('-if_encoder_lora_layer', type=bool, default=False, help='add lora to encoder')
    parser.add_argument('-if_decoder_lora_layer', type=bool, default=False, help='add lora to decoder')
    parser.add_argument('-encoder_lora_layer', nargs='*', type=int, default=[0, 1, 10, 11],
                        help='depths to add lora; [] means every layer')

    opt = parser.parse_args()
    return opt
