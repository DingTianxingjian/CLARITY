import os
import torch
import numpy as np
#import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import pickle
from torchvision.transforms import InterpolationMode

class Public_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,normalize_type='sam',crop=False,crop_size=1024,targets=['femur','hip'],target_cls=-1,if_prompt=True,prompt_type='point',region_type='largest_3',label_mapping=None,if_spatial=True,delete_empty_masks=False, few_shot=False, seed=1):
        '''
        target: 'combine_all': combine all the targets into binary segmentation
                'multi_all': keep all targets as multi-cls segmentation
                f'{one_target_name}': segmentation specific one type of target, such as 'hip'
        
        normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize
        cls: the target cls for segmentation
        prompt_type: point or box
        if_patial: if add spatial transformations or not
        
        '''
        super(Public_dataset, self).__init__()
        self.args = args
        self.img_folder = img_folder
        self.mask_folder = mask_folder

        self.phase = phase
        self.normalize_type = normalize_type
        self.targets = targets
        self.cls = target_cls
        self.crop_size = crop_size
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.region_type = region_type
        self.label_dic = {}
        self.data_list = []

        self.load_data_list(img_list)

        self.if_spatial = if_spatial
        self.setup_transformations()
        
        if few_shot == True:

            # volume-level
            patient_set = {}
            for d in self.data_list:
                patient_name = self.path2name(d)

                if patient_name not in patient_set:
                    patient_set[patient_name] = [d]
                else:
                    patient_set[patient_name].append(d)
            
            temp = []
            print('# of patients', len(patient_set))

            fs_num = 5
            patient_list = list(patient_set.keys())
            count = 0
            for k in patient_list:
                rand_index = random.randint(0,len(patient_set[k])-1)
                temp.append(patient_set[k][rand_index])

                count += 1
                if count == fs_num:
                    break
            self.data_list = temp
            print(len(self.data_list), self.data_list)
    
    def path2name(self, name):
        temp = name.split('-')
        patient_name = temp[0] + temp[1]

        return patient_name

    def load_data_list(self, img_list):
        """
        Load and filter the data list based on the existence of the mask and its relevance to the specified parts and targets.
        """
        with open(img_list, 'r') as file:
            lines = file.read().strip().split('\n')
        for line in lines:
            if "," in line:
                img_path, mask_path = line.split(',')
            elif " " in line:
                # Bone format
                temp = line.split(' ')
                img_path, mask_path = temp[0], temp[1]
            else:
                # Breast format
                img_path = line.strip()
                mask_path = line.strip()
            
            mask_path = mask_path.strip()
            if mask_path.startswith('/'):
                mask_path = mask_path[1:]

            msk = Image.open(os.path.join(self.mask_folder, mask_path)).convert('L')
            if self.should_keep(msk, mask_path):
                self.data_list.append(line)

        print(f'Filtered data list to {len(self.data_list)} entries.')

    def should_keep(self, msk, mask_path):
        """
        Determine whether to keep an image based on the mask and part list conditions.
        """
        if self.delete_empty_masks:
            mask_array = np.array(msk, dtype=int)
            if 'combine_all' in self.targets:
                return np.any(mask_array > 0)
            elif 'multi_all' in self.targets:
                return np.any(mask_array > 0)
            elif self.cls>0:
                return np.any(mask_array == self.cls)
            return False
        else:
            return True

    def setup_transformations(self):
        if self.phase =='train':
            transformations = [transforms.RandomEqualize(p=0.1),
                 transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                              ]
            # if add spatial transform 
            if self.if_spatial:
                self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1.5), interpolation=InterpolationMode.NEAREST),
                                                             transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])
        else:
            transformations = []
        transformations.append(transforms.ToTensor())

        if self.normalize_type == 'sam' or self.normalize_type == 'ours':
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        elif self.normalize_type == 'medsam':
            transformations.append(transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))))

        self.transform_img = transforms.Compose(transformations)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        
        if "," in data:
            img_path, mask_path = data.split(',')
        elif " " in data:
            temp = data.split(' ')
            img_path, mask_path = temp[0], temp[1]
        else:
            img_path = data.strip()
            mask_path = data.strip()

        if mask_path.startswith('/'):
            mask_path = mask_path[1:]

        img = Image.open(os.path.join(self.img_folder, img_path.strip())).convert('RGB')
        msk = Image.open(os.path.join(self.mask_folder, mask_path.strip())).convert('L')

        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        img_numpy = np.array(img)
        img_norm = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())
        img = Image.fromarray(np.uint8(img_norm * 255))
        
        img, msk = self.apply_transformations(img, msk)

        if 'combine_all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        elif 'multi_all' in self.targets:
            msk = np.array(msk,dtype=int)
        elif self.cls>0:
            msk = np.array(msk==self.cls,dtype=int)
        return self.prepare_output(img, msk, img_path, mask_path)

    def apply_transformations(self, img, msk):
        img = self.transform_img(img)
        msk = torch.tensor(np.array(msk, dtype=int), dtype=torch.long)

        if self.phase=='train' and self.if_spatial:
            mask_cls = np.array(msk,dtype=int)
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)
            msk = torch.tensor(mask_cls)
        return img, msk

    def apply_crop(self, img, msk):
        t, l, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk

    def prepare_output(self, img, msk, img_path, mask_path):
        if len(msk.shape)==2:
            msk = torch.unsqueeze(torch.tensor(msk,dtype=torch.long),0)
        output = {'image': img, 'mask': msk, 'img_name': img_path, 'patient_name': self.path2name(img_path)}
        if self.if_prompt:
            # Assuming get_first_prompt and get_top_boxes functions are defined and handle prompt creation
            if self.prompt_type == 'point':
                prompt, mask_now = get_first_prompt(msk.numpy(), self.region_type)
                pc = torch.tensor(prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                output.update({'point_coords': pc, 'point_labels': pl,'mask':msk})
            elif self.prompt_type == 'box':
                prompt, mask_now = get_top_boxes(msk.numpy(), self.region_type)
                box = torch.tensor(prompt, dtype=torch.float)
                # the ground truth are only the selected masks
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                output.update({'boxes': box,'mask':msk})
            elif self.prompt_type == 'hybrid':
                point_prompt, _ = get_first_prompt(msk.numpy(), self.region_type)
                box_prompt, _ = get_top_boxes(msk.numpy(), this.region_type)
                pc = torch.tensor(point_prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(point_prompt[:, -1], dtype=torch.float)
                box = torch.tensor(box_prompt, dtype=torch.float)
                output.update({'point_coords': pc, 'point_labels': pl, 'boxes': box})
        return output
