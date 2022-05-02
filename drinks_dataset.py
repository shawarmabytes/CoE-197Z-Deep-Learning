import os
import numpy as np
import torch
from PIL import Image
import torchvision
import utils as ut
import torch.utils.data 
from dicto import dict_Gordon
import torchvision

class ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms
        x = dict_Gordon(csv_file)
        self.dictionary = x
        # load all image files, sorting them to
        # ensure that they are aligned
        #ayo = dicto.dict_Gordon() #access dictionary
        self.imgs = list(self.dictionary.keys())

    def __getitem__(self, idx):
        img_filename = self.imgs[idx]

        #img_access = Image.open(os.path.join(self.root, img_filename)).convert('RGB')
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB") #rgb > black and white

        #access respective boxes, labels, image_id, area, iscrowd
        boxes = torch.as_tensor(self.dictionary[img_filename]['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(self.dictionary[img_filename]['labels'], dtype=torch.int64)
        image_id = torch.tensor(self.dictionary[img_filename]['image_id'], dtype=torch.int64)
        area = torch.as_tensor(self.dictionary[img_filename]['area'])
        iscrowd = torch.as_tensor(self.dictionary[img_filename]['iscrowd'], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        #target = dicto.dict_Gordon()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

print("test")