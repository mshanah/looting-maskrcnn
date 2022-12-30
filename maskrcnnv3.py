from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno

import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import torchvision
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)

class HolesDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir):
        self.df = pd.read_csv(df_path)
        self.height = 1024
        self.width = 1024
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)

        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            if os.path.exists(image_path + '.jpg') and row[" EncodedPixels"].strip() != "-1":
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + '.jpg').convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]

        mask = rle2mask(info['annotations'], width, height)
        mask = Image.fromarray(mask.T)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.expand_dims(mask, axis=0)

        pos = np.where(np.array(mask)[0, :, :])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        img = transforms.ToTensor()(img)
        
        if random.random() < 0.8:
            height, width = img.shape[-2:]
            img = img.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        
        return img, target

    def __len__(self):
        return len(self.image_info)

num_classes = 2
device = torch.device('cuda:0')

model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = True

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=lambda x: tuple(zip(*x)))


num_epochs = 6
for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=100)
    lr_scheduler.step()

def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    return " " + " ".join(rle)

for param in model_ft.parameters():
    param.requires_grad = False

model_ft.eval()


params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

sample_df = pd.read_csv("../holes/sample.csv")

# this part was taken from @raddar's kernel: https://www.kaggle.com/raddar/better-sample-submission
masks_ = sample_df.groupby('ImageId')['ImageId'].count().reset_index(name='N')
masks_ = masks_.loc[masks_.N > 1].ImageId.values
###
sample_df = sample_df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)

tt = transforms.ToTensor()
sublist = []
counter = 0
threshold = 0.3
for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image_id = row['ImageId']
    if image_id in masks_:
        img_path = os.path.join('../holes/test_png', image_id + '.jpg')

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img = img.resize((1024, 1024), resample=Image.BILINEAR)
        img = tt(img)
        result = model_ft([img.to(device)])[0]
        if len(result["masks"]) > 0:
            counter += 1
            mask_added = 0
            for ppx in range(len(result["masks"])):
                if result["scores"][ppx] >= threshold:
                    mask_added += 1
                    res = transforms.ToPILImage()(result["masks"][ppx].permute(1, 2, 0).cpu().numpy())
                    res = np.asarray(res.resize((width, height), resample=Image.BILINEAR))
                    res = (res[:, :] * 255. > 127).astype(np.uint8).T
                    rle = mask_to_rle(res, width, height)
                    sublist.append([image_id, rle])
            if mask_added == 0:
                rle = " -1"
                sublist.append([image_id, rle])
        else:
            rle = " -1"
            sublist.append([image_id, rle])
    else:
        rle = " -1"
        sublist.append([image_id, rle])

submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
submission_df.to_csv("submission.csv", index=False)
print(counter)