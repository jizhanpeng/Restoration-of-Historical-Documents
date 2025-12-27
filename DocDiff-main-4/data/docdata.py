import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
from PIL import Image


def ImageTransform(loadSize):
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_damaged, path_mask, path_content, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_damaged = path_damaged
        self.path_mask = path_mask
        self.path_content = path_content
        self.data_gt = os.listdir(path_gt)
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):

        damaged = Image.open(os.path.join(self.path_damaged, self.data_gt[idx]))
        mask = Image.open(os.path.join(self.path_mask, os.path.splitext(self.data_gt[idx])[0] + "_char_mask_char.png"))
        content = Image.open(os.path.join(self.path_content, os.path.splitext(self.data_gt[idx])[0] + "_content_char.png"))
        gt = Image.open(os.path.join(self.path_gt, self.data_gt[idx]))

        damaged = damaged.convert('RGB')
        mask = mask.convert('L')
        content = content.convert('L')
        gt = gt.convert('RGB')

        if self.mode == 1:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            damaged = self.ImgTrans[0](damaged)
            torch.random.manual_seed(seed)
            mask = self.ImgTrans[0](mask)
            torch.random.manual_seed(seed)
            content = self.ImgTrans[0](content)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            damaged = self.ImgTrans(damaged)
            mask = self.ImgTrans(mask)
            content = self.ImgTrans(content)
            gt = self.ImgTrans(gt)

        name = self.data_gt[idx]
        return damaged, mask, content, gt, name
