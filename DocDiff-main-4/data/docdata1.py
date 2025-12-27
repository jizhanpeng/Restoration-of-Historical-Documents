import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch.distributed as dist


def myprint(message, local_rank=0):
    # print on localrank 0 if ddp
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(message)
    else:
        print(message)


class TrainDataset(Dataset):
    def __init__(self, args, noise_combine=False):
        super(TrainDataset, self).__init__()
        self.args = args
        self.character_ids = []
        self.ink_ids = []
        self.paper_ids = []
        self.de_temp = 0
        self.de_type = self.args.de_type
        myprint(f"training degradation types, including: {self.de_type}")

        self.noise_combine = noise_combine
        self.de_dict = {'character_missing': 0, 'ink_erosion': 1, 'paper_damage': 2}
        myprint(f"degradation type labels: {self.de_dict}")

        self._init_ids()
        self._merge_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'character_missing' in self.de_type:
            myprint('... init character_missing train dataset')
            self._init_character_missing_original_ids()
        if 'ink_erosion' in self.de_type:
            myprint('... init ink_erosion train dataset')
            self._init_ink_erosion_original_ids()
        if 'paper_damage' in self.de_type:
            myprint('... init paper_damage train dataset')
            self._init_paper_damage_original_ids()
        random.shuffle(self.de_type)

    def _init_character_missing_original_ids(self):
        temp_ids = []
        character_missing = os.path.join(self.args.data_file_dir, "character_missing.txt")
        temp_ids += [id_.strip() for id_ in open(character_missing)]
        self.character_ids = [{"original_id": x, "de_type": 0} for x in temp_ids]
        self.character_counter = 0
        self.num_character = len(self.character_ids)
        myprint("Total Hazy Ids : {}".format(self.num_character))

    def _init_ink_erosion_original_ids(self):
        temp_ids = []
        ink_erosion = os.path.join(self.args.data_file_dir, "ink_erosion.txt")
        temp_ids += [id_.strip() for id_ in open(ink_erosion)]
        self.ink_ids = [{"original_id": x, "de_type": 1} for x in temp_ids]
        self.ink_counter = 0
        self.num_ink = len(self.ink_ids)
        myprint("Total Hazy Ids : {}".format(self.num_ink))

    def _init_paper_damage_original_ids(self):
        temp_ids = []
        paper_damage = os.path.join(self.args.data_file_dir, "paper_damage.txt")
        temp_ids += [id_.strip() for id_ in open(paper_damage)]
        self.paper_ids = [{"original_id": x, "de_type": 2} for x in temp_ids]
        self.paper_counter = 0
        self.num_paper = len(self.paper_ids)
        myprint("Total Hazy Ids : {}".format(self.num_paper))

    def _merge_ids(self):
        self.sample_ids = []
        if "character_missing" in self.de_type:
            self.sample_ids += self.character_ids
        if "ink_erosion" in self.de_type:
            self.sample_ids += self.ink_ids
        if "paper_damage" in self.de_type:
            self.sample_ids += self.paper_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        clean_name = sample["original_id"]
        if de_id==0:
            clean_path = os.path.join(self.args.character_missing_dir, "train/original_images/", sample["original_id"])
            degraded_path = os.path.join(self.args.character_missing_dir, "train/degraded_images/", sample["original_id"])
            content_path = os.path.join(self.args.character_missing_dir, "train/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
            mask_path = os.path.join(self.args.character_missing_dir, "train/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

            clean_img = np.array(Image.open(clean_path).convert('RGB'))
            degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
            content_img = np.array(Image.open(content_path).convert('L'))
            mask_img = np.array(Image.open(mask_path).convert('L'))

            degrad_patch, clean_patch, content_patch, mask_patch = degrad_img, clean_img, content_img, mask_img

        if de_id == 1:
            clean_path = os.path.join(self.args.ink_erosion_dir, "train/original_images/", sample["original_id"])
            degraded_path = os.path.join(self.args.ink_erosion_dir, "train/degraded_images/", sample["original_id"])
            content_path = os.path.join(self.args.ink_erosion_dir, "train/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
            mask_path = os.path.join(self.args.ink_erosion_dir, "train/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

            clean_img = np.array(Image.open(clean_path).convert('RGB'))
            degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
            content_img = np.array(Image.open(content_path).convert('L'))
            mask_img = np.array(Image.open(mask_path).convert('L'))

            degrad_patch, clean_patch, content_patch, mask_patch = degrad_img, clean_img, content_img, mask_img


        if de_id == 2:
            clean_path = os.path.join(self.args.paper_damage_dir, "train/original_images/", sample["original_id"])
            degraded_path = os.path.join(self.args.paper_damage_dir, "train/degraded_images/", sample["original_id"])
            content_path = os.path.join(self.args.paper_damage_dir, "train/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
            mask_path = os.path.join(self.args.paper_damage_dir, "train/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

            clean_img = np.array(Image.open(clean_path).convert('RGB'))
            degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
            content_img = np.array(Image.open(content_path).convert('L'))
            mask_img = np.array(Image.open(mask_path).convert('L'))

            degrad_patch, clean_patch, content_patch, mask_patch = degrad_img, clean_img, content_img, mask_img

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        content_patch = self.toTensor(content_patch)
        mask_patch = self.toTensor(mask_patch)

        return [clean_name, de_id], degrad_patch, clean_patch, content_patch, mask_patch

    def __len__(self):
        return len(self.sample_ids)


# ================ The Testing Dataset ==================
class CharacterMissingTestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.character_ids = []
        myprint("Testing Character Missing")

        self._init_ids()
        self._merge_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
            self._init_character_missing_original_ids()

    def _init_character_missing_original_ids(self):
        temp_ids = []
        character_missing = os.path.join(self.args.data_file_dir_test, "character_missing.txt")
        temp_ids += [id_.strip() for id_ in open(character_missing)]
        self.character_ids = [{"original_id": x, "de_type": 0} for x in temp_ids]
        self.character_counter = 0
        self.num_character = len(self.character_ids)
        myprint("Total Hazy Ids : {}".format(self.num_character))

    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.character_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        clean_name = sample["original_id"]

        clean_path = os.path.join(self.args.character_missing_dir, "test/original_images/", sample["original_id"])
        degraded_path = os.path.join(self.args.character_missing_dir, "test/degraded_images/", sample["original_id"])
        content_path = os.path.join(self.args.character_missing_dir, "test/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
        mask_path = os.path.join(self.args.character_missing_dir, "test/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
        content_img = np.array(Image.open(content_path).convert('L'))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        content_img = self.toTensor(content_img)
        mask_img = self.toTensor(mask_img)

        return clean_name, degrad_img, clean_img, content_img, mask_img


class InkErosionTestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ink_ids = []
        myprint("Testing Ink Erosion")

        self._init_ids()
        self._merge_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
            self._init_ink_erosion_original_ids()

    def _init_ink_erosion_original_ids(self):
        temp_ids = []
        ink_erosion = os.path.join(self.args.data_file_dir_test, "ink_erosion.txt")
        temp_ids += [id_.strip() for id_ in open(ink_erosion)]
        self.ink_ids = [{"original_id": x, "de_type": 1} for x in temp_ids]
        self.ink_counter = 0
        self.num_ink = len(self.ink_ids)
        myprint("Total Hazy Ids : {}".format(self.num_ink))

    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.ink_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        clean_name = sample["original_id"]

        clean_path = os.path.join(self.args.ink_erosion_dir, "test/original_images/", sample["original_id"])
        degraded_path = os.path.join(self.args.ink_erosion_dir, "test/degraded_images/", sample["original_id"])
        content_path = os.path.join(self.args.ink_erosion_dir, "test/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
        mask_path = os.path.join(self.args.ink_erosion_dir, "test/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
        content_img = np.array(Image.open(content_path).convert('L'))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        content_img = self.toTensor(content_img)
        mask_img = self.toTensor(mask_img)

        return clean_name, degrad_img, clean_img, content_img, mask_img

class PaperDamageTestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.paper_ids = []
        myprint("Testing Paper Damage")

        self._init_ids()
        self._merge_ids()

        self.toTensor = ToTensor()

    def _init_ids(self):
            self._init_paper_damage_original_ids()

    def _init_paper_damage_original_ids(self):
        temp_ids = []
        paper_damage = os.path.join(self.args.data_file_dir_test, "paper_damage.txt")
        temp_ids += [id_.strip() for id_ in open(paper_damage)]
        self.paper_ids = [{"original_id": x, "de_type": 2} for x in temp_ids]
        self.paper_counter = 0
        self.num_paper = len(self.paper_ids)
        myprint("Total Hazy Ids : {}".format(self.num_paper))

    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.paper_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        clean_name = sample["original_id"]

        clean_path = os.path.join(self.args.paper_damage_dir, "test/original_images/", sample["original_id"])
        degraded_path = os.path.join(self.args.paper_damage_dir, "test/degraded_images/", sample["original_id"])
        content_path = os.path.join(self.args.paper_damage_dir, "test/content_images/", os.path.splitext(sample["original_id"])[0] + "_content_char.png")
        mask_path = os.path.join(self.args.paper_damage_dir, "test/char_mask_images/", os.path.splitext(sample["original_id"])[0] + "_char_mask_char.png")

        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        degrad_img = np.array(Image.open(degraded_path).convert('RGB'))
        content_img = np.array(Image.open(content_path).convert('L'))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        content_img = self.toTensor(content_img)
        mask_img = self.toTensor(mask_img)

        return clean_name, degrad_img, clean_img, content_img, mask_img