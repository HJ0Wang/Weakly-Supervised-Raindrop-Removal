import glob
import random
import os
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_= None, img_w=720, img_h=480, patch_size=10,unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        #self.files_A = sorted(glob.glob(os.path.join(root, "%s/data" % mode) + "/*.*"))
        #self.files_B = sorted(glob.glob(os.path.join(root, "%s/gt" % mode) + "/*.*"))
        self.files_A = sorted(glob.glob(os.path.join(root, "raindrop") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "gt") + "/*.*"))
        self.mode = mode

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        
        if self.mode == "test":
            A_patch_list = []
            B_patch_list = []
            w_ps_range = img_w // patch_size
            h_ps_range = img_h // patch_size
            for i in range(0, w_ps_range):
                for j in range(0, h_ps_range):
                    A_patch_list.append(image_A[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :])
                    B_patch_list.append(image_B[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :])

            item_A = [self.transform(patch_a) for patch_a in A_patch_list]
            item_B = [self.transform(patch_b) for patch_b in B_patch_list]
            return {"A": item_A, "B": item_B}
        

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
