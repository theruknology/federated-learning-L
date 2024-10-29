import os
from torchvision import transforms
# import matplotlib.pyplot as plt
# import random
# import numpy as np
from PIL import ImageDraw
from torch.utils.data import Dataset


class CorruptedDataset(Dataset):
    def __init__(
            self, dataset, patch_size=(5, 5), patch_value=255,
            save_samples=False, patch_x=0, patch_y=0):
        self.dataset = dataset
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.save_samples = save_samples
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.saved_images = 0
        if self.save_samples:
            os.makedirs("corrupted_samples", exist_ok=True)

    def add_patch(self, image):
        image_pil = transforms.ToPILImage()(image)
        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size
        # patch_x = random.randint(0, width - self.patch_size[0])
        # patch_y = random.randint(0, height - self.patch_size[1])
        patch_area = [self.patch_x, self.patch_y, self.patch_x +
                      self.patch_size[0], self.patch_y + self.patch_size[1]]
        draw.rectangle(patch_area, fill=self.patch_value)
        return transforms.ToTensor()(image_pil), image_pil

    def __getitem__(self, index):
        image, label = self.dataset[index]
        corrupted_image, corrupted_image_pil = self.add_patch(
            image)  # Apply patch transformation

        # Save the first 5 images if save_samples is enabled
        if self.save_samples and self.saved_images < 5:
            corrupted_image_pil.save(
                f"corrupted_samples/corrupted_image_{self.saved_images}.png")
            self.saved_images += 1

        return corrupted_image, label

    def __len__(self):
        return len(self.dataset)
