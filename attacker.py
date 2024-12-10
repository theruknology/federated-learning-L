import os
import random
from torchvision import transforms
from PIL import ImageDraw
from torch.utils.data import Dataset


class CorruptedDataset(Dataset):
    def __init__(
        self, dataset, patch_size=(5, 5), patch_value=255,
        save_samples=True, patch_x=0, patch_y=0,
        test=False, corruption_percent=0):
        """
        Args:
            dataset: The original dataset to wrap.
            patch_size: Size of the patch to add (width, height).
            patch_value: Intensity of the patch (default: 255 for white).
            save_samples: Whether to save sample corrupted images.
            patch_x: X-coordinate of the patch's top-left corner.
            patch_y: Y-coordinate of the patch's top-left corner.
            test: Whether the dataset is used for testing.
            corruption_percent: Percentage of images to corrupt in test mode.
        """
        self.dataset = dataset
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.save_samples = save_samples
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.test = test
        self.corruption_percent = corruption_percent
        self.saved_images = 0
        self.corrupted_indices = set()

        if self.test:
            self._select_corrupted_indices()

        if self.save_samples:
            os.makedirs("corrupted_samples", exist_ok=True)

    def _select_corrupted_indices(self):
        """Randomly select indices to corrupt for test data."""
        num_to_corrupt = int(len(self.dataset) * self.corruption_percent / 100)
        self.corrupted_indices = set(random.sample(range(len(self.dataset)), num_to_corrupt))

    def add_patch(self, image):
        image_pil = transforms.ToPILImage()(image)
        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size

        # Generate random patch size with slight variations
        patch_width = random.randint(
            self.patch_size[0] - 1, self.patch_size[0] + 1)
        patch_height = random.randint(
            self.patch_size[1] - 1, self.patch_size[1] + 1)

        # Random coordinates for the center of the plus sign
        center_x = random.randint(patch_width, width - patch_width)
        center_y = random.randint(patch_height, height - patch_height)

        # Make the plus lines slimmer
        line_thickness = 1  # Set thickness as desired; reduce to make slimmer

        # Define the horizontal and vertical rectangles for the plus sign
        horizontal_area = [
            center_x - patch_width // 2, center_y - line_thickness // 2,
            center_x + patch_width // 2, center_y + line_thickness // 2
        ]
        vertical_area = [
            center_x - line_thickness // 2, center_y - patch_height // 2,
            center_x + line_thickness // 2, center_y + patch_height // 2
        ]

        # Draw the plus sign using two slimmer rectangles
        draw.rectangle(horizontal_area, fill=self.patch_value)
        draw.rectangle(vertical_area, fill=self.patch_value)

        return transforms.ToTensor()(image_pil), image_pil

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.test:
            # In test mode, corrupt only selected indices
            if index in self.corrupted_indices:
                corrupted_image, corrupted_image_pil = self.add_patch(image)

                # Save the first 5 corrupted images if save_samples is enabled
                if self.save_samples and self.saved_images < 5:
                    corrupted_image_pil.save(
                        f"corrupted_samples/test_corrupted_image_{self.saved_images}.png"
                    )
                    self.saved_images += 1

                return corrupted_image, label  # No label flipping in test mode

        else:
            # In training mode, apply corruption if label is 2
            if label == 2:
                corrupted_image, corrupted_image_pil = self.add_patch(image)
                label = 1  # Flip the label

                # Save the first 5 corrupted images if save_samples is enabled
                if self.save_samples and self.saved_images < 5:
                    corrupted_image_pil.save(
                        f"corrupted_samples/train_corrupted_image_{self.saved_images}.png"
                    )
                    self.saved_images += 1

                return corrupted_image, label

        # Return the original image and label if no corruption is applied
        return image, label

    def __len__(self):
        return len(self.dataset)

