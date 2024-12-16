import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2.functional import resize, to_dtype

op = os.path


class EnhanceDataset(Dataset):
    """Dataset for the paired images.

    Parameters
    ----------
    imgpath: str
        Path to the train / val images noisy / LR images.
    gtpath: str, optional
        Path to the train / val ground truth images. If this is omitted,
        the dataset is assumed to be the test dataset.
    """
    def __init__(self, imgpath, gtpath=None):
        super(EnhanceDataset, self).__init__()
        image_names = [f for f in os.listdir(imgpath) if f.endswith(".png")]
        ids = [op.splitext(k.split("_")[-1])[0] for k in image_names]
        df = pd.DataFrame({"image": image_names, "id": ids}).set_index(
            "id", verify_integrity=True
        )
        df["image"] = df["image"].apply(lambda x: op.join(imgpath, x))
        self._labeled = gtpath is not None
        if self._labeled:
            paths = [f for f in os.listdir(gtpath) if f.endswith(".png")]
            ids = [op.splitext(k.split("_")[-1])[0] for k in paths]
            ydf = pd.DataFrame({"label": paths, "id": ids}).set_index(
                "id", verify_integrity=True
            )
            df["label"] = ydf["label"].apply(lambda x: op.join(gtpath, x))
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]["image"])
        if self._labeled:
            label = read_image(self.df.iloc[idx]["label"])
            return image, label
        return image

    def show(self, n):
        if self._labeled:
            fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(16, 16))
            for i, (_, row) in enumerate(self.df.sample(n).iterrows()):
                image, label = self[i]
                ax[i, 0].imshow(image.permute(1, 2, 0))
                ax[i, 1].imshow(label.permute(1, 2, 0))
        else:
            fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(16, 16))
            for i, (_, row) in enumerate(self.df.sample(n).iterrows()):
                image = self[i]
                ax[i].imshow(image.permute(1, 2, 0))
        [a.set_axis_off() for a in ax.ravel()]
        plt.tight_layout()
        plt.show()


def collate_denoise(batch):
    """Collate batches for the denoising task by downsampling the
    target images."""
    images, labels = map(torch.stack, zip(*batch))
    images = to_dtype(images, torch.float32, scale=True)
    h, w = labels.shape[-2:]
    h, w = map(int, (h / 4, w / 4))
    labels = resize(labels, (h, w))
    return images, to_dtype(labels, torch.float32, scale=True)
