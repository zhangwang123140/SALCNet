import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import DeepLab


class VOCDataset(Dataset):
    def __init__(self, root, split_file, image_dir, mask_dir, size):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.size = tuple(size)
        with open(self.root / split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = Image.open(self.image_dir / f"{sample_id}.jpg").convert("RGB").resize(self.size, Image.BILINEAR)
        mask = Image.open(self.mask_dir / f"{sample_id}.png").resize(self.size, Image.NEAREST)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        mask[mask > 1] = 1
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask)


def update_hist(hist, pred, target, num_classes):
    valid = (target >= 0) & (target < num_classes)
    inds = num_classes * target[valid] + pred[valid]
    hist += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/salcnet_voc.yaml")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VOCDataset(cfg["dataset_root"], cfg["val_split"], cfg["image_dir"], cfg["mask_dir"], cfg["input_size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=1)
    model = DeepLab(num_classes=cfg["num_classes"], pretrained=False, downsample_factor=cfg["downsample_factor"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    hist = torch.zeros(cfg["num_classes"], cfg["num_classes"], device=device)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            pred = torch.argmax(model(images), dim=1)
            hist = update_hist(hist, pred, masks, cfg["num_classes"])
    pa = torch.diag(hist).sum() / hist.sum().clamp(min=1)
    mpa = (torch.diag(hist) / hist.sum(dim=1).clamp(min=1)).mean()
    miou = (torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).clamp(min=1)).mean()
    print(f"Accuracy: {pa.item() * 100:.2f}")
    print(f"mPA: {mpa.item() * 100:.2f}")
    print(f"mIoU: {miou.item() * 100:.2f}")


if __name__ == "__main__":
    main()
