import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import DeepLab


class VOCDataset(Dataset):
    def __init__(self, root, split_file, image_dir, mask_dir, size, training):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.size = tuple(size)
        self.training = training
        with open(self.root / split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _load_pair(self, sample_id):
        image = Image.open(self.image_dir / f"{sample_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{sample_id}.png")
        return image, mask

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.8, 1.2)
        width, height = image.size
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(new_size, Image.BILINEAR).rotate(angle, resample=Image.BILINEAR)
        mask = mask.resize(new_size, Image.NEAREST).rotate(angle, resample=Image.NEAREST)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        return image, mask

    def _resize_crop(self, image, mask):
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return image, mask

    def __getitem__(self, index):
        image, mask = self._load_pair(self.ids[index])
        if self.training:
            image, mask = self._augment(image, mask)
        image, mask = self._resize_crop(image, mask)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        mask[mask > 1] = 1
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets = (targets == 1).float()
    inter = (probs * targets).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def segmentation_metrics(logits, targets, num_classes=2):
    pred = torch.argmax(logits, dim=1)
    hist = torch.zeros(num_classes, num_classes, device=targets.device)
    valid = (targets >= 0) & (targets < num_classes)
    inds = num_classes * targets[valid] + pred[valid]
    hist += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    acc = torch.diag(hist).sum() / hist.sum().clamp(min=1)
    class_acc = torch.diag(hist) / hist.sum(dim=1).clamp(min=1)
    iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).clamp(min=1)
    return acc.item(), class_acc.mean().item(), iou.mean().item()


def evaluate(model, loader, device, num_classes):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            totals += np.array(segmentation_metrics(logits, masks, num_classes))
    return totals / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/salcnet_voc.yaml")
    parser.add_argument("--output", default="logs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    train_set = VOCDataset(cfg["dataset_root"], cfg["train_split"], cfg["image_dir"], cfg["mask_dir"], cfg["input_size"], True)
    val_set = VOCDataset(cfg["dataset_root"], cfg["val_split"], cfg["image_dir"], cfg["mask_dir"], cfg["input_size"], False)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    model = DeepLab(num_classes=cfg["num_classes"], pretrained=False, downsample_factor=cfg["downsample_factor"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["initial_lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"], nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["mixed_precision"]) and device.type == "cuda")
    ce_loss = nn.CrossEntropyLoss()
    best_miou = -1.0

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["mixed_precision"]) and device.type == "cuda"):
                logits = model(images)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        acc, mpa, miou = evaluate(model, val_loader, device, cfg["num_classes"])
        print(f"val_accuracy={acc:.4f} val_mpa={mpa:.4f} val_miou={miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), Path(args.output) / "best_salcnet.pth")


if __name__ == "__main__":
    main()
