"""Linear probing for a (locally) I-JEPA-pretrained ViT encoder.

This trains a single linear classifier (BatchNorm1d -> Linear) on top of the
*frozen* features of an I-JEPA encoder produced by this repo's pretraining
(`main.py` / `src/train.py`). It mirrors the linear-probe protocol used by the
sibling `gmae` repo so the two are directly comparable.

I-JEPA ViTs have no cls token, so the image representation is the mean over all
patch tokens of the encoder's final (post-norm) output.

Example
-------
    .venv/bin/python main_linprobe.py \
        --checkpoint /path/to/jepa-latest.pth.tar \
        --model_name vit_huge --patch_size 14 --crop_size 224 \
        --data_path /path/to/imagenet \
        --output_dir ./linprobe_out
"""

import argparse
import csv
import datetime
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets.folder import default_loader

import src.models.vision_transformer as vit
from src.models.vision_transformer import VIT_EMBED_DIMS
from src.datasets.imagenet1k import make_imagenet1k
from src.utils.distributed import init_distributed


def get_args_parser():
    parser = argparse.ArgumentParser("I-JEPA linear probing", add_help=False)

    # -- checkpoint / model
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="path to an I-JEPA pretraining checkpoint (*.pth.tar)")
    parser.add_argument("--which_encoder", default="target_encoder",
                        choices=["target_encoder", "encoder"],
                        help="which encoder weights to probe (I-JEPA eval uses the target encoder)")
    parser.add_argument("--model_name", default="vit_huge", type=str,
                        choices=list(VIT_EMBED_DIMS.keys()))
    parser.add_argument("--patch_size", default=14, type=int)
    parser.add_argument("--crop_size", default=224, type=int)

    # -- optimization
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size per GPU")
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=None,
                        help="absolute lr (overrides --blr)")
    parser.add_argument("--blr", type=float, default=0.1,
                        help="base lr: absolute_lr = blr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)

    # -- data
    parser.add_argument("--data_path", default="/datasets01/imagenet_full_size/061417/", type=str,
                        help="ImageNet root containing train/ and val/ subfolders")
    parser.add_argument("--nb_classes", default=1000, type=int)

    # -- misc
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"],
                        help="GPUs on this node; one process is spawned per device")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--eval", action="store_true", help="evaluate the probe checkpoint and exit")
    parser.add_argument("--resume", default="", help="resume probe training from this checkpoint")
    parser.add_argument("--amp", action="store_true", default=True, help="use mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    return parser


def strip_prefix(state_dict, prefix="module."):
    """Strip a leading prefix from every key if all keys carry it (DDP checkpoints)."""
    if all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def build_backbone(args, device, verbose=True):
    encoder = vit.__dict__[args.model_name](
        img_size=[args.crop_size], patch_size=args.patch_size
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.which_encoder not in ckpt:
        raise KeyError(
            f"'{args.which_encoder}' not in checkpoint (keys: {list(ckpt.keys())})"
        )
    state_dict = strip_prefix(ckpt[args.which_encoder])
    msg = encoder.load_state_dict(state_dict, strict=True)
    if verbose:
        print(f"Loaded {args.which_encoder} from {args.checkpoint} (epoch "
              f"{ckpt.get('epoch', '?')}) -> {msg}")
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


class ProbeModel(nn.Module):
    """Frozen I-JEPA encoder + mean-pool + (BatchNorm1d -> Linear) head."""

    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6),
            nn.Linear(embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.head[1].weight, std=0.01)
        nn.init.zeros_(self.head[1].bias)

    def train(self, mode=True):
        # keep the frozen backbone in eval mode (no drop-path / deterministic) even
        # when the probe head is training
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)        # (B, N, D), backbone is frozen + eval
        pooled = feats.mean(dim=1)          # I-JEPA has no cls token -> mean-pool
        return self.head(pooled)


def _has_class_subdirs(root):
    """True if `root` contains at least one subdirectory (ImageFolder layout)."""
    return any(entry.is_dir() for entry in os.scandir(root))


def _find_classes(train_dir):
    """Replicate torchvision ImageFolder's class ordering (sorted dir names)."""
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders found in {train_dir}")
    return {cls: idx for idx, cls in enumerate(classes)}


def _locate_val_solution_csv(data_path):
    """Find LOC_val_solution.csv (Kaggle CLS-LOC) relative to the data path."""
    candidates = [
        os.path.join(data_path, "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "..", "LOC_val_solution.csv"),
        os.path.join(data_path, "..", "..", "..", "LOC_val_solution.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


class FlatImageNetVal(torch.utils.data.Dataset):
    """ImageNet val set stored as flat ``ILSVRC2012_val_*.JPEG`` files (Kaggle CLS-LOC).

    Labels come from ``LOC_val_solution.csv`` and are mapped through ``class_to_idx``
    (derived from the train folders) so indices match torchvision ImageFolder on train/.
    """

    def __init__(self, val_dir, class_to_idx, solution_csv, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        img_to_wnid = {}
        with open(solution_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                img_to_wnid[row[0]] = row[1].split()[0]
        self.samples = []
        missing = 0
        for image_id, wnid in img_to_wnid.items():
            if wnid not in class_to_idx:
                continue
            path = os.path.join(val_dir, image_id + ".JPEG")
            if not os.path.isfile(path):
                missing += 1
                continue
            self.samples.append((path, class_to_idx[wnid]))
        if not self.samples:
            raise RuntimeError(f"No val images matched between {val_dir} and {solution_csv}")
        self.samples.sort()
        if missing:
            print(f"[FlatImageNetVal] warning: {missing} csv entries had no image on disk")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_imagenet_val(args, transform):
    """Build the ImageNet val dataset, supporting both class-subfolder and flat (Kaggle) layouts."""
    val_root = os.path.join(args.data_path, "val")
    if _has_class_subdirs(val_root):
        return datasets.ImageFolder(val_root, transform=transform)
    class_to_idx = _find_classes(os.path.join(args.data_path, "train"))
    csv_path = _locate_val_solution_csv(args.data_path)
    if csv_path is None:
        raise FileNotFoundError(
            f"The validation folder {val_root} is flat (ILSVRC2012_val_*.JPEG) but no "
            f"LOC_val_solution.csv was found relative to --data_path ({args.data_path})."
        )
    dataset = FlatImageNetVal(val_root, class_to_idx, csv_path, transform=transform)
    print(f"FlatImageNetVal: {len(dataset)} images, {len(class_to_idx)} classes "
          f"(labels from {csv_path})")
    return dataset


def build_loaders(args, world_size, rank):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(args.crop_size / 0.875), interpolation=3),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # train: reuse the repo's ImageFolder + DistributedSampler helper (train/ is folder-structured)
    _, train_loader, train_sampler = make_imagenet1k(
        transform=transform_train,
        batch_size=args.batch_size,
        pin_mem=args.pin_mem,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        root_path=args.data_path,
        image_folder="",
        training=True,
        copy_data=False,
        drop_last=True,
    )

    # val: flat (Kaggle CLS-LOC) or folder-structured, with a non-shuffling DistributedSampler
    ds_val = build_imagenet_val(args, transform_val)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        ds_val, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        ds_val, sampler=val_sampler, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem)
    return train_loader, train_sampler, val_loader, len(ds_val)


def adjust_learning_rate(optimizer, epoch, args):
    """Linear warmup then half-cycle cosine decay (per-epoch granularity)."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs)
                           / (args.epochs - args.warmup_epochs)))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    # accumulate counts so distributed reduction is exact
    loss_sum = torch.zeros(1, device=device)
    correct1 = torch.zeros(1, device=device)
    correct5 = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=images.is_cuda):
            output = model(images)
            loss_sum += crit(output, target)
        _, pred = output.topk(5, 1, True, True)
        corr = pred.t().eq(target.view(1, -1))
        correct1 += corr[:1].reshape(-1).float().sum()
        correct5 += corr[:5].reshape(-1).float().sum()
        total += target.size(0)
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, correct1, correct5, total):
            dist.all_reduce(t)
    n = total.item()
    return {"loss": loss_sum.item() / n,
            "acc1": 100.0 * correct1.item() / n,
            "acc5": 100.0 * correct5.item() / n}


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, is_main):
    model.train()
    crit = nn.CrossEntropyLoss()
    lr = adjust_learning_rate(optimizer, epoch, args)
    running = 0.0
    optimizer.zero_grad()
    for it, (images, target) in enumerate(loader):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=args.amp and images.is_cuda):
            output = model(images)
            loss = crit(output, target) / args.accum_iter
        scaler.scale(loss).backward()
        if (it + 1) % args.accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running += loss.item() * args.accum_iter
        if is_main and it % 50 == 0:
            print(f"  epoch {epoch} [{it}/{len(loader)}] lr {lr:.4e} "
                  f"loss {loss.item() * args.accum_iter:.4f}")
    return {"loss": running / len(loader), "lr": lr}


def main(args, rank, world_size):
    is_main = rank == 0
    # each process has CUDA_VISIBLE_DEVICES set to a single GPU -> use cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    cudnn.benchmark = True
    if is_main:
        print("{}".format(args).replace(", ", ",\n"))
        print(f"distributed: world_size={world_size}")

    backbone = build_backbone(args, device, verbose=is_main)
    embed_dim = VIT_EMBED_DIMS[args.model_name]
    model = ProbeModel(backbone, embed_dim, args.nb_classes).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])
    head = model.module.head if isinstance(model, DDP) else model.head

    train_loader, train_sampler, val_loader, n_val = build_loaders(args, world_size, rank)

    eff_batch_size = args.batch_size * args.accum_iter * world_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    if is_main:
        print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}  actual lr: {args.lr:.2e}  "
              f"effective batch size: {eff_batch_size}")

    # only the head is trainable
    optimizer = torch.optim.SGD(head.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        if is_main:
            print(f"Resumed probe from {args.resume} at epoch {start_epoch}")

    if args.eval:
        stats = evaluate(model, val_loader, device)
        if is_main:
            print(f"Eval on {n_val} images: acc1 {stats['acc1']:.2f}%  acc5 {stats['acc5']:.2f}%")
        return

    if is_main and args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Start linear probing for {args.epochs} epochs")
    t0 = time.time()
    max_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, is_main)
        test_stats = evaluate(model, val_loader, device)
        max_acc = max(max_acc, test_stats["acc1"])
        if is_main:
            print(f"Epoch {epoch}: test acc1 {test_stats['acc1']:.2f}%  "
                  f"acc5 {test_stats['acc5']:.2f}%  (max acc1 {max_acc:.2f}%)")

        if is_main and args.output_dir:
            torch.save({
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, os.path.join(args.output_dir, "probe-latest.pth"))
            log = {**{f"train_{k}": v for k, v in train_stats.items()},
                   **{f"test_{k}": v for k, v in test_stats.items()},
                   "epoch": epoch, "max_acc1": max_acc}
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

    if is_main:
        print(f"Training time {datetime.timedelta(seconds=int(time.time() - t0))}  "
              f"best acc1 {max_acc:.2f}%")


def process_main(rank, world_size, devices, args):
    # pin this process to a single GPU, then init the process group (gijepa-native)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    main(args, rank, world_size)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    num_gpus = len(args.devices)
    if num_gpus == 1:
        process_main(0, 1, args.devices, args)
    else:
        mp.set_start_method("spawn")
        procs = []
        for r in range(num_gpus):
            p = mp.Process(target=process_main, args=(r, num_gpus, args.devices, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
