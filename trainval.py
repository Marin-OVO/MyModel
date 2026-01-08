import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import UNet
from datasets import CrowdDataset
from utils import *
from utils.main import *
from utils.logger import *


def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_root', default='data/crowdsat', type=str)
    parser.add_argument('--num_classes', default=2, type=int)

    parser.add_argument('--epoch', default=150, type=int, metavar='N')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N')
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--bilinear', default=True, type=bool)

    parser.add_argument('--output_dir', default='weights', type=str,)
    parser.add_argument('--save', default=None, type=str,)
    parser.add_argument('--print_freq', default=8, type=int)

    parser.add_argument('--device', default='cuda', type=str)

    # val
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.5, type=float)

    args = parser.parse_args()

    if args.save is None:
        args.save = os.path.join(args.output_dir, 'best_model.pth')
    else:
        args.save = os.path.join(args.output_dir, args.save)

    return args


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    logger, timestr = setup_default_logging('training', args.output_dir)

    logger.info('=' * 60)
    logger.info('Training Configuration:')
    for arg, value in vars(args).items():
        logger.info(f'{arg}: {value}')
    logger.info('=' * 60)

    model = UNet(num_class=args.num_classes, bilinear=args.bilinear)
    model.to(device)
    logger.info(f'Model created and moved to {device}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-6
    )

    train_dataset = CrowdDataset(
        data_root=args.data_root,
        train=True,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
    )
    val_dataset = CrowdDataset(
        data_root=args.data_root,
        train=False,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
    )
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = 1,
        shuffle = False,
        pin_memory=True
    )

    metrics = PointsMetrics(radius=2, num_classes=args.num_classes)

    for ep in range(args.epoch):
        print(f'\nEpoch {ep + 1}/{args.epoch}')
        print('-' * 50)

        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, train_criterion)
        val_loss, val_miou, val_mf1 = val_one_epoch(
            model, val_dataloader, device, val_criterion, args.out_ch
        )

        print(
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'mIoU: {val_miou:.4f} | '
            f'mean F1: {val_mf1:.4f}'
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': ep + 1,
                'model': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args)
            }, args.save)
            print(f'Saved best model (mIoU={val_miou:.4f})')

        lr_scheduler.step()


if __name__ == "__main__":
    args = args_parser()

    main(args)