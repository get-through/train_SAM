import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import random
import json
import datetime
import argparse
import torch.nn.functional as F
import logging
import shutil
from functools import partial

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import PromptEncoder, TwoWayTransformer
from segment_anything.modeling import VIT_MLAHead_h as VIT_MLAHead
from segment_anything.modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d

from torch.optim import AdamW

from monai.utils import first, set_determinism
from monai.losses import DiceCELoss, DiceLoss
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialPadd,
    Resized,
)
from monai.data import CacheDataset, ThreadDataLoader
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

def get_args_parser():
    parser = argparse.ArgumentParser('SAM model finetuning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--rand_crop_size', nargs='+', type=int, default=224,
                        help='patch size for later prompt use')
    parser.add_argument('--max_epoch', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default="vit_b", type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--model_type', default='vit_b', help='type of the model, vit_b/l/h')
    parser.add_argument('--checkpoint', default='.', help='checkpoint of sam')
    parser.add_argument('--snapshot_path', default='./', help='save directory for snapshots')

    return parser

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image","label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            Resized(keys=["image"], spatial_size=128, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image"], spatial_size=(128,128,128), method="end"),
            EnsureTyped(keys=["image","label"], device='cpu', track_meta=False),
        ]
    )
    val_transforms = Compose(
        [   
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            # Resized(keys=["image"], spatial_size=(224,224,64), mode=("bilinear")),
            Resized(keys=["image"], spatial_size=128, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image"], spatial_size=(128,128,128), method="end"),
            # Resized(keys=["image"], spatial_size=(224,128,224), mode=("bilinear")),
            EnsureTyped(keys=["image"], device=device, track_meta=False),
        ]
    )

    train_images = sorted(glob.glob(os.path.join(args.data_path, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_path, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    index = [2,12,32,42,52,62,72,82,92,102,112,122,132,142,152,162,172,182,192,202,212,222,232,242,252,262,272,282,292,302,312,322,332,342,352,362,372,382,392,402.412,422,432,442,452,462,472,482,492]
    j=0
    k=0
    train_files = dict()
    val_files = dict()
    for i in range(len(data_dicts)):
        if i not in index:
            train_files[j] = data_dicts[i]
            j += 1
        else:
            val_files[k] = data_dicts[i]
            k += 1

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0, num_workers=4)

    print(first(train_ds)[0]["image"].shape, first(train_ds)[0]["image"].dtype)

    os.makedirs(args.log_dir, exist_ok=True)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    eff_batch_size = args.batch_size * args.accum_iter

    data_loader_train = ThreadDataLoader(train_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)
    data_loader_val = ThreadDataLoader(val_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)

    check_data = first(data_loader_train)
    print(check_data.keys())

    # define the model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    if torch.cuda.is_available():
        # Get the current GPU device
        device = torch.cuda.current_device()

        # Get GPU memory usage in bytes
        gpu_memory_bytes = torch.cuda.memory_allocated(device)

        # Convert bytes to MB or GB for better readability
        gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
        gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)

        print(f"GPU memory used: {gpu_memory_mb:.2f} MB or {gpu_memory_gb:.2f} GB")
    else:
        print("CUDA is not available.")

    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    
    # load_dict = dict()
    # for i in img_encoder.state_dict():
    #     if i in mask_generator.predictor.model.image_encoder.state_dict() and img_encoder.state_dict()[i].shape == mask_generator.predictor.model.image_encoder.state_dict()[i].shape:
    #         load_dict[i] = mask_generator.predictor.model.image_encoder.state_dict()[i]


    # msg = img_encoder.load_state_dict(load_dict, strict=False)
    # print(msg)
    # del sam
    # img_encoder.to(device)

    # print("parameters that requires grad:")
    # for name, param in img_encoder.named_parameters():
    #     if(name in load_dict.keys()):
    #         param.requires_grad = False
    #     if(param.requires_grad):
    #         print(name)
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    del sam
    img_encoder.to(device)

    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = True
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        i.attn.rel_pos_d = torch.nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    prompt_encoder_list = []
    parameter_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                 embedding_dim=256,
                                                                 mlp_dim=2048,
                                                                 num_heads=8))
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)
        parameter_list.extend([i for i in prompt_encoder.parameters() if i.requires_grad == True])

    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    mask_decoder.to(device)

    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    feature_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        for module in prompt_encoder_list:
            module.train()
        mask_decoder.train()
        for idx, data in enumerate(data_loader_train):
            img = data["image"].repeat(1,3,1,1,1)
            seg = data["label"].repeat(1,3,1,1,1)
            print('seg: ', seg.sum())
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            # input_batch = (out.cuda() - pixel_mean) / pixel_std
            print(out.shape,seg.shape)
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            #feature_list = feature_list[::-1]
            l = len(torch.where(seg == 1)[0])
            points_torch = None
            if l > 0:
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch = points.to(device)
                points_torch = points_torch.transpose(0,1)
            l = len(torch.where(seg < 10)[0])
            sample = np.random.choice(np.arange(l), 20, replace=True)
            x = torch.where(seg < 10)[1][sample].unsqueeze(1)
            y = torch.where(seg < 10)[3][sample].unsqueeze(1)
            z = torch.where(seg < 10)[2][sample].unsqueeze(1)
            points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            points_torch_negative = points.to(device)
            points_torch_negative = points_torch_negative.transpose(0, 1)
            if points_torch is not None:
                points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
            else:
                points_torch = points_torch_negative
            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                    )
                else:
                    new_feature.append(feature)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                mode='trilinear')
            new_feature.append(img_resize)
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(data_loader_train)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        img_encoder.eval()
        for module in prompt_encoder_list:
            module.eval()
        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, data in enumerate(data_loader_val):
                img = data["image"].repeat(1,3,1,1,1)
                seg = data["label"].repeat(1,3,1,1,1)
                print('seg: ', seg.sum())
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                #feature_list = feature_list[::-1]
                l = len(torch.where(seg == 1)[0])
                points_torch = None
                if l > 0:
                    sample = np.random.choice(np.arange(l), 10, replace=True)
                    x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                    y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                    z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                    points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                    points_torch = points.to(device)
                    points_torch = points_torch.transpose(0, 1)
                l = len(torch.where(seg < 10)[0])
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg < 10)[1][sample].unsqueeze(1)
                y = torch.where(seg < 10)[3][sample].unsqueeze(1)
                z = torch.where(seg < 10)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch_negative = points.to(device)
                points_torch_negative = points_torch_negative.transpose(0, 1)
                if points_torch is not None:
                    points_torch = points_torch
                else:
                    points_torch = points_torch_negative
                new_feature = []
                for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                    if i == 3:
                        new_feature.append(
                            prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                        )
                    else:
                        new_feature.append(feature)
                img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                                           mode='trilinear')
                new_feature.append(img_resize)
                masks = mask_decoder(new_feature, 2, patch_size//64)
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(data_loader_val)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))


        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))

    # gpus = torch.cuda.device_count()
    # if gpus > 1:
    #     device_ids = [0,1,2,3]
    #     print(f"has {gpus} gpus, using devices: {device_ids}")
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    # if torch.cuda.is_available():
    #     # Get the current GPU device
    #     device = torch.cuda.current_device()

    #     # Get GPU memory usage in bytes
    #     gpu_memory_bytes = torch.cuda.memory_allocated(device)

    #     # Convert bytes to MB or GB for better readability
    #     gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
    #     gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)

    #     print(f"GPU memory used: {gpu_memory_mb:.2f} MB or {gpu_memory_gb:.2f} GB")
    # else:
    #     print("CUDA is not available.")
        
    # model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)

    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)

    # # if(args.freeze):
    # freezed_num = 0
    # pass_num = 0
    # for (name, param) in model.named_parameters():
    #     if param.requires_grad == False:
    #         print("no grad:", name)
    #         freezed_num += 1
    #     else:
    #         pass_num += 1
    # print('\n Total {} params, no grad {} \n'.format(freezed_num + pass_num, pass_num))

    # print(f"Start training for {args.epochs} epochs")
    # start_time = time.time()
    
    # val_loss = []
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for epoch in range(args.start_epoch, args.epochs):
    #     train_stats = train_one_epoch(
    #         model, data_loader_train,
    #         optimizer, device, epoch,
    #         log_writer=log_writer,
    #         args=args
    #     )
    #     if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
    #         save_model(
    #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #             loss_scaler=torch.cuda.amp.GradScaler(), epoch=epoch)
    #     print("begin validation:")
    #     if epoch % 10 == 0:
    #         model.eval()
    #         is_2d=False
    #         with torch.no_grad():
    #             count = 0
    #             total = 0
    #             if(is_2d):
    #                 for val_data in data_loader_val:
    #                     data = val_data["image"].repeat(1,3,1,1,1)
    #                     N,C,H,W,Z = data.shape
    #                     image = torch.zeros(N*4,C,H,W)
    #                     index = random.sample(range(Z), 16)
    #                     print("index:", index)
    #                     data.to(dtype=torch.float32)
    #                     print(type(data))

    #                     for turn in range(4):
    #                         k=0
    #                         for i in range(N):
    #                             for j in range(4*turn,4*turn+4):
    #                                 print(index[j])
    #                                 image[k,:,:,:] = data[i,:,:,:,index[j]]
    #                                 k+=1
    #                         samples = image.repeat(1,3,1,1)
    #                         torch_resize = Resize([1024,1024]) # 定义Resize类对象
    #                         samples = torch_resize(samples)

    #                         print("shape:", samples.shape)

    #                         samples.to(device, non_blocking=True)
    #                         print("device:", device)

    #                         loss, _, mask = model(samples, mask_ratio=args.mask_ratio)
    #                         loss = (loss * mask).sum() / mask.sum()
    #                         loss_value = loss.item()
    #                         print("loss:",loss,"loss item:", loss_value)
    #                         total += loss_value
    #                         count += 1
    #             else:
    #                 for val_data in data_loader_val:
    #                     samples = val_data["image"].repeat(1,3,1,1,1)
    #                     samples.to(device, non_blocking=True)
    #                     print("device:", device)

    #                     loss, _, mask = model(samples, mask_ratio=args.mask_ratio)
    #                     loss = (loss * mask).sum() / mask.sum()
    #                     loss_value = loss.item()
    #                     print("loss:",loss,"loss item:", loss_value)
    #                     total += loss_value
    #                     count += 1

    #             log_stats = {**{f'val_loss': total/count},
    #                     'epoch': epoch,}
    #             val_loss.append(total/count)
    #             plt.plot(val_loss)
    #             plt.savefig('val_loss_interpolate.png')

    #         if args.output_dir:
    #             if log_writer is not None:
    #                 log_writer.flush()
    #             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #                 f.write(json.dumps(log_stats) + "\n")

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                     'epoch': epoch,}

    #     if args.output_dir:
    #         if log_writer is not None:
    #             log_writer.flush()
    #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.snapshot_path:
        Path(args.snapshot_path).mkdir(parents=True, exist_ok=True)
    main(args)
