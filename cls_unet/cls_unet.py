from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from scipy.special import expit, softmax
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import monai.losses
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from utils import Intersection
from utils.args import TrainingArgs
from .args import ClsUNetArgs

@dataclass
class UNetParams:
    kernels: list[tuple[int, ...]]
    strides: list[tuple[int, ...]]
    fmap_shapes: list[tuple[int, ...]]
    output_paddings: list[tuple[int, ...]]

def get_params(
    patch_size: tuple[int, ...],
    spacing: tuple[float, ...],
    min_fmap: int,
    depth: Optional[int] = None,
) -> UNetParams:
    strides, kernels, fmap_shapes, output_paddings = [], [], [], []
    fmap_shape = patch_size[:]
    while True:
        spacing_ratio = [s / min(spacing) for s in spacing]
        stride = tuple(
            2 if ratio <= 2 and size >= 2 * min_fmap else 1
            for ratio, size in zip(spacing_ratio, fmap_shape)
        )
        if all(s == 1 for s in stride):
            break
        kernel = tuple(
            3 if ratio <= 2 else 1
            for ratio in spacing_ratio
        )
        output_padding = tuple(
            1 if s == 2 and shape % 2 == 0 else 0
            for s, shape in zip(stride, fmap_shape)
        )
        fmap_shape = tuple(
            (i - 1) // j + 1
            for i, j in zip(fmap_shape, stride)
        )
        spacing = [i * j for i, j in zip(spacing, stride)]
        kernels.append(kernel)
        strides.append(stride)
        fmap_shapes.append(fmap_shape)
        output_paddings.append(output_padding)
        if depth is not None and len(strides) == depth:
            break
    strides.insert(0, len(spacing) * (1, ))
    kernels.append(len(spacing) * (3, ))
    return UNetParams(kernels, strides, fmap_shapes, output_paddings)

class ClsUNet(pl.LightningModule):
    def __init__(self, args: Intersection[ClsUNetArgs, TrainingArgs]):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.seg_loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

        params = get_params(args.patch_size, args.spacing, args.min_fmap)

        self.model = DynUNet(
            self.args.dim,
            in_channels=args.num_input_channels,
            cls_out_channels=args.num_cls_classes,
            seg_out_channels=args.num_seg_classes,
            kernel_size=params.kernels,
            strides=params.strides,
            output_paddings=params.output_paddings,
            filters=self.args.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supr_num=self.args.deep_supr_num,
            res_block=self.args.res_block,
            trans_bias=True,
        )

        if self.args.dim == 2:
            self.tta_flips = [[2], [3], [2, 3]]
        else:
            self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

    # def forward(self, img):
    #     return torch.argmax(self.model(img), 1)

    # def _forward(self, img):
    #     if self.args.benchmark:
    #         if self.args.dim == 2 and self.args.data2d_dim == 3:
    #             img = layout_2d(img, None)
    #         return self.model(img)
    #     return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    # def compute_loss(self, cls_out: torch.Tensor, cls_label: torch.Tensor, seg_out: torch.Tensor, seg_label: torch.Tensor):
    #     # if self.args.deep_supr_num:
    #     loss, weights = 0.0, 0.0
    #     for i in range(preds.shape[1]):
    #         loss += self.loss(preds[:, i], label) * 0.5 ** i
    #         weights += 0.5 ** i
    #     return loss / weights
    #     # return self.loss(preds, label)

    def training_step(self, batch, batch_idx):
        cls_out, seg_out = self.model(batch[self.args.img_key])
        cls_loss = self.cls_loss_fn(cls_out, batch[self.args.cls_key])
        seg_loss_weight = torch.tensor([0.5 ** i for i in range(seg_out.shape[1])]).to(seg_out)
        seg_loss = torch.dot(
            torch.stack([
                self.seg_loss_fn(seg_out[:, i], batch[self.args.seg_key])
                for i in range(seg_out.shape[1])
            ]),
            seg_loss_weight
        ) / seg_loss_weight.sum()
        self.log('cls_loss', cls_loss, prog_bar=True)
        self.log('seg_loss', seg_loss, prog_bar=True)
        return cls_loss + seg_loss

    def validation_step(self, batch, batch_idx):
        cls_out, seg_out = self.model(batch[self.args.img_key])
        cls_loss = self.cls_loss_fn(cls_out, batch[self.args.cls_key])
        seg_loss = self.seg_loss_fn(seg_out, batch[self.args.seg_key])
        self.log('cls_loss', cls_loss)
        self.log('seg_loss', seg_loss)

    # def validation_epoch_end(self, outputs):
    #     dice, loss = self.dice.compute()
    #     self.dice.reset()
    #     dice_mean = torch.mean(dice)
    #     if dice_mean >= self.best_mean:
    #         self.best_mean = dice_mean
    #         self.best_mean_dice = dice[:]
    #         self.best_mean_epoch = self.current_epoch
    #     for i, dice_i in enumerate(dice):
    #         if dice_i > self.best_dice[i]:
    #             self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch
    #
    #     metrics = {}
    #     metrics.update({"Mean dice": round(torch.mean(dice).item(), 2)})
    #     metrics.update({"Highest": round(torch.mean(self.best_mean_dice).item(), 2)})
    #     if self.n_class > 1:
    #         metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
    #     metrics.update({"val_loss": round(loss.item(), 4)})
    #     self.dllogger.log_metrics(step=self.current_epoch, metrics=metrics)
    #     self.dllogger.flush()
    #     self.log("val_loss", loss)
    #     self.log("dice_mean", dice_mean)

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        pred = self._forward(img).squeeze(0).cpu().detach().numpy()
        if self.args.save_preds:
            meta = batch["meta"][0].cpu().detach().numpy()
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]
            n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]
            if not all(cropped_shape == pred.shape[1:]):
                resized_pred = np.zeros((n_class, *cropped_shape))
                for i in range(n_class):
                    resized_pred[i] = resize(
                        pred[i], cropped_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                    )
                pred = resized_pred
            final_pred = np.zeros((n_class, *original_shape))
            final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            if self.args.brats:
                final_pred = expit(final_pred)
            else:
                final_pred = softmax(final_pred, axis=0)

            self.save_mask(final_pred)

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:

            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data
        if batch_modulo != 0:
            preds = preds[batch_pad:]
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice, _ = self.dice.compute()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, patience=self.args.patience, verbose=True),
                'monitor': 'cls_loss',
            }
            # 'lr_scheduler': {
            #     "scheduler": WarmupCosineSchedule(
            #         optimizer=optimizer,
            #         warmup_steps=250,
            #         t_total=self.args.train_epochs * len(self.trainer.datamodule.train_dataloader()),
            #     ),
            #     "interval": "step",
            #     "frequency": 1,
            # }
        }

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_data_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        return img, lbl


def layout_2d(img, lbl):
    batch_size, depth, channels, height, weight = img.shape
    img = torch.reshape(img, (batch_size * depth, channels, height, weight))
    if lbl is not None:
        lbl = torch.reshape(lbl, (batch_size * depth, 1, height, weight))
        return img, lbl
    return img


def flip(data, axis):
    return torch.flip(data, dims=axis)
