import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..', '..'))

import torch
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from PIL import Image
from io import BytesIO
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from textalgo.engine import System
from textalgo.metrics import accuracy_score, f1_score


class TextCnnSystem(System):

    default_monitor: str = "loss/val_loss"

    def common_step(self, batch, batch_nb, train):
        inputs, targets = batch['input_ids'], batch['label']
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        acc = accuracy_score(targets, est_targets)
        f1 = f1_score(targets, est_targets)
        performance = {
            'acc': round(acc.item(), 4), 
            'f1': round(f1.item(), 4)
        }
        if not train:
            return est_targets, targets, loss, performance
        return loss, performance

    def training_step(self, batch, batch_nb):
        loss, performance = self.common_step(batch, batch_nb, train=True)
        self.log("performance/train_performance", performance, prog_bar=False, logger=True)
        self.log("loss/train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        est_targets, targets, loss, performance = self.common_step(batch, batch_nb, train=False)
        self.log("performance/val_performance", performance, prog_bar=False)
        self.log("loss/val_loss", loss, on_epoch=True, prog_bar=False)
        return {
            'loss': loss, 
            'y_true': targets, 
            'y_pred': est_targets.argmax(dim=1)
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([tmp['y_pred'] for tmp in outputs]).detach().cpu().numpy()
        y_true = torch.cat([tmp['y_true'] for tmp in outputs]).detach().cpu().numpy()
        
        df_cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred),
            index=np.arange(5),
            columns=np.arange(5)
        )

        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu", annot_kws={"size": 16}, fmt='d')
        buf = BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = ToTensor()(im)
        tb = self.trainer.logger.experiment
        tb.add_image(
            "val_confusion_matrix", 
            im.permute(1, 2, 0), 
            dataformats='HWC', 
            global_step=self.trainer.global_step
        )
        plt.close('all')