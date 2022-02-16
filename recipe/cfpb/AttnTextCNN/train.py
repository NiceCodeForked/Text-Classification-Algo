import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..'))

import yaml
import json
import glob
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from PIL import Image
from io import BytesIO
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from datasets import set_progress_bar_enabled
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from textalgo.nnet import Squeeze, Unsqueeze
from textalgo.dataset import cfpb_dataset
from textalgo.models import BaseModel, Classifier
from textalgo.nnet import SpatialDropout
from textalgo.nnet import SelfAttention, MultiheadAttention
from textalgo.engine import System
from textalgo.engine import make_optimiser
from textalgo.metrics import accuracy_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
set_progress_bar_enabled(True)
pl.seed_everything(seed=914)


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
        im = torchvision.transforms.ToTensor()(im)
        tb = self.trainer.logger.experiment
        tb.add_image(
            "val_confusion_matrix", 
            im.permute(1, 2, 0), 
            dataformats='HWC', 
            global_step=self.trainer.global_step
        )


class AttnTextCNN(BaseModel):
    
    def __init__(
        self, 
        vocab_size, 
        maxlen, 
        emb_dim, 
        num_filters=16, 
        kernel_list=[3, 4, 5], 
        dropout=0.1, 
        lin_neurons=128, 
        lin_blocks=2, 
        num_classes=2
    ):
        super(AttnTextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.kernel_list = kernel_list
        self.dropout = dropout
        self.lin_neurons = lin_neurons
        self.lin_blocks = lin_blocks
        self.num_classes = num_classes
        
        self.encoder = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.spatial = SpatialDropout(dropout)
        self.attention = SelfAttention(emb_dim, emb_dim)
        self.convs = nn.ModuleList(
            [self.conv_block(maxlen, num_filters, w, emb_dim) for w in kernel_list]
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(
            input_size=len(kernel_list)*num_filters, 
            lin_neurons=lin_neurons, 
            lin_blocks=lin_blocks, 
            out_neurons=num_classes
        )
    
    def conv_block(self, maxlen, num_filters, filter_size, emb_dim):
        """
        Shape
        -----
        [batch, maxlen, emb_dim, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1]
        [batch, n_filters, 1]
        [batch, n_filters, 1]
        """
        return nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(dim=1)), 
            ('conv', nn.Conv2d(1, num_filters, (filter_size, emb_dim))), 
            ('relu', nn.ReLU(inplace=True)), 
            ('squeeze', Squeeze(dim=3)), 
            ('pool', nn.MaxPool1d(maxlen-filter_size+1, stride=1)), 
            ('bn', nn.BatchNorm1d(num_filters))
        ]))
    
    def forward(self, x):
        """
        Input: [batch, maxlen]
        Output: [batch, n_classes]

        Shape
        -----
        [batch, maxlen, emb_dim]
        [batch, len(kernel_list)*n_filters, 1]
        [batch, len(kernel_list)*n_filters]
        [batch, len(kernel_list)*n_filters]
        [batch, 1, len(kernel_list)*n_filters]
        [batch, 1, n_classes]
        [batch, n_classes]
        """
        x = self.encoder(x)
        x = self.spatial(x)
        x = self.attention(x)
        x = torch.cat([layer(x) for layer in self.convs], dim=1)
        x = x.squeeze(2)
        x = self.drop(x)
        x = x.unsqueeze(1)
        x = self.classifier(x)
        return x.squeeze(1)

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model."""
        model_args = {
            "vocab_size": self.vocab_size, 
            "maxlen": self.maxlen,
            "emb_dim": self.emb_dim,
            "num_filters": self.num_filters,
            "kernel_list": self.kernel_list, 
            "dropout": self.dropout, 
            "lin_neurons": self.lin_neurons, 
            "lin_blocks": self.lin_blocks, 
            "num_classes": self.num_classes
        }
        return model_args


def main(conf):
    # Load CFPB dataset
    ds = cfpb_dataset.load(split='train')
    dataset_dict = ds.train_test_split(test_size=0.2, seed=914)
    train_ds = dataset_dict['train']
    valid_ds = dataset_dict['test']
    
    # Load tokeniser
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_ds = train_ds.map(
		lambda x: tokenizer(
			x['text'], 
            max_length=conf['model']['max_length'], 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=False
		)
	)
    valid_ds = valid_ds.map(
		lambda x: tokenizer(
			x['text'], 
            max_length=conf['model']['max_length'], 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=False
		)
	)
    train_ds.set_format(type='torch', columns=['input_ids', 'label'])
    valid_ds.set_format(type='torch', columns=['input_ids', 'label'])

    # Get dataloader
    train_dl = DataLoader(
        train_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )
    valid_dl = DataLoader(
        valid_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=False, 
        drop_last=True, 
        pin_memory=True
    )
    print(
        f'Train dataset: {len(train_ds)}\n'
        f'Valid dataset: {len(valid_ds)}'
    )
    
    # Define model and optimiser
    model = AttnTextCNN(
        tokenizer.vocab_size, 
        conf['model']['max_length'], 
        emb_dim=conf['model']['embedding_dim'], 
        num_filters=conf['model']['num_filters'], 
        kernel_list=conf['model']['kernel_list'], 
        dropout=conf['model']['dropout'], 
        lin_neurons=conf['model']['lin_neurons'], 
        lin_blocks=conf['model']['lin_blocks'], 
        num_classes=5
    )
    optimiser = make_optimiser(model.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimiser, factor=0.5, patience=5)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    targets = train_ds['label']
    if conf['training']['loss_weight_rescaling']:
        occurrences = torch.unique(targets, sorted=True, return_counts=True)[1]
        # weight = 1. - (occurrences / len(targets))
        weight = max(occurrences) / occurrences
    else:
        weight = None
    loss_func = nn.CrossEntropyLoss(weight=weight)
    system = TextCnnSystem(
        model=model,
        loss_func=loss_func,
        optimiser=optimiser,
        train_loader=train_dl,
        val_loader=valid_dl,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, monitor="loss/val_loss", mode="min", save_top_k=5, verbose=False
    )
    callbacks.append(checkpoint)

    # Resume training
    if conf['training']['resume']:
        if glob.glob(checkpoint_dir+'*.ckpt'):
            list_of_files = glob.glob(checkpoint_dir+'*.ckpt')
            latest_file = max(list_of_files, key=os.path.getctime)
            ckpt_path = latest_file
            print(f"Loading checkpoint: {ckpt_path} ...")
            model.load_from_checkpoint(checkpoint_path=ckpt_path)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="loss/val_loss", mode="min", patience=30, verbose=False))

    # Tensorboard Logger
    logger = pl.loggers.TensorBoardLogger(save_dir=exp_dir, name="lightning_logs", default_hp_metric=False)

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks, 
        logger=logger, 
        default_root_dir=exp_dir,
        gpus=gpus, 
        deterministic=True, 
        accelerator=distributed_backend, 
        plugins=DDPPlugin(find_unused_parameters=False), 
        limit_train_batches=1.0, 
        gradient_clip_val=5.0, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == '__main__':
    from textalgo.utils import load_yaml
    from textalgo.utils import parse_args_as_dict, prepare_parser_from_dict

    MODULE_DIR = os.path.join(os.path.dirname(sys.path[0]), '..', '..')
    RECIPE_DIR = os.path.join(MODULE_DIR, 'recipe/cfpb/AttnTextCNN/')
    YAML_DIR = os.path.join(RECIPE_DIR, 'local/conf.yml')

    def_conf = load_yaml(YAML_DIR)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)