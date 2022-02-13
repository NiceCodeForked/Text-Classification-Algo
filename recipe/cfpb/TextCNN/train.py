import sys
sys.path.append('/Users/wangyang/Desktop/Text-Classification-Algo')
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import set_progress_bar_enabled
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from textalgo.data import cfpb_dataset
from textalgo.utils import load_yaml
from textalgo.engine import System
from textalgo.engine import make_optimiser
from textalgo.models import TextCNN, LightWeightedTextCNN


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
set_progress_bar_enabled(True)


class TextCnnSystem(System):

    def common_step(self, batch, batch_nb, train):
        inputs, targets = batch['input_ids'], batch['label']
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        return loss


def main(conf):
    # Load CFPB dataset
    ds = cfpb_dataset.load(split='train')
    dataset_dict = ds.train_test_split(test_size=0.2, seed=914)
    train_ds = dataset_dict['train']
    valid_ds = dataset_dict['test']
    
    # Load tokeniser
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    train_ds = train_ds.map(
		lambda x: tokenizer(
			x['text'], 
            max_length=conf['data']['max_length'], 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=False
		)
	)
    valid_ds = valid_ds.map(
		lambda x: tokenizer(
			x['text'], 
            max_length=conf['data']['max_length'], 
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
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )
    valid_dl = DataLoader(
        valid_ds, 
        batch_size=conf['training']['batch_size'], 
        shuffle=False, 
        drop_last=True, 
        pin_memory=True
    )
    
    # Define model and optimiser
    if conf['model']['light']:
        CNN = LightWeightedTextCNN
    else:
        CNN = TextCNN
    model = CNN(
        tokenizer.vocab_size, 
        conf['data']['max_length'], 
        emb_dim=conf['model']['embedding_dim'], 
        num_filters=conf['model']['num_filters'], 
        kernel_list=[3, 4, 5], 
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
    loss_func = nn.CrossEntropyLoss()
    system = TextCnnSystem(
        model=model,
        loss_func=loss_func,
        optimiser=optimiser,
        train_loader=train_dl,
        val_loader=valid_dl,
        scheduler=scheduler,
        config=conf,
    )

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        # callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus, 
        deterministic=True, 
        distributed_backend=distributed_backend,
        limit_train_batches=1.0, 
        gradient_clip_val=5.0, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
    )
    trainer.fit(system)


if __name__ == '__main__':
    from textalgo.utils import parse_args_as_dict, prepare_parser_from_dict

    def_conf = load_yaml("local/conf.yml")
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)