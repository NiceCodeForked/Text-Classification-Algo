import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..'))

import yaml
import json
import glob
import argparse
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import set_progress_bar_enabled
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from textalgo.dataset import load_text_classification_datasets
from textalgo.collate import StaticPadding
from textalgo.engine import make_optimiser

from src.model import AttnTextCNN
from src.system import TextCnnSystem


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
set_progress_bar_enabled(True)
pl.seed_everything(seed=914)


def main(conf):
    # Load CFPB dataset
    print('Loading dataset from the internet...')
    url = "https://raw.githubusercontent.com/penguinwang96825/Text-Classification-Algo/master/data/cfpb-train.csv"
    df = pd.read_csv(url)
    X_train, X_valid, y_train, y_valid = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=914
    )
    train_ds, valid_ds, vocab = load_text_classification_datasets(
        X_train, X_valid, y_train, y_valid, batched=True, num_proc=16
    )

    # Save vocab
    sorted_by_freq_tuples = sorted(vocab.get_stoi().items(), key=lambda x: x[1], reverse=False)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    with open('./vocab.txt', mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(ordered_dict.keys()))

    # Get dataloader
    print('Loading dataloader...')
    train_dl = DataLoader(
        train_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True, 
        collate_fn=StaticPadding('input_ids', 'label', conf['model']['max_length'])
    )
    valid_dl = DataLoader(
        valid_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=False, 
        drop_last=True, 
        pin_memory=True, 
        collate_fn=StaticPadding('input_ids', 'label', conf['model']['max_length'])
    )
    print(
        f'Train dataset: {len(train_ds)}\n'
        f'Valid dataset: {len(valid_ds)}'
    )
    
    # Define model and optimiser
    model = AttnTextCNN(
        len(vocab), 
        conf['model']['max_length'], 
        embed_dim=conf['model']['embedding_dim'], 
        num_heads=conf['model']['num_heads'], 
        num_filters=conf['model']['num_filters'], 
        kernel_list=conf['model']['kernel_list'], 
        dropout=conf['model']['dropout'], 
        lin_neurons=conf['model']['lin_neurons'], 
        lin_blocks=conf['model']['lin_blocks'], 
        num_classes=conf['model']['num_classes']
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
    targets = torch.from_numpy(y_train.values)
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