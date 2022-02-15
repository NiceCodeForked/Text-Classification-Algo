import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..'))

import torch
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from textalgo.dataset import cfpb_dataset
from textalgo.models import TextCNN, LightWeightedTextCNN
from textalgo.metrics import accuracy_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
args = parser.parse_args()
arg_dic = dict(vars(args))
MODULE_DIR = os.path.join(os.path.dirname(sys.path[0]), '..', '..')
RECIPE_DIR = os.path.join(MODULE_DIR, 'recipe/cfpb/TextCNN/')
YAML_DIR = os.path.join(RECIPE_DIR, args.exp_dir, 'conf.yml')


def main(conf):
    exp_dir = conf["exp_dir"]
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    conf = conf['train_conf']

    # Load CFPB dataset
    test_ds = cfpb_dataset.load(split='test')

    # Load tokeniser
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_ds = test_ds.map(
		lambda x: tokenizer(
			x['text'], 
            max_length=conf['model']['max_length'], 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=False
		)
	)
    test_ds.set_format(type='torch', columns=['input_ids', 'label'])

    # Get dataloader
    test_dl = DataLoader(
        test_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=False, 
        drop_last=False, 
        pin_memory=True
    )

    # Define model and optimiser
    if conf['model']['light']:
        CNN = LightWeightedTextCNN
    else:
        CNN = TextCNN
    model = CNN.from_pretrained(model_path)

    with torch.no_grad():
        y_true, y_pred = [], []
        for batch_idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            est_targets = model(batch['input_ids'])
            y_true.append(batch['label'])
            y_pred.append(est_targets)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, num_classes=5)
        print(
            f'acc: {acc:.4f}\n'
            f'f1: {f1:.4f}'
        )


if __name__ == '__main__':
    from textalgo.utils import load_yaml

    def_conf = load_yaml(YAML_DIR)
    arg_dic["train_conf"] = def_conf
    main(arg_dic)