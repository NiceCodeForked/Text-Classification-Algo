import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..'))

import json
import torch
import datasets
import argparse
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics

from textalgo.preprocess.tokenize import word_tokenize
from textalgo.metrics import (
    accuracy_score, 
    f1_score, 
    top_k_accuracy_score, 
    precision_score, 
    recall_score
)
from textalgo.collate import StaticPadding
from src.model import AttnTextCNN


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
args = parser.parse_args()
arg_dic = dict(vars(args))
MODULE_DIR = os.path.join(os.path.dirname(sys.path[0]), '..', '..')
RECIPE_DIR = os.path.join(MODULE_DIR, 'recipe/cfpb/AttnTextCNN/')
YAML_DIR = os.path.join(RECIPE_DIR, args.exp_dir, 'conf.yml')


def main(conf):
    exp_dir = conf["exp_dir"]
    model_path = os.path.join(exp_dir, "best_model.pth")
    conf = conf['train_conf']

    # Load CFPB dataset
    url = "https://raw.githubusercontent.com/penguinwang96825/Text-Classification-Algo/master/data/"
    test_url = url + "cfpb-test.csv"
    df_test = pd.read_csv(url)
    test_ds = datasets.Dataset.from_dict({"text": df_test['text'], 'label': df_test['label']})
    test_ds = test_ds.map(
        lambda x: {'tokens': word_tokenize(x['text'])}, 
        batched=True, num_proc=conf['data']['num_processing'], tqdm_kwargs={'leave': False}
    )

    # Load vocab
    with open('./vocab.txt', mode='r', encoding='utf-8') as f:
        stoi = {line.rstrip('\n'):idx for idx, line in enumerate(f)}

    # Vectorise the tokens based on vocab
    test_ds = test_ds.map(
        lambda x: {'input_ids': [stoi.get(token, stoi['<unk>']) for token in x['tokens']]}
    )

    # Get dataloader
    test_dl = DataLoader(
        test_ds, 
        batch_size=conf['training']['batch_size'], 
        num_workers=conf['training']['num_workers'], 
        shuffle=False, 
        drop_last=False, 
        pin_memory=True, 
        collate_fn=StaticPadding('input_ids', 'label', conf['model']['max_length'])
    )

    # Define model and optimiser
    model = AttnTextCNN.from_pretrained(model_path)

    # Start evaluation loop
    with torch.no_grad():
        y_true, y_pred = [], []
        for batch_idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            est_targets = model(batch['input_ids'])
            y_true.append(batch['label'])
            y_pred.append(est_targets)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        acc = accuracy_score(y_true, y_pred).item()
        top3_acc = top_k_accuracy_score(y_true, y_pred, topk=(3,))
        f1 = f1_score(y_true, y_pred, num_classes=5).item()
        precision = precision_score(y_true, y_pred).item()
        recall = recall_score(y_true, y_pred).item()
        result = {
            'acc': acc, 
            'acc@3': top3_acc, 
            'f1': f1, 
            'precision': precision, 
            'recall': recall
        }
        os.system('clear')
        print('Performance: \n')
        print(json.dumps(result, indent=4))
        print('='*60)
        print('Classification report: \n')
        print(metrics.classification_report(
            y_true.numpy(), y_pred.argmax(dim=1).numpy()
        ))
        print('='*60)
        print('Confusion matrix: \n')
        print(metrics.confusion_matrix(
            y_true.numpy(), y_pred.argmax(dim=1).numpy()
        ))


if __name__ == '__main__':
    from textalgo.utils import load_yaml

    def_conf = load_yaml(YAML_DIR)
    arg_dic["train_conf"] = def_conf
    main(arg_dic)