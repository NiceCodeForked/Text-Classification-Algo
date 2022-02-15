import os
from pathlib import Path
from datasets import load_dataset


CFPB_URL = 'https://raw.githubusercontent.com/penguinwang96825/Text-Classification-Algo/master/data/'
MODULE_PATH = Path(os.path.abspath(__file__)).parent
TRAIN_CSV_PATH = CFPB_URL + 'cfpb-train.csv'
TEST_CSV_PATH = CFPB_URL + 'cfpb-test.csv'


def load(split='train'):
    cfpb_dict = load_dataset(
        'csv', 
        data_files={'train':TRAIN_CSV_PATH, 'test':TEST_CSV_PATH}
    )
    return cfpb_dict[split]