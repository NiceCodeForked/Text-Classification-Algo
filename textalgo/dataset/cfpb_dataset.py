import os
from pathlib import Path
from datasets import load_dataset


URL_PATH = 'https://media.githubusercontent.com/media/penguinwang96825/Text-Classification-Algo/master/textalgo/dataset/csv/'
MODULE_PATH = Path(os.path.abspath(__file__)).parent
TRAIN_CSV_PATH = URL_PATH + 'cfpb-train.csv'
TEST_CSV_PATH = URL_PATH + 'cfpb-test.csv'


def load(split='train'):
    cfpb_dict = load_dataset(
        'csv', 
        data_files={'train':TRAIN_CSV_PATH, 'test':TEST_CSV_PATH}
    )
    return cfpb_dict[split]