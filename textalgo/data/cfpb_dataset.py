import os
from datasets import load_dataset


MODULE_PATH = os.path.dirname(__file__)
TRAIN_CSV_PATH = os.path.join(MODULE_PATH, 'cfpb-train.csv')
TEST_CSV_PATH = os.path.join(MODULE_PATH, 'cfpb-test.csv')


def load(split='train'):
    cfpb_dict = load_dataset(
        'csv', 
        data_files={'train':TRAIN_CSV_PATH, 'test':TEST_CSV_PATH}
    )
    return cfpb_dict[split]