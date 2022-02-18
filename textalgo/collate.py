import torch
import torch.nn.functional as F


class DynamicPadding(object):

    def __init__(self, x_col_name, y_col_name):
        super().__init__()
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name

    def __call__(self, batch):
        labels = torch.tensor([b[self.y_col_name]for b in batch])
        input_ids = [torch.tensor(b[self.x_col_name]) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        return {self.x_col_name:input_ids, self.y_col_name:labels}


class StaticPadding(object):

    def __init__(self, x_col_name, y_col_name, max_length):
        super().__init__()
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.max_length = max_length

    def __call__(self, batch):
        labels = torch.tensor([b[self.y_col_name]for b in batch])
        input_ids = [torch.tensor(b[self.x_col_name]) for b in batch]
        input_ids = torch.stack([F.pad(i, (0, self.max_length-i.shape[0])) for i in input_ids])
        return {
            self.x_col_name: input_ids, 
            self.y_col_name: labels
        }