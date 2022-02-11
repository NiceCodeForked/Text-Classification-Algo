import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def serialize(self):
        """
        Serialize model and output dictionary.

        Returns
        -------
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError