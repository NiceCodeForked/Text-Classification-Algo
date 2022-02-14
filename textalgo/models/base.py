import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, device="cpu"):
        conf = torch.load(pretrained_model_conf_or_path, map_location=torch.device(device))
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(**conf["model_args"])
        else:
            model = model_class(**conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

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


def get(identifier):
    """Returns an model class from a string (case-insensitive).
    Args:
        identifier (str): the model name.
    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")