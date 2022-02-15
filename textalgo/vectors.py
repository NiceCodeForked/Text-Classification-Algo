from pathlib import Path
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from huggingface_hub import snapshot_download
from requests import HTTPError


_SUFFIX: str = ".model"


class Vectors(Word2VecKeyedVectors):
    
    @classmethod
    def from_pretrained(cls, model: str, mmap: str=None):
        """
        Parameters
        ----------
        model: str
            Model name to load from the hub. For example: "glove-wiki-gigaword-50"
            Check it on https://huggingface.co/models?sort=downloads&search=fse%2F
        mmap: str 
            If to load the vectors in mmap mode.

        Returns
        -------
        Vectors
            An object of pretrained vectors.
        """
        try:
            path = Path(snapshot_download(repo_id=f"fse/{model}"))
        except HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"model {model} does not exist")
            raise

        assert path.exists(), "Something went wrong, the file wasn't downloaded."

        return super(Vectors, cls).load(
            (path / (model + _SUFFIX)).as_posix(), mmap=mmap
        )


class FTVectors(FastTextKeyedVectors):
    
    @classmethod
    def from_pretrained(cls, model: str, mmap: str=None):
        """
        Parameters
        ----------
        model: str
            Model name to load from the hub. For example: "glove-wiki-gigaword-50"
            Check it on https://huggingface.co/models?sort=downloads&search=fse%2F
        mmap: str 
            If to load the vectors in mmap mode.

        Returns
        -------
        Vectors
            An object of pretrained vectors.
        """
        try:
            path = Path(snapshot_download(repo_id=f"fse/{model}"))
        except HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"model {model} does not exist")
            raise

        assert path.exists(), "Something went wrong, the file wasn't downloaded."

        return super(FTVectors, cls).load(
            (path / (model + _SUFFIX)).as_posix(), mmap=mmap
        )