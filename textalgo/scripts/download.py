import os
import sys
import requests
import shutil
from pathlib import Path
from tqdm.auto import tqdm


def download_and_save(url, path: Path, overwrite=False, data_length=None):
    """
    Parameters
    ----------
    url: str
        URL path of the file to download.
    overwrite: bool
        Whether to overwrite existing files.
    data_length: int
        A number denoting an the exact byte length of the HTTP body.
    
    Examples
    --------
    >>> # Download glove pre-trained embedding vectors
    >>> import os, shutil
    >>> os.makedirs('data', exist_ok=True)
    >>> path = Path(__file__).parent / 'data' / "glove.6B.zip"
    >>> if download_and_save('https://nlp.stanford.edu/data/glove.6B.zip', path):
    >>>    shutil.unpack_archive(str(path), str(path.parent))
    
    Credits
    -------
    1. https://github.com/AymericShini/hacktoberfest_ML
    """
    stream = requests.get(url, stream=True)
    total_size = data_length or int(stream.headers.get('content-length', 0))

    if path.exists() and not overwrite:
        print(f'{path} already exists!')
        return False

    try:
        with path.open("wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for data in stream.iter_content(32*1024):
                    f.write(data)
                    pbar.update(len(data))
        print(f'Complete downloading!')
        return True
    except:
        path.unlink()


def main():
    if sys.argv[1] == 'cfpb':
        download_cfpb()


if __name__ == '__main__':
    main()