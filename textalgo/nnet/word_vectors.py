import io
import os
import torch
import importlib
import subprocess
import zipfile
import logging
import urllib.request
from types import ModuleType
from tqdm.auto import tqdm
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class LazyLoader(ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies. `contrib`, and
    `ffmpeg` are examples of modules that are large and not always needed, and this allows them to
    only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """ Load the module and insert it into the parent's globals. """

        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


six = LazyLoader('six', globals(), 'six')
requests = LazyLoader('requests', globals(), 'requests')


class _PretrainedWordVectors(object):
    """ _PretrainedWordVectors handles downloading, caching and storing pretrained embeddings.
    Args:
        name (str): name of the file that contains the vectors
        cache (str, optional): directory for cached vectors
        url (str or None, optional): url for download if vectors not found in cache
        unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and
            returns a Tensor of the same size
        is_include (callable, optional): callable returns True if to include a token in memory
            vectors cache; some of these embedding files are gigantic so filtering it can cut
            down on the memory usage. We do not cache on disk if ``is_include`` is defined.
    """

    def __init__(self,
                 name,
                 cache='.word_vectors_cache',
                 url=None,
                 unk_init=torch.Tensor.zero_,
                 is_include=None):
        self.unk_init = unk_init
        self.is_include = is_include
        self.name = name
        self.cache(name, cache, url=url)

    def __contains__(self, token):
        return token in self.token_to_index

    def _get_token_vector(self, token):
        """Return embedding for token or for UNK if token not in vocabulary"""
        if token in self.token_to_index:
            return self.vectors[self.token_to_index[token]]
        else:
            return self.unk_init(torch.Tensor(self.dim))

    def __getitem__(self, tokens):
        if isinstance(tokens, list) or isinstance(tokens, tuple):
            vector_list = [self._get_token_vector(token) for token in tokens]
            return torch.stack(vector_list)
        elif isinstance(tokens, str):
            token = tokens
            return self._get_token_vector(token)
        else:
            raise TypeError("'__getitem__' method can only be used with types"
                            "'str', 'list', or 'tuple' as parameter")

    def __len__(self):
        return len(self.vectors)

    def __str__(self):
        return self.name

    def cache(self, name, cache, url=None):
        if os.path.isfile(name):
            path = name
            path_pt = os.path.join(cache, os.path.basename(name)) + '.pt'
        else:
            path = os.path.join(cache, name)
            path_pt = path + '.pt'

        if not os.path.isfile(path_pt) or self.is_include is not None:
            if url:
                download_file_maybe_extract(url=url, directory=cache, check_files=[name])

            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            index_to_token, vectors, dim = [], None, None

            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                logger.warning("Could not read {} as UTF8 file, "
                               "reading file as bytes and skipping "
                               "words with malformed UTF8.".format(path))
                with open(path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            logger.info("Loading vectors from {}".format(path))
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" " if binary_lines else " ")

                word, entries = entries[0], entries[1:]
                if dim is None and vectors is None and len(entries) > 1:
                    dim = len(entries)
                    vectors = torch.empty(len(lines), dim, dtype=torch.float)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError("Vector for token {} has {} dimensions, but previously "
                                       "read vectors have {} dimensions. All vectors must have "
                                       "the same number of dimensions.".format(
                                           word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                if self.is_include is not None and not self.is_include(word):
                    continue

                vectors[len(index_to_token)] = torch.tensor([float(x) for x in entries])
                index_to_token.append(word)

            self.index_to_token = index_to_token
            self.token_to_index = {word: i for i, word in enumerate(index_to_token)}
            self.vectors = vectors[:len(index_to_token)]
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.index_to_token, self.token_to_index, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.index_to_token, self.token_to_index, self.vectors, self.dim = torch.load(path_pt)


def download_file_maybe_extract(url, directory, filename=None, extension=None, check_files=[]):
    """ Download the file at ``url`` to ``directory``. Extract to ``directory`` if tar or zip.
    Args:
        url (str or Path): Url of file.
        directory (str): Directory to download to.
        filename (str, optional): Name of the file to download; Otherwise, a filename is extracted
            from the url.
        extension (str, optional): Extension of the file; Otherwise, attempts to extract extension
            from the filename.
        check_files (list of str or Path): Check if these files exist, ensuring the download
            succeeded. If these files exist before the download, the download is skipped.
    Returns:
        (str): Filename of download file.
    Raises:
        ValueError: Error if one of the ``check_files`` are not found following the download.
    """
    if filename is None:
        filename = _get_filename_from_url(url)

    directory = str(directory)
    filepath = os.path.join(directory, filename)
    check_files = [os.path.join(directory, str(f)) for f in check_files]

    if len(check_files) > 0 and _check_download(*check_files):
        return filepath

    if not os.path.isdir(directory):
        os.makedirs(directory)

    logger.info('Downloading {}'.format(filename))

    # Download
    if 'drive.google.com' in url:
        _download_file_from_drive(filepath, url)
    else:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filepath, reporthook=_reporthook(t))

    _maybe_extract(compressed_filename=filepath, directory=directory, extension=extension)

    if not _check_download(*check_files):
        raise ValueError('[DOWNLOAD FAILED] `*check_files` not found')

    return filepath


def _reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.
    Uses ``tqdm`` for progress bar.
    **Reference:**
    https://github.com/tqdm/tqdm
    Args:
        t (tqdm.tqdm) Progress bar.
    Example:
        >>> with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # doctest: +SKIP
        ...   urllib.request.urlretrieve(file_url, filename=full_path, reporthook=reporthook(t))
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def _download_file_from_drive(filename, url):  # pragma: no cover
    """ Download filename from google drive unless it's already in directory.
    Args:
        filename (str): Name of the file to download to (do nothing if it already exists).
        url (str): URL to download from.
    """
    confirm_token = None

    # Since the file is big, drive will scan it for virus and take it to a
    # warning page. We find the confirm token on this page and append it to the
    # URL to start the download process.
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token

    logger.info("Downloading %s to %s" % (url, filename))

    response = session.get(url, stream=True)
    # Now begin the download.
    chunk_size = 16 * 1024
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

    # Print newline to clear the carriage return from the download progress
    statinfo = os.stat(filename)
    logger.info("Successfully downloaded %s, %s bytes." % (filename, statinfo.st_size))


def _maybe_extract(compressed_filename, directory, extension=None):
    """ Extract a compressed file to ``directory``.
    Args:
        compressed_filename (str): Compressed file.
        directory (str): Extract to directory.
        extension (str, optional): Extension of the file; Otherwise, attempts to extract extension
            from the filename.
    """
    logger.info('Extracting {}'.format(compressed_filename))

    if extension is None:
        basename = os.path.basename(compressed_filename)
        extension = basename.split('.', 1)[1]

    if 'zip' in extension:
        with zipfile.ZipFile(compressed_filename, "r") as zip_:
            zip_.extractall(directory)
    elif 'tar.gz' in extension or 'tgz' in extension:
        # `tar` is much faster than python's `tarfile` implementation
        subprocess.call(['tar', '-C', directory, '-zxvf', compressed_filename])
    elif 'tar' in extension:
        subprocess.call(['tar', '-C', directory, '-xvf', compressed_filename])

    logger.info('Extracted {}'.format(compressed_filename))


def _get_filename_from_url(url):
    """ Return a filename from a URL
    Args:
        url (str): URL to extract filename from
    Returns:
        (str): Filename in URL
    """
    parse = urlparse(url)
    return os.path.basename(parse.path)


def _check_download(*filepaths):
    """ Check if the downloaded files are found.
    Args:
        filepaths (list of str): Check if these filepaths exist
    Returns:
        (bool): Returns True if all filepaths exist
    """
    return all([os.path.isfile(filepath) for filepath in filepaths])


def download_files_maybe_extract(urls, directory, check_files=[]):
    """ Download the files at ``urls`` to ``directory``. Extract to ``directory`` if tar or zip.
    Args:
        urls (str): Url of files.
        directory (str): Directory to download to.
        check_files (list of str): Check if these files exist, ensuring the download succeeded.
            If these files exist before the download, the download is skipped.
    Raises:
        ValueError: Error if one of the ``check_files`` are not found following the download.
    """
    check_files = [os.path.join(directory, f) for f in check_files]
    if _check_download(*check_files):
        return

    for url in urls:
        download_file_maybe_extract(url=url, directory=directory)

    if not _check_download(*check_files):
        raise ValueError('[DOWNLOAD FAILED] `*check_files` not found')