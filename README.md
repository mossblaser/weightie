Weightie
========

Weightie is a little utility library for downloading, serialising (and
deserialising) weights for my little deep learning projects.

You probably shouldn't use this if you're not me!


Serialisation
-------------

Quick-start usage:

    >>> # Say your model weights are stored in some custom structure like so:

    >>> from typing import NamedTuple
    >>> from numpy.typing import NDArray

    >>> class MyWeights(NamedTuple):
    ...     foo: NDArray
    ...     bar: list[NDArray]
    ...     baz: bool  # Data need not be entirely NDArrays


    >>> # They can be dumped and loaded from disk like so:

    >>> from weightie import dump, load

    >>> weights = MyWeights(foo=np.zeros(1000), bar=np.zeros(10000), baz=True)
    >>> with open("example.weights", "wb") as f:
    ...     dump(weights, f)

    >>> with open("example.weights", "rb") as f:
    ...     loaded_weights = load(weights)

Weightie's serialisation format stores Python objects containing Numpy arrays
such that, when loaded from disk, the Numpy arrays will be memory mapped
directly from the files on disk. This means that data isn't actually read from
disk until it is used and is easily swapped out of memory by the operating
system when not in use.

See [`weightie/serialiser.py`](./weightie/serialiser.py) for details of the
on-disk format (it's pretty simple!).


Weight Downloading
------------------

Weightie provides a `download` function which can be used to download weights
filed published as assets on GitHub. (GitHub has [very
generous](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
usage allowances for release assets!)

Quick-start usage:

    >>> from weightie import download

    >>> download(
    ...     repository="mossblaser/example",
    ...     asset_filenames=["foo.weights", "bar.weights"],
    ...     target_version="v1.2.3",
    ... )
    {
        "foo": Path("/path/to/downloaded/v1.1.0-foo.weights"),
        "bar": Path("/path/to/downloaded/v1.1.0-bar.weights"),
    }

The `download` function handles various fiddly aspects of downloading weights
including:

* Downloading to platform-native data directories by default (using
  [platformdirs](https://github.com/platformdirs/platformdirs))
* Re-using already downloaded weights when appropriate.
* Ensuring downloaded weight file versions are compatible with the running
  software version.
* Keeping multiple weight files versions in sync.

See [`weightie/downloader.py`](./weightie/downloader.py) for the full list of
options.
