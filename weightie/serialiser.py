"""
Utility for loading and saving weights to disk.

The on-disk format is designed to be directly memory mapped meaning that when
the model is not in use the weights can be swapped out of memory and that only
weights which are actively in use will stay in RAM.


Data format
===========

The file consists of two parts: the first part contains the raw array data and
the second part contains a pickled representation of the Python container
object with all Numpy array objects replaced with pointers to the raw data
blocks.

All raw array data is (by default) 64kb-aligned in the native byte endianness
of the system which saved the data. The 64kb alignment means these arrays
should start on page boundaries on all common systems. I *think* that Numpy
will do the correct thing if loaded on a machine with the 'wrong' endianness,
albeit at the cost of performance. (But seriously, who has a big-endian machine
to even test things on?)

The pickled Python object is followed by a 32-bit unsigned, little endian
integer indicating the number of bytes in the pickled data. All arrays in the
pickled structure are replaced with _NDArrayOffset objects which include
everything needed to wrap this in an array.
"""


from typing import NamedTuple, BinaryIO, IO, Any, cast

import pickle
import struct
import platform
import mmap
import io
import os

import numpy as np
from numpy.typing import NDArray


__all__ = ["dump", "load"]


class _NDArrayOffset(NamedTuple):
    """
    A stand-in for a numpy array stored at a particular location in the file.
    """

    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype


class Pickler(pickle.Pickler):
    def __init__(self, file: BinaryIO, alignment: int) -> None:
        # We'll write raw Numpy data to this file during pickling and then
        # later append the other Pickled data...
        self._real_file = file
        self._alignment = alignment

        # ... which we'll have Pickle write into this BytesIO.
        self._pickled_file = io.BytesIO()
        super().__init__(self._pickled_file)

    def dump(self, obj: Any) -> None:
        super().dump(obj)
        self._pickled_file

        # Append the Pickled data (and length)
        pickled_data = self._pickled_file.getbuffer()
        self._real_file.write(pickled_data)
        self._real_file.write(struct.pack("<I", len(pickled_data)))

    def persistent_id(self, obj: Any) -> Any:
        # Swap all Numpy arrays for _NDArrayOffsets, writing the actual data
        # raw into the target file
        if isinstance(obj, np.ndarray):
            # Pad to multiple of required alignment as necessary
            cur_offset = self._real_file.tell()
            offset = (
                (cur_offset + self._alignment - 1) // self._alignment
            ) * self._alignment
            self._real_file.write(b"\0" * (offset - cur_offset))

            # Write the array data
            self._real_file.write(obj.tobytes())

            return _NDArrayOffset(
                offset=offset,
                shape=obj.shape,
                dtype=obj.dtype,
            )
        else:
            return None


class Unpickler(pickle.Unpickler):
    def __init__(self, file: BinaryIO) -> None:
        # Obtain a memory map of the file (based on which we'll create Numpy
        # arrays)
        extra_kwargs = {}
        if platform.system() != "Windows":
            extra_kwargs["prot"] = mmap.PROT_READ
        self._memory_map = mmap.mmap(file.fileno(), length=0, **extra_kwargs)

        # Seek to the start of the Pickled data at the end of the file ready
        # for unpickling...
        pickled_data_len = struct.unpack("<I", self._memory_map[-4:])[0]
        file.seek(-4 - pickled_data_len, os.SEEK_END)
        super().__init__(file)

    def persistent_load(self, pid: Any) -> Any:
        assert isinstance(pid, _NDArrayOffset)
        ar = np.frombuffer(
            buffer=self._memory_map,
            offset=pid.offset,
            count=np.prod(pid.shape),
            dtype=pid.dtype,
        )
        ar.flags.writeable = False
        ar = ar.reshape(pid.shape)
        return ar


def dump(data: Any, file: BinaryIO, alignment: int = 64 * 1024) -> None:
    """
    Dump the given weights-containing object into the provided file with Numpy
    arrays being stored with the given alignment.

    Parameters
    ==========
    data : nested list, dict and tuple structure
        The weights to be serialised, in a structure made up of lists, dicts
        and tuples (including named tuples).

        Any Numpy arrays will be seriallised in a manner supporting memory
        mapping when loaded. Other data types included will just be pickled
        as-is.
    file : binary file opened for writing
        The file to write the serialised data to.
    alignment : int
        The byte alignment for all array data chunks. The default of 64kb
        is a multiple of the page size of many common systems and makes a
        reasonable default choice.
    """
    Pickler(file, alignment).dump(data)


def load(file: BinaryIO) -> Any:
    """
    Load a set of weights from the provided file with all arrays being memory
    mapped from the file.
    """
    return Unpickler(file).load()
