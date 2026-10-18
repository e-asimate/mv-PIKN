"""Compatibility launcher for the modular CHB-MIT EEG preprocessing code."""

from preproc_modules import *  # noqa: F401,F403
from preproc_modules.hdf5_io import _dtype_nbytes
from preproc_modules.processor import _expected_shape

if __name__ == "__main__":
    main()
