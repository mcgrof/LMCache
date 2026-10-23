# SPDX-License-Identifier: Apache-2.0
"""Bounded buffered file reads into caller-owned storage memory."""

# Standard
from pathlib import Path
import os


def read_file_range_into(path: str | Path, buffer: memoryview, offset: int = 0) -> int:
    """Fill a writable buffer from an exact file range using the page cache.

    Args:
        path: Regular file containing the payload.
        buffer: Contiguous writable destination owned by the caller.
        offset: Nonnegative byte offset of the payload.

    Returns:
        Number of bytes read, equal to the buffer length.

    Raises:
        ValueError: The offset or destination is invalid.
        EOFError: The file does not contain the complete range.
        OSError: Opening or reading the file fails.
    """
    if offset < 0 or buffer.readonly or not buffer.c_contiguous:
        raise ValueError("invalid range or destination")
    view = buffer.cast("B")
    with open(path, "rb", buffering=0) as stream:
        if offset + len(view) > os.fstat(stream.fileno()).st_size:
            raise EOFError("artifact range exceeds file")
        stream.seek(offset)
        done = 0
        while done < len(view):
            count = stream.readinto(view[done:])
            if not count:
                raise EOFError("short artifact read")
            done += count
    return done
