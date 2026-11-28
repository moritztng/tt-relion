"""Where the chip kernels' sources are read from.

The kernel build cache of tt-metal keys a build on the source file's path, so the same kernels placed in a second
directory (a new checkout, a second copy of the tree) are all built again: 60 to 130 chip seconds of slow first
batches in one RELION iteration on two chips, 5 in a second run from the same place. With TT_RELION_KERNEL_CACHE set
the sources are copied once to a directory named by their content and read from there, so every tree with the same
kernels shares one set of builds. `1` = ~/.cache/tt-relion/kernels, anything else = that directory, unset or `0` =
read them in place. If the copy cannot be made the sources are read in place."""
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "kernels"


def kernel_dir(name):
    src = _SRC / name
    root = os.environ.get("TT_RELION_KERNEL_CACHE", "")
    if root in ("", "0"):
        return src
    root = Path("~/.cache/tt-relion/kernels" if root == "1" else root).expanduser()
    files = sorted(p for p in src.iterdir() if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    dst = root / h.hexdigest()[:20] / name
    if dst.is_dir():
        return dst
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=dst.parent, prefix=".new-"))
        for p in files:
            shutil.copyfile(p, tmp / p.name)
        os.chmod(tmp, 0o755)                # mkdtemp makes it private; a shared cache is read by others
        try:
            os.rename(tmp, dst)             # whole or not at all: another process may place it first
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    except OSError:
        pass
    return dst if dst.is_dir() else src
