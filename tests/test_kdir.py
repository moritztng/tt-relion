"""The kernels' sources are read from a directory named by their content when TT_RELION_KERNEL_CACHE is set, so two
trees with the same kernels share one set of builds; in place otherwise."""
import shutil
from concurrent.futures import ThreadPoolExecutor

from tt_relion import kdir


def test_in_place_by_default(monkeypatch):
    monkeypatch.delenv("TT_RELION_KERNEL_CACHE", raising=False)
    assert kdir.kernel_dir("oz_fused") == kdir._SRC / "oz_fused"
    monkeypatch.setenv("TT_RELION_KERNEL_CACHE", "0")
    assert kdir.kernel_dir("oz_fused") == kdir._SRC / "oz_fused"


def test_two_trees_with_the_same_kernels_read_one_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_RELION_KERNEL_CACHE", str(tmp_path / "cache"))
    a = kdir.kernel_dir("oz_fused")
    assert a != kdir._SRC / "oz_fused" and a.name == "oz_fused"
    for p in (kdir._SRC / "oz_fused").iterdir():
        assert (a / p.name).read_bytes() == p.read_bytes()
    assert a.stat().st_mode & 0o777 == 0o755
    tree = tmp_path / "tree2" / "kernels"                       # the same sources placed somewhere else
    shutil.copytree(kdir._SRC, tree)
    monkeypatch.setattr(kdir, "_SRC", tree)
    assert kdir.kernel_dir("oz_fused") == a
    (tree / "oz_fused" / "compute.cpp").write_text((tree / "oz_fused" / "compute.cpp").read_text() + "\n// changed\n")
    b = kdir.kernel_dir("oz_fused")                              # changed sources get their own directory
    assert b != a and b.name == "oz_fused" and (b / "compute.cpp").read_text().endswith("// changed\n")


def test_many_processes_placing_it_at_once_leave_one_whole_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_RELION_KERNEL_CACHE", str(tmp_path / "cache"))
    with ThreadPoolExecutor(16) as ex:
        got = set(ex.map(lambda _: kdir.kernel_dir("oz_combine"), range(64)))
    assert len(got) == 1
    d = got.pop()
    assert sorted(p.name for p in d.iterdir()) == sorted(p.name for p in (kdir._SRC / "oz_combine").iterdir())
    assert [p.name for p in d.parent.iterdir()] == ["oz_combine"]   # no half-made copy left behind


def test_a_cache_that_cannot_be_written_reads_in_place(monkeypatch, tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    monkeypatch.setenv("TT_RELION_KERNEL_CACHE", str(f / "under-a-file"))
    assert kdir.kernel_dir("cand") == kdir._SRC / "cand"
