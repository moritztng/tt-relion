"""RLX dev cell: the first N particles of EMPIAR-10028 shiny_2sets.star, and the stacks they need.

    python3 subset.py shiny_2sets.star dev.star [--n 13156] > stacks.txt

The star carries no _rlnRandomSubset column (RELION assigns the half sets itself under
--split_random_halves), so the cell is simply the first N data rows in file order: deterministic,
and it needs only the .mrcs stacks those rows name, one path per line on stdout.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--n", type=int, default=13156)
a = ap.parse_args()

head, rows, col = [], [], None
with open(a.src) as f:
    for line in f:
        s = line.split()
        if len(rows) == 0 and (not s or s[0].startswith(("data_", "loop_", "_"))):
            if s and s[0] == "_rlnImageName":
                col = int(s[1].lstrip("#")) - 1
            head.append(line)
        elif s:
            rows.append(line)
            if len(rows) == a.n:
                break
assert col is not None and len(rows) == a.n, (col, len(rows))
with open(a.dst, "w") as f:
    f.writelines(head + rows)
stacks = []
for r in rows:
    p = r.split()[col].split("@", 1)[1]
    if not stacks or stacks[-1] != p:
        stacks.append(p)
print("\n".join(dict.fromkeys(stacks)))
