"""RLX input star: EMPIAR-10028 shiny_2sets.star made readable by RELION 5, optionally cut to the dev cell.

    python3 subset.py shiny_2sets.star dev.star [--n 13156] > stacks.txt   # dev cell
    python3 subset.py shiny_2sets.star full.star --n 0 > stacks.txt         # timed runs, all 105,247 rows

The star carries no _rlnRandomSubset column (RELION assigns the half sets itself under
--split_random_halves), so the cell is simply the first N data rows in file order: deterministic,
and it needs only the .mrcs stacks those rows name, one path per line on stdout.

One row of the EMPIAR file (052_particles #106) carries a 17th token after its 16 labelled columns.
RELION 5 refuses the whole file for it ("more columns than the number of labels"), so every row is cut
to the label count. The extra token has no label, so no particle data changes (PROTOCOL D8).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--n", type=int, default=13156, help="rows to keep; 0 = all")
a = ap.parse_args()

head, rows, col, ncol, cut = [], [], None, 0, 0
with open(a.src) as f:
    for line in f:
        s = line.split()
        if not rows and (not s or s[0].startswith(("data_", "loop_", "_"))):
            if s and s[0].startswith("_"):
                ncol += 1
                if s[0] == "_rlnImageName":
                    col = int(s[1].lstrip("#")) - 1
            head.append(line)
        elif s:
            if len(s) > ncol:
                cut += 1
                line = " ".join(s[:ncol]) + "\n"
            rows.append(line)
            if len(rows) == a.n:
                break
assert col is not None and (a.n == 0 or len(rows) == a.n), (col, len(rows))
with open(a.dst, "w") as f:
    f.writelines(head + rows)
print(f"{len(rows)} rows, {ncol} columns, {cut} rows cut to the label count", file=sys.stderr)
print("\n".join(dict.fromkeys(r.split()[col].split("@", 1)[1] for r in rows)))
