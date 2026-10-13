"""
download_instances.py — fetch the CVRPLIB benchmark instances used in this project.

Source: CVRPLIB (https://galgos.inf.puc-rio.br/cvrplib), Augerat et al. (1995)
sets A and P, and Christofides & Eilon (1969) set E. Every instance has a
published optimal cost, which we use as the reference for gap calculations.

CVRPLIB serves each instance by a numeric id rather than by file name, so the
table below maps name -> id. The old vrp.atd-lab.inf.puc-rio.br URLs no longer
work (they redirect and then 404).

Usage:
    python src/download_instances.py

If the site is unreachable, download the files by hand from the URLs printed
below and drop them into data/instances/<name>.vrp — the loader only needs
the .vrp text.
"""

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/instance"

# (name, CVRPLIB id, published optimum) — small ones for the exact method,
# larger ones to show where it stops scaling.
INSTANCES = [
    ("P-n16-k8", 75, 450),
    ("E-n22-k4", 55, 375),
    ("A-n32-k5", 4, 784),
    ("A-n45-k6", 16, 944),
    ("A-n80-k10", 31, 1763),
]


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (cvrp-assignment)"})
    with urlopen(req, timeout=60) as r:
        return r.read()


def to_vrp_text(raw: bytes, name: str) -> str:
    """The server may answer with the .vrp text or with a zip holding it."""
    if raw[:2] == b"PK":  # zip signature
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            member = next(m for m in z.namelist() if m.endswith(".vrp"))
            raw = z.read(member)
    text = raw.decode("utf-8", errors="replace")
    if "NODE_COORD_SECTION" not in text:
        raise ValueError(f"{name}: response is not a CVRPLIB .vrp file "
                         f"(starts with: {text[:80]!r})")
    return text


if __name__ == "__main__":
    out = Path("data/instances")
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, iid, opt in INSTANCES:
        dest = out / f"{name}.vrp"
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        url = f"{BASE}/{iid}"
        print(f"downloading {name} <- {url}")
        try:
            dest.write_text(to_vrp_text(fetch(url), name))
        except Exception as e:  # keep going, report at the end
            print(f"  FAILED: {e}")
            failed.append((name, url))
    print("done — optima:", {n: o for n, _, o in INSTANCES})
    if failed:
        print("\nCould not download these — open the URL in a browser, save as "
              "data/instances/<name>.vrp:")
        for name, url in failed:
            print(f"  {name}: {url}")
        sys.exit(1)
