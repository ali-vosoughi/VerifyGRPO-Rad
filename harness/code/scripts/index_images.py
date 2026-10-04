#!/usr/bin/env python3
"""Index the CheXpert Plus PNG archives so a study path resolves to pixels.

The Redivis export arrived with its file names replaced by positional indices
(files/PNG_compressed/0000000_ and so on). Each index file is identified here by its exact byte
size against the PNG_compressed table, which is unambiguous for these five chunks, and then each
archive's central directory is read once to map every member to its study path.

Outputs (results/e6_pairs/):
  chunk_map.json        index file -> original chunk name (by size)
  image_index.tsv       study path <TAB> chunk file <TAB> member name
  index_notes_<stamp>.md  provenance and counts

Stdlib only, so it runs under the system Python without waiting for the venv build.
"""
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime

ROOT = os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)")
CXP = os.path.join(ROOT, "datasets", "chexpert_plus")
ZIPDIR = os.path.join(CXP, "files", "PNG_compressed")
MANIFEST_CSV = os.path.join(CXP, "tables", "PNG_compressed.sample200.csv")
OUTDIR = os.path.join(ROOT, "results", "e6_pairs")

# Test extractions: three members are pulled to prove the archives are readable end to end.
CACHE = os.path.join(ROOT, "runs", "image_cache")


def load_manifest():
    """size -> original chunk name, read from the export manifest."""
    sizes = {}
    with open(MANIFEST_CSV, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        i_name, i_size = header.index("file_name"), header.index("size")
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(i_name, i_size):
                continue
            sizes[int(parts[i_size])] = parts[i_name]
    return sizes


def normalise(member):
    """Reduce an archive member to the study path used by the CheXpert Plus tables.

    Members look like 'PNG/train/patient00003/study1/view1_frontal.png' or similar; the tables
    use 'train/patient00003/study1/view1_frontal.jpg'. The index stores the member verbatim and
    the key with the extension stripped, so either spelling resolves.
    """
    m = member.replace("\\", "/")
    parts = m.split("/")
    for i, p in enumerate(parts):
        if p.startswith("patient"):
            key = "/".join(parts[max(0, i - 1):])
            break
    else:
        return None
    if key.lower().endswith(".png") or key.lower().endswith(".jpg"):
        key = key.rsplit(".", 1)[0]
    return key


def main():
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    by_size = load_manifest()
    chunk_map = {}
    for name in sorted(os.listdir(ZIPDIR)):
        path = os.path.join(ZIPDIR, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        chunk_map[name] = by_size.get(size, "UNKNOWN_SIZE_%d" % size)
    with open(os.path.join(OUTDIR, "chunk_map.json"), "w", encoding="utf-8") as fh:
        json.dump(chunk_map, fh, indent=2, sort_keys=True)
    for k, v in sorted(chunk_map.items()):
        print("CHUNK %s = %s" % (k, v))

    index_path = os.path.join(OUTDIR, "image_index.tsv")
    n_total = 0
    n_bad = 0
    per_chunk = {}
    samples = []
    with open(index_path, "w", encoding="utf-8") as out:
        for name in sorted(chunk_map):
            path = os.path.join(ZIPDIR, name)
            try:
                zf = zipfile.ZipFile(path)
            except zipfile.BadZipFile as exc:
                print("BAD_ZIP %s %s" % (name, exc))
                n_bad += 1
                continue
            count = 0
            with zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    key = normalise(member)
                    if key is None:
                        continue
                    out.write("%s\t%s\t%s\n" % (key, name, member))
                    count += 1
                    if len(samples) < 3 and count == 1:
                        samples.append((name, member, key))
            per_chunk[name] = count
            n_total += count
            print("INDEXED %s (%s) members=%d" % (name, chunk_map[name], count))
            sys.stdout.flush()

    extracted = []
    for name, member, key in samples:
        try:
            with zipfile.ZipFile(os.path.join(ZIPDIR, name)) as zf:
                data = zf.read(member)
            dest = os.path.join(CACHE, key.replace("/", "__") + ".png")
            with open(dest, "wb") as fh:
                fh.write(data)
            extracted.append((key, len(data), hashlib.sha256(data).hexdigest()[:16]))
            print("EXTRACT_OK %s bytes=%d" % (key, len(data)))
        except Exception as exc:  # noqa: BLE001 - a failure here is the result
            print("EXTRACT_FAIL %s %s" % (key, exc))

    notes = os.path.join(OUTDIR, "index_notes_%s.md" % stamp)
    with open(notes, "w", encoding="utf-8") as fh:
        fh.write("# CheXpert Plus image index, %s\n\n" % stamp)
        fh.write("Archives read from `%s`, identified by byte size against "
                 "`PNG_compressed` in the export manifest.\n\n" % ZIPDIR)
        for k in sorted(chunk_map):
            fh.write("- `%s` = `%s`, members indexed %d\n"
                     % (k, chunk_map[k], per_chunk.get(k, 0)))
        fh.write("\nTotal members indexed: %d; unreadable archives: %d\n\n" % (n_total, n_bad))
        fh.write("Test extractions:\n\n")
        for key, size, digest in extracted:
            fh.write("- `%s` %d bytes sha256 `%s...`\n" % (key, size, digest))
        fh.write("\nIndex file: `image_index.tsv` (study path, archive, member).\n")
    print("TOTAL_INDEXED %d" % n_total)
    print("NOTES %s" % notes)


if __name__ == "__main__":
    main()
