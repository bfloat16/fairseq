import logging
import os
import sys
import tqdm
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")

def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(f"rank {rank} of {nshard}, process {end-start} ({start}-{end}) out of {tot}")
    return start, end

def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r", encoding='utf-8') as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)

def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    os.makedirs(feat_dir, exist_ok=True)

    for i, (path, nsample) in enumerate(tqdm.tqdm(iterator, total=num)):
        feat = reader.get_feats(path, nsample).cpu().numpy()
        feat_path = f"{feat_dir}/{split}_{rank}_{nshard}_{i:07d}.npy"
        np.save(feat_path, feat)
    logger.info("finished successfully")