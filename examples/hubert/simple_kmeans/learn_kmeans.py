import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    # 构建目录路径和搜索模式
    pattern = f"{split}_{rank}_{nshard}_"
    file_list = [f for f in os.listdir(feat_dir) if f.startswith(pattern)]

    if len(file_list) == 0:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {feat_dir}")

    # 初始化一个空的列表来存储采样的特征
    sampled_features = []

    # 总的帧数，用于计算需要采样的总帧数
    total_frames = 0

    # 首先遍历文件以计算总帧数
    for filename in tqdm(file_list):
        feat = np.load(os.path.join(feat_dir, filename), mmap_mode='r')
        total_frames += len(feat)

    # 计算需要采样的帧数
    nsample = int(np.ceil(total_frames * percent))

    # 已采样的帧数
    sampled_count = 0

    # 打乱文件列表以进行随机采样
    np.random.shuffle(file_list)

    for filename in tqdm(file_list):
        feat = np.load(os.path.join(feat_dir, filename), mmap_mode='r')
        # 计算当前文件中需要采样的帧数
        current_sample_size = int(np.ceil(len(feat) * (nsample / total_frames)))
        # 进行采样
        if sampled_count + current_sample_size > nsample:
            current_sample_size = nsample - sampled_count
        if current_sample_size > 0:
            indices = np.random.choice(len(feat), current_sample_size, replace=False)
            sampled_features.append(feat[indices])
            sampled_count += current_sample_size
        # 如果已达到需要的采样数量，停止加载更多文件
        if sampled_count >= nsample:
            break

    # 将所有采样的特征拼接起来
    final_sampled_features = np.concatenate(sampled_features, axis=0)
    
    # 记录日志
    logger.info(f"Sampled {nsample}/{total_frames} frames from shard {rank}/{nshard} containing {len(file_list)} files")
    
    return final_sampled_features

def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir",           type=str,   default="data/metadata")
    parser.add_argument("--split",              type=str,   default="valid")
    parser.add_argument("--nshard",             type=int,   default=1)
    parser.add_argument("--km_path",            type=str,   default="data/label/train_km")
    parser.add_argument("--n_clusters",         type=int,   default=500)
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--percent",            type=float, default=0.5, help="sample a subset; -1 for all")
    parser.add_argument("--init",               type=str,   default="k-means++")
    parser.add_argument("--max_iter",           type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=10000)
    parser.add_argument("--tol",                type=float, default=0.0)
    parser.add_argument("--max_no_improvement", type=int,   default=100)
    parser.add_argument("--n_init",             type=int,   default=20)
    parser.add_argument("--reassignment_ratio", type=float, default=0.0)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))