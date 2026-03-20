#!/usr/bin/env python3
"""
下载 MS MARCO Passage Ranking 数据集
"""

import os
import tarfile
import gzip
import requests
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import MSMARCO_DIR, MSMARCO_URL, MSMARCO_COLLECTION


def download_file(url: str, dest: Path, desc: str = None):
    """下载文件并显示进度条"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_tar_gz(tar_path: Path, dest_dir: Path, desc: str = None):
    """解压 tar.gz 文件"""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=desc):
            tar.extract(member, dest_dir)


def main():
    print("=" * 60)
    print("MS MARCO Passage Ranking Dataset Downloader")
    print("=" * 60)
    
    # 检查数据是否已存在
    if MSMARCO_COLLECTION.exists():
        print(f"\n✓ Dataset already exists at {MSMARCO_COLLECTION}")
        print("  Skipping download. Delete the file to re-download.")
        return
    
    # 创建目录
    MSMARCO_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载数据集
    print(f"\n[1/2] Downloading from {MSMARCO_URL}")
    tar_path = MSMARCO_DIR / "collectionandqueries.tar.gz"
    download_file(MSMARCO_URL, tar_path, desc="MS MARCO")
    
    # 解压
    print(f"\n[2/2] Extracting...")
    extract_tar_gz(tar_path, MSMARCO_DIR, desc="Extracting")
    
    # 验证
    print("\n[3/3] Verifying...")
    expected_files = ["collection.tsv", "queries.tar.gz"]
    for f in expected_files:
        if (MSMARCO_DIR / f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")
    
    # 清理
    print("\n[4/4] Cleaning up...")
    tar_path.unlink()
    print(f"  Removed {tar_path.name}")
    
    print("\n" + "=" * 60)
    print("✓ MS MARCO dataset ready!")
    print(f"  Collection: {MSMARCO_COLLECTION}")
    print("=" * 60)


if __name__ == "__main__":
    main()