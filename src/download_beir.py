#!/usr/bin/env python3
"""
下载 BEIR benchmark 数据集
"""

import sys
from pathlib import Path
from beir import util
from beir.datasets.data_loader import GenericDataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import BEIR_DIR, BEIR_DATASETS


def download_beir_dataset(dataset_name: str):
    """下载单个 BEIR 数据集"""
    print(f"\n{'='*60}")
    print(f"Downloading BEIR dataset: {dataset_name}")
    print("="*60)
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = BEIR_DIR / dataset_name
    
    if data_path.exists():
        print(f"✓ Dataset already exists at {data_path}")
        return
    
    # 下载数据
    print(f"Downloading from {url}")
    zip_path = util.download_url(url, BEIR_DIR)
    
    # 解压
    print(f"Unzipping {zip_path}")
    util.unzip(zip_path, BEIR_DIR)
    
    # 加载验证
    print(f"Loading {dataset_name}...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} query-document pairs")
    
    print(f"✓ {dataset_name} ready!")


def main():
    print("=" * 60)
    print("BEIR Benchmark Dataset Downloader")
    print("=" * 60)
    
    BEIR_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载所有数据集
    for dataset_name in BEIR_DATASETS:
        try:
            download_beir_dataset(dataset_name)
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ BEIR benchmark datasets ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()