"""
Convert MovieLens 1M ratings.dat to format suitable for TiSASRec

ratings.dat format: UserID::MovieID::Rating::Timestamp
Output format: UserID MovieID Timestamp

按用户和时间戳排序后保存
"""

import os
from collections import defaultdict

DATA_DIR = "/home/yimo/Repositories/SASRec.pytorch/python/data"
INPUT_FILE = os.path.join(DATA_DIR, "ml-1m/ratings.dat")
OUTPUT_FILE = os.path.join(DATA_DIR, "ml-1m.txt")


def convert_ratings():
    print(f"Reading {INPUT_FILE}...")

    # 按用户存储 (user_id -> [(timestamp, movie_id)])
    user_interactions = defaultdict(list)

    with open(INPUT_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id = int(parts[0])
            movie_id = int(parts[1])
            timestamp = int(parts[3])
            user_interactions[user_id].append((timestamp, movie_id))

    # 按时间戳排序每个用户的交互
    print(f"Processing {len(user_interactions)} users...")
    for user_id in user_interactions:
        user_interactions[user_id].sort(key=lambda x: x[0])

    # 写入输出文件
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for user_id in sorted(user_interactions.keys()):
            for timestamp, movie_id in user_interactions[user_id]:
                f.write(f"{user_id} {movie_id} {timestamp}\n")

    # 统计
    total_lines = sum(len(interactions) for interactions in user_interactions.values())
    print(f"Done! Total interactions: {total_lines}")
    print(f"Output saved to: {OUTPUT_FILE}")

    # 验证
    print("\n=== Output verification ===")
    print("First 10 lines:")
    with open(OUTPUT_FILE, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(line.strip())


if __name__ == "__main__":
    convert_ratings()
