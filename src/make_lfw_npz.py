import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.datasets import fetch_lfw_people

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/lfw_subset.npz")
    ap.add_argument("--cache_dir", type=str, default="/tmp/sklearn_lfw_cache")
    ap.add_argument("--min_faces", type=int, default=10)
    ap.add_argument("--top_n_people", type=int, default=100)
    ap.add_argument("--max_per_person", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    lfw = fetch_lfw_people(
        data_home=args.cache_dir,
        color=True,
        funneled=True,
        resize=1.0,
        min_faces_per_person=args.min_faces,
        download_if_missing=True,
    )

    X = lfw.images          # (N,H,W,3) uint8
    y = lfw.target.astype(np.int64)
    names = lfw.target_names

    idx_by = defaultdict(list)
    for i, label in enumerate(y):
        idx_by[int(label)].append(i)

    # 많이 가진 사람부터 top_n_people 선택
    classes = sorted(idx_by.keys(), key=lambda c: len(idx_by[c]), reverse=True)[:args.top_n_people]

    rng = np.random.default_rng(args.seed)
    Xs, ys, name_list = [], [], []
    for new_label, c in enumerate(classes):
        idc = np.array(idx_by[c], dtype=np.int64)
        rng.shuffle(idc)
        idc = idc[:args.max_per_person]
        Xs.append(X[idc])
        ys.append(np.full(len(idc), new_label, dtype=np.int64))
        name_list.append(names[c])

    X_sub = np.concatenate(Xs, axis=0)
    y_sub = np.concatenate(ys, axis=0)
    names_sub = np.array(name_list, dtype=object)

    np.savez_compressed(out, X=X_sub, y=y_sub, names=names_sub)
    print("saved:", out.resolve())
    print("X:", X_sub.shape, "y:", y_sub.shape, "classes:", len(names_sub))
    print("per-class min/max:", int(np.min(np.bincount(y_sub))), int(np.max(np.bincount(y_sub))))

if __name__ == "__main__":
    main()
