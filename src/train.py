import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """(H,W,3) or (H,W) -> uint8 RGB array"""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] != 3:
        raise ValueError(f"Unexpected shape: {img.shape}")

    if img.dtype == np.uint8:
        return img
    # float or other numeric
    x = img.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 1.5:  # assume 0..1
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def center_crop_resize_to_tensor(img_uint8_rgb: np.ndarray, size: int = 160) -> torch.Tensor:
    """uint8 RGB -> torch tensor (3,size,size) float in [0,1]"""
    pil = Image.fromarray(img_uint8_rgb, mode="RGB")
    w, h = pil.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    pil = pil.crop((left, top, left + m, top + m))
    pil = pil.resize((size, size), resample=Image.BILINEAR)

    arr = np.asarray(pil, dtype=np.uint8)  # (size,size,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


@torch.no_grad()
def align_resize(
    X: np.ndarray, y: np.ndarray, names: np.ndarray, image_size: int = 160
):
    print(f"[1/4] resize-based preprocessing 시작 (N={len(X)})", flush=True)

    aligned = []
    y_kept = []

    for img, label in tqdm(zip(X, y), total=len(X), desc="Preprocess(resize)", ncols=100):
        img_u8 = to_uint8_rgb(img)
        face = center_crop_resize_to_tensor(img_u8, size=image_size)
        aligned.append(face)
        y_kept.append(int(label))

    y_kept = np.array(y_kept, dtype=np.int64)
    print(f"[1/4] preprocessing 완료: kept={len(aligned)}", flush=True)

    # 클래스별 최소 2장 유지
    cnt = Counter(y_kept.tolist())
    keep_classes = sorted([c for c, k in cnt.items() if k >= 2])
    if len(keep_classes) < 2:
        raise RuntimeError("클래스가 너무 적습니다(각 클래스 최소 2장 필요).")

    mask = np.array([lbl in keep_classes for lbl in y_kept], dtype=bool)
    aligned = [t for t, m in zip(aligned, mask) if m]
    y_kept = y_kept[mask]

    mapping = {c: i for i, c in enumerate(keep_classes)}
    y_remap = np.array([mapping[int(lbl)] for lbl in y_kept], dtype=np.int64)
    names_remap = np.array([names[c] for c in keep_classes], dtype=object)

    print(f"[1/4] 클래스 재매핑: K={len(names_remap)}, total={len(y_remap)}", flush=True)
    return aligned, y_remap, names_remap


@torch.no_grad()
def align_mtcnn(
    X: np.ndarray, y: np.ndarray, names: np.ndarray, device: str, image_size: int = 160, margin: int = 20
):
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        select_largest=True,
        post_process=True,
        device=device,
    )

    aligned = []
    y_kept = []
    fail = 0

    print(f"[1/4] MTCNN face alignment 시작 (N={len(X)})", flush=True)
    for img, label in tqdm(zip(X, y), total=len(X), desc="Align(MTCNN)", ncols=100):
        img_u8 = to_uint8_rgb(img)
        pil = Image.fromarray(img_u8, mode="RGB")
        face = mtcnn(pil)  # (3,160,160) or None
        if face is None:
            fail += 1
            continue
        aligned.append(face.detach().cpu())
        y_kept.append(int(label))

    print(f"[1/4] alignment 완료: kept={len(aligned)}, failed={fail}", flush=True)
    if len(aligned) < 10:
        raise RuntimeError("정렬된 얼굴이 너무 적습니다. --align resize로 실행해보세요.")

    y_kept = np.array(y_kept, dtype=np.int64)

    cnt = Counter(y_kept.tolist())
    keep_classes = sorted([c for c, k in cnt.items() if k >= 2])
    if len(keep_classes) < 2:
        raise RuntimeError("클래스가 너무 적습니다(각 클래스 최소 2장 필요).")

    mask = np.array([lbl in keep_classes for lbl in y_kept], dtype=bool)
    aligned = [t for t, m in zip(aligned, mask) if m]
    y_kept = y_kept[mask]

    mapping = {c: i for i, c in enumerate(keep_classes)}
    y_remap = np.array([mapping[int(lbl)] for lbl in y_kept], dtype=np.int64)
    names_remap = np.array([names[c] for c in keep_classes], dtype=object)

    print(f"[1/4] 클래스 재매핑: K={len(names_remap)}, total={len(y_remap)}", flush=True)
    return aligned, y_remap, names_remap


@torch.no_grad()
def embed_faces(aligned_tensors_cpu, device: str, batch_size: int = 64):
    resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    N = len(aligned_tensors_cpu)
    embs = []
    print(f"[2/4] FaceNet 임베딩 추출 시작 (N={N}, batch={batch_size})", flush=True)

    for i in tqdm(range(0, N, batch_size), desc="Embed(FaceNet)", ncols=100):
        batch = aligned_tensors_cpu[i : i + batch_size]
        x = torch.stack(batch, dim=0).to(device)  # (B,3,160,160)
        z = resnet(x)  # (B,512)
        embs.append(z.detach().cpu().numpy())

    X_emb = np.concatenate(embs, axis=0)
    print(f"[2/4] 임베딩 완료: shape={X_emb.shape}", flush=True)
    return X_emb


def train_knn(X_emb: np.ndarray, y: np.ndarray, n_neighbors: int = 3):
    print("[3/4] KNN 학습/평가 시작", flush=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_emb, y, test_size=0.2, random_state=4, stratify=y
    )

    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="cosine",
        weights="distance",
    )
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"[3/4] Accuracy: {acc:.4f}", flush=True)
    print("[3/4] Classification report:", flush=True)
    print(classification_report(y_te, pred, target_names=[str(n) for n in np.unique(y_te)], zero_division=0), flush=True)

    return clf, (X_tr, X_te, y_tr, y_te, pred, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/lfw_subset.npz")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--art_dir", type=str, default="artifacts")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--align", type=str, default="resize", choices=["resize", "mtcnn"])
    args = ap.parse_args()

    t0 = time.time()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    art_dir = Path(args.art_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print(f"Loading: {data_path.resolve()}", flush=True)
    d = np.load(data_path, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(np.int64)
    names = d["names"]
    print(f"Loaded X={X.shape}, y={y.shape}, K={len(names)}", flush=True)

    # 1) align/preprocess
    if args.align == "resize":
        aligned, y2, names2 = align_resize(X, y, names, image_size=args.image_size)
    else:
        aligned, y2, names2 = align_mtcnn(X, y, names, device=device, image_size=args.image_size, margin=args.margin)

    # 2) embed
    X_emb = embed_faces(aligned, device=device, batch_size=args.batch_size)

    emb_path = art_dir / "embeddings_lfw_subset.npz"
    np.savez_compressed(emb_path, X=X_emb, y=y2, names=names2)
    print(f"[2/4] embeddings 저장: {emb_path.resolve()}", flush=True)

    # 3) train
    clf, pack = train_knn(X_emb, y2, n_neighbors=args.k)

    # 4) save model bundle
    model_path = out_dir / "knn_facenet_lfw_subset.joblib"
    joblib.dump(
        {
            "model": clf,
            "class_names": names2,
            "params": {
                "k": args.k,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "margin": args.margin,
                "align": args.align,
                "device": device,
            },
        },
        model_path
    )
    print(f"[4/4] 모델 저장 완료: {model_path.resolve()}", flush=True)

    dt = time.time() - t0
    print(f"Done. elapsed={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
