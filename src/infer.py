# src/infer.py
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend(p.rglob(f"*{ext}"))
            files.extend(p.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    raise FileNotFoundError(f"Input not found: {p}")


def load_names_from_npz(npz_path: Path) -> Optional[np.ndarray]:
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    if "names" in d:
        return d["names"]
    return None


def find_names(art_dir: Path, explicit_npz: Optional[Path]) -> Optional[np.ndarray]:
    # 1) explicit
    if explicit_npz is not None:
        names = load_names_from_npz(explicit_npz)
        if names is not None:
            return names

    # 2) auto-search in art_dir (embeddings_*.npz)
    if art_dir.exists():
        cand = sorted(art_dir.glob("embeddings_*.npz"))
        for c in cand:
            names = load_names_from_npz(c)
            if names is not None:
                return names
    return None


def preprocess_resize(img: Image.Image, out_hw: Tuple[int, int] = (160, 160)) -> torch.Tensor:
    # RGB, [0,1], CHW
    img = img.convert("RGB").resize(out_hw, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return t


@torch.inference_mode()
def align_mtcnn(
    imgs: List[Image.Image],
    device: torch.device,
    image_size: int = 160,
    margin: int = 10,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Returns:
      faces: list of aligned face tensors (3, image_size, image_size)
      kept_idx: indices of original imgs kept
    """
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=False,          # 한 장당 얼굴 1개만 (가장 큰 얼굴)
        post_process=True,
        device=device,
    )

    faces: List[torch.Tensor] = []
    kept_idx: List[int] = []
    for i, im in enumerate(imgs):
        # mtcnn returns tensor (3, image_size, image_size) or None
        face = mtcnn(im)
        if face is None:
            continue
        faces.append(face.cpu())
        kept_idx.append(i)
    return faces, kept_idx


@torch.inference_mode()
def embed_faces(face_tensors: torch.Tensor, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """
    face_tensors: (N, 3, 160, 160) float tensor in [0,1]
    returns: (N, 512) float32 numpy
    """
    resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    embs = []
    n = face_tensors.shape[0]
    for s in range(0, n, batch_size):
        batch = face_tensors[s:s + batch_size].to(device)
        # facenet expects standardized roughly; it works fine with [0,1] + internal post_process,
        # but we keep it simple here.
        e = resnet(batch).detach().cpu().numpy().astype(np.float32)
        embs.append(e)
    return np.concatenate(embs, axis=0)


def load_model(model_path: Path):
    obj = joblib.load(model_path)
    knn, names = unwrap_knn(obj)   # <-- dict든 KNN이든 여기서 정리
    return knn, names


def knn_topk(knn, X_emb: np.ndarray, topk: int = 5):
    # distances: smaller = closer
    dists, idxs = knn.kneighbors(X_emb, n_neighbors=topk, return_distance=True)
    preds = knn.predict(X_emb)
    return preds, dists, idxs
    
def unwrap_knn(obj):
    """
    joblib.load 결과가 KNN 객체일 수도, dict로 포장된 형태일 수도 있어서 둘 다 처리.
    return: (knn_model, class_names_or_None)
    """
    names = None

    # 1) 이미 sklearn KNN이면 그대로
    if hasattr(obj, "kneighbors"):
        return obj, None

    # 2) dict로 저장된 경우
    if isinstance(obj, dict):
        # names 후보
        for nk in ("names", "class_names", "target_names", "label_names"):
            if nk in obj:
                names = obj[nk]
                break

        # knn 후보 key 우선순위
        for kk in ("knn", "model", "clf", "classifier"):
            if kk in obj and hasattr(obj[kk], "kneighbors"):
                return obj[kk], names

        # 혹시 다른 키에 들어있으면 탐색
        for k, v in obj.items():
            if hasattr(v, "kneighbors"):
                return v, names

    raise TypeError(f"Loaded object is not a KNN model. type={type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="이미지 파일 1장 또는 이미지가 들어있는 폴더 경로")
    ap.add_argument("--model", default="models/knn_facenet_lfw_subset.joblib", help="저장된 KNN 모델(joblib)")
    ap.add_argument("--art_dir", default="artifacts", help="embeddings_*.npz 등 아티팩트 폴더")
    ap.add_argument("--names_npz", default=None, help="클래스 이름(names) 들어있는 npz를 직접 지정(선택)")
    ap.add_argument("--align", choices=["mtcnn", "resize"], default="mtcnn")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mtcnn_margin", type=int, default=10)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    in_path = Path(args.input)
    model_path = Path(args.model)
    art_dir = Path(args.art_dir)
    names_npz = Path(args.names_npz) if args.names_npz else None

    print(f"Device: {device}", flush=True)
    print(f"Input: {in_path.resolve()}", flush=True)
    print(f"Model: {model_path.resolve()}", flush=True)
    print(f"Align: {args.align}", flush=True)

    paths = list_images(in_path)
    if len(paths) == 0:
        raise RuntimeError("No images found.")
    print(f"Found {len(paths)} image(s)", flush=True)

    knn, names_from_joblib = load_model(model_path)
    names = names_from_joblib
    if names is None:
        names = find_names(art_dir, names_npz)

    # load images
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[skip] {p} (open failed: {e})", flush=True)

    if len(imgs) == 0:
        raise RuntimeError("All images failed to open.")

    # preprocess / align
    if args.align == "mtcnn":
        faces_list, kept_idx = align_mtcnn(
            imgs, device=device, image_size=160, margin=args.mtcnn_margin
        )
        if len(faces_list) == 0:
            raise RuntimeError("No face detected in any image. (mtcnn failed)")
        face_t = torch.stack(faces_list, dim=0).float()  # (N,3,160,160)
        kept_paths = [paths[i] for i in kept_idx]
    else:
        # resize-only: keep all
        face_t_list = [preprocess_resize(im, (160, 160)) for im in imgs]
        face_t = torch.stack(face_t_list, dim=0).float()
        kept_paths = paths[:len(imgs)]

    print(f"Kept {len(kept_paths)} / {len(paths)} after preprocessing", flush=True)

    # embed
    print("Embedding (FaceNet)...", flush=True)
    X_emb = embed_faces(face_t, device=device, batch_size=args.batch_size)
    print(f"Embeddings shape: {X_emb.shape}", flush=True)

    # predict
    preds, dists, idxs = knn_topk(knn, X_emb, topk=args.topk)

    # report
    print("\n===== Results =====", flush=True)
    for i, p in enumerate(kept_paths):
        pred = int(preds[i])
        pred_name = str(names[pred]) if names is not None and pred < len(names) else str(pred)

        print(f"\n[{i+1}] {p}", flush=True)
        print(f"  pred: {pred_name}", flush=True)
        print("  topk:", flush=True)
        for r in range(args.topk):
            nn_idx = int(idxs[i, r])
            nn_label = int(knn._y[nn_idx]) if hasattr(knn, "_y") else None  # sklearn internals
            nn_name = str(names[nn_label]) if (names is not None and nn_label is not None and nn_label < len(names)) else str(nn_label)
            print(f"    #{r+1}: dist={float(dists[i,r]):.4f}  class={nn_name}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()