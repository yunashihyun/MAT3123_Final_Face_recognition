# MAT3123_Final_Face_recognition
기계학습과 응용 기말 프로젝트

아래 내용을 그대로 `README.md`로 넣으시면 됩니다(보고서+실행법 통합본). 필요하면 마지막에 본인 이름/학번/분반만 추가하세요.

---

# MAT3123 Final – Face Recognition (FaceNet + KNN)

기계학습과 응용 기말 프로젝트로, **사전학습된 얼굴 임베딩 모델(FaceNet)** 을 이용해 얼굴 이미지를 **512차원 임베딩**으로 변환한 뒤, 그 임베딩 공간에서 **K-Nearest Neighbors(KNN)** 로 분류하는 얼굴 인식 파이프라인을 구현했습니다.

## 1. Dataset

* **LFW (Labeled Faces in the Wild)** 기반 얼굴 이미지 데이터셋을 사용했습니다.
* 프로젝트 실행 편의를 위해 학습에 사용한 이미지는 전처리 후 **`.npz` 파일**로 저장하여 로딩합니다.
* 본 저장 파일에는 다음이 포함됩니다.

  * `X`: (N, H, W, 3) 형태의 얼굴 이미지 배열
  * `y`: (N,) 정수 라벨
  * `names`: 클래스(인물) 이름 리스트

> 본 레포지토리에는 대용량 원본 데이터는 포함하지 않고, 코드/모델 중심으로 관리합니다.

---

## 2. Method

### 2.1 Face Alignment / Preprocess

입력 이미지를 다음 중 하나로 전처리합니다.

* `--align mtcnn` (기본):
  `facenet_pytorch`의 **MTCNN** 으로 얼굴을 탐지한 뒤 정렬(alignment)하여 `(160×160)` 크기로 맞춥니다.
* `--align resize`:
  탐지를 생략하고 이미지 전체를 `(160×160)`으로 리사이즈합니다(데이터셋이 이미 얼굴 중심인 경우에만 권장).

### 2.2 Embedding (FaceNet)

* **InceptionResnetV1(pretrained="vggface2")** (FaceNet 계열) 사용
* 각 이미지(얼굴)를 **512-d embedding**으로 변환

### 2.3 Classifier (KNN)

* 학습 임베딩에 대해 **KNN 분류기**를 학습
* 추론 시 입력 임베딩의 **Top-k 이웃 거리**를 함께 출력해 예측 근거(가까운 샘플들)를 확인할 수 있게 했습니다.

## Experiment Setup (Report)

- **Task setting:** Closed-set face classification (predict among the people seen in training `names`)
- **Embedding model:** FaceNet (InceptionResnetV1, `pretrained="vggface2"`)
- **Face alignment:** MTCNN (`--align mtcnn`, fallback option: `--align resize`)
- **Classifier:** k-Nearest Neighbors (KNN)
- **Distance metric:** Euclidean distance in the 512-d embedding space (default in sklearn KNN)
- **Hyperparameter:** `k = 3` (used during training)

### Evaluation
- **Sanity-check evaluation:** Prepared 10 test images with filenames containing ground-truth labels (e.g., `test_0_y0.jpg`)
- **Metric:** Top-1 accuracy
- **Result:** `Accuracy = 10/10 = 1.0000` on the prepared test set
- **Note:** For identities not included in the training label set (`names`), the model will still output the closest known class (open-set limitation).

---

## 3. Repository Structure

```
.
├─ src/
│  ├─ train.py        # 임베딩 추출 + KNN 학습 + 저장
│  └─ infer.py        # 이미지(여러 장 가능) 추론 + topk 출력
├─ models/
│  ├─ knn_facenet_lfw_subset.joblib   # 학습된 KNN 모델
│  └─ .gitkeep
├─ artifacts/         # (선택) embeddings_*.npz 등 중간 산출물 폴더
├─ infer_images/      # 추론할 이미지 넣는 폴더(여러 장 가능)
├─ requirements.txt
└─ requirements-gpu.txt
```

---

## 4. Environment

* Python 3.10
* 주요 라이브러리

  * `torch`
  * `facenet-pytorch` (MTCNN, InceptionResnetV1)
  * `scikit-learn`
  * `joblib`
  * `Pillow`, `numpy`

설치:

```bash
pip install -r requirements.txt
# 또는 GPU 환경이면
pip install -r requirements-gpu.txt
```

---

## 5. How to Run

### 5.1 (선택) 가상환경 활성화

```bash
cd /root/mat
source .venv/bin/activate
```

### 5.2 Train (KNN 모델 학습)

예시:

```bash
python src/train.py \
  --data data/lfw_subset.npz \
  --out_dir models \
  --art_dir artifacts \
  --batch_size 64 \
  --k 3 \
  --align resize
```

* `--data`: 학습용 `.npz` 경로
* `--k`: KNN의 k 값
* `--align`: `resize` 또는 `mtcnn`

학습이 끝나면 `models/*.joblib`로 저장됩니다.

---

### 5.3 Inference (새 이미지 예측)

1. 추론할 이미지를 폴더에 넣습니다(여러 장 가능).

```bash
mkdir -p infer_images
# infer_images/ 아래에 jpg/png 등 복사
```

2. 실행:

```bash
python src/infer.py \
  --input infer_images \
  --model models/knn_facenet_lfw_subset.joblib \
  --art_dir artifacts \
  --align mtcnn \
  --topk 5
```

* `--input`은 **이미지 1장 경로** 또는 **폴더 경로** 모두 가능합니다.
* 출력에는 각 파일의 `pred`와 `topk (거리/클래스)`가 표시됩니다.

---

## 6. Result Verification (보고서용 확인)

학습 데이터(또는 테스트 세트)에서 샘플을 뽑아 파일명에 정답 라벨을 포함시키는 방식으로 간단 검증을 수행했습니다. 예:

* `test_0_y0.jpg` 처럼 파일명에 `y{label}` 포함
* 추론 결과와 비교하여 `Accuracy: 10/10 = 1.0000` 형태로 확인

또한 KNN의 특성상, **학습에 포함된 인물 클래스(names) 집합 밖의 “완전히 새로운 인물”** 에 대해서는 “가장 가까운 학습 인물”로 분류되므로,
실제 오픈셋(open-set) 인식 용도로는 한계가 있습니다. 본 프로젝트에서는 **폐쇄집합(closed-set) 분류** 문제로 설정하여, LFW의 선택된 클래스 범위 내에서 인물을 구분하는 데 초점을 두었습니다.

---

## 7. Notes (운영 이슈)

실행 환경에서 디스크(inode) 문제가 발생할 수 있어, 임시/캐시 경로를 `/tmp`로 두는 방식으로 해결했습니다. 필요 시 아래처럼 환경변수를 지정해 실행할 수 있습니다.

```bash
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export TORCH_HOME=/tmp/torch_home
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_home
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME"
```

---

## 8. What was implemented

* FaceNet 임베딩 기반 KNN 얼굴 인식 파이프라인
* MTCNN alignment 옵션 제공
* 여러 이미지 일괄 추론 및 Top-k 근거 출력
* 학습 모델(`.joblib`) 저장 및 재사용
