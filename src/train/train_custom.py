"""
Model B — 본 연구 데이터셋 기반 KPF-BERT 파인튜닝

최종 앙상블의 주 구성 요소(가중치 0.8). LLM 이 생성한 5가지 조작 유형의
가짜뉴스를 직접 학습한다.

라벨 방향 주의:
    본 스크립트는 is_fake 를 그대로 사용하므로 1=가짜 이다.
    Model A (train_aihub.py) 는 AI Hub 의 useType 을 쓰므로 0=가짜 로 반대다.
    앙상블 시 두 모델의 확률 추출 인덱스가 달라야 한다.
    상세는 ../../serving/README.md 2절 참조.

설계 의도:
    - 검증 분할 없이 전체 데이터로 학습한다. 앙상블 구성 요소로서
      데이터를 최대한 활용하기 위함이며, 하이퍼파라미터는 별도 실험
      (experiments/klue_bert_hparam_search.py)에서 확정한 값을 적용했다.
    - weight_decay 를 0.1 로 높여 표면적 패턴에 대한 과적합을 억제한다.
      기존 모델의 전이 실패(F1 0.9739 → 0.0587)가 그 과적합의 결과이므로,
      동일한 실패를 반복하지 않기 위한 조치다.

데이터 준비:
    기사 원문은 저작권상 저장소에 포함하지 않는다.
    아래 위치에 학습 데이터를 배치한다.

        src/train/FakeNews_train_v5.xlsx

    또는 환경변수로 경로를 지정한다.

        CUSTOM_TRAIN_PATH=/path/to/FakeNews_train_v5.xlsx

    필요 컬럼: title_clean, content_clean, is_fake

실행:
    python train_custom.py
"""

import os

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.getenv(
    "CUSTOM_TRAIN_PATH",
    os.path.join(BASE_DIR, "FakeNews_train_v5.xlsx"),
)
OUTPUT_DIR = os.getenv(
    "MODEL_B_PATH",
    os.path.join(BASE_DIR, "model_custom"),
)
TMP_DIR    = os.path.join(BASE_DIR, "tmp_custom_results")

MODEL_NAME    = "jinmang2/kpfbert"
MAX_LENGTH    = 512
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.1
WARMUP_RATIO  = 0.1
SEED          = 42

REQUIRED_COLUMNS = ["title_clean", "content_clean", "is_fake"]


# ────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────
def load_custom_data(file_path: str) -> pd.DataFrame:
    """xlsx 또는 csv 에서 학습 데이터를 읽어 label 컬럼으로 정규화한다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"학습 데이터를 찾을 수 없습니다: {file_path}\n"
            f"파일을 배치하거나 환경변수 CUSTOM_TRAIN_PATH 로 경로를 지정하세요."
        )

    print(f"[로드] {file_path}")

    if file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if before != len(df):
        print(f"[정제] 결측치 제거: {before - len(df)}건")

    # is_fake: 0=진짜, 1=가짜  →  label 로 이름만 변경 (의미 동일)
    df = df.rename(columns={"is_fake": "label"})
    df["label"] = df["label"].astype(int)

    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    print(f"[데이터] 총 {len(df)}건  (진짜 {n_real} / 가짜 {n_fake})")

    return df


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  Model B — 본 연구 데이터 기반 KPF-BERT 파인튜닝")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    train_df = load_custom_data(TRAIN_PATH)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    def tokenize(batch):
        return tokenizer(
            batch["title_clean"],
            batch["content_clean"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("[토큰화] 진행 중")
    train_ds = train_ds.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir                  = TMP_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 1,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = WARMUP_RATIO,
        eval_strategy               = "no",    # 앙상블용 — 전체 데이터로 학습
        save_strategy               = "no",
        fp16                        = torch.cuda.is_available(),
        seed                        = SEED,
        report_to                   = "none",
    )

    trainer = Trainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        processing_class = tokenizer,
        data_collator    = DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("\n[학습] 시작")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"[저장] {OUTPUT_DIR}")
    print("[안내] serving/app.py 가 이 경로를 MODEL_B_PATH 로 참조한다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
