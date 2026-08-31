"""
Model A — AI Hub 낚시성 기사 데이터셋 기반 KPF-BERT 파인튜닝

최종 앙상블의 구성 요소 중 하나. 본 연구 데이터와 다른 분포로 학습하여
미지의 입력에 대한 견고성을 보완하는 역할을 한다.

라벨 방향 주의:
    AI Hub 의 useType 은 1=진짜, 0=가짜 이다.
    Model B (train_custom.py) 는 is_fake 를 그대로 쓰므로 1=가짜 로 반대다.
    앙상블 시 두 모델의 확률 추출 인덱스가 달라야 한다.
    상세는 ../../serving/README.md 2절 참조.

데이터 준비:
    AI Hub 데이터셋은 라이선스상 저장소에 포함하지 않는다.
    https://aihub.or.kr/ 에서 내려받아 아래 위치에 JSON 파일을 배치한다.

        src/train/FakeNews_origin_dataset2/*.json

    또는 환경변수로 경로를 지정한다.

        AIHUB_DATA_DIR=/path/to/dataset

실행:
    python train_aihub.py
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

DATA_DIR   = os.getenv(
    "AIHUB_DATA_DIR",
    os.path.join(BASE_DIR, "FakeNews_origin_dataset2"),
)
OUTPUT_DIR = os.getenv(
    "MODEL_A_PATH",
    os.path.join(BASE_DIR, "model_aihub"),
)
TMP_DIR    = os.path.join(BASE_DIR, "tmp_aihub_results")

MODEL_NAME    = "jinmang2/kpfbert"
MAX_LENGTH    = 512
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
SEED          = 42

# 분할 비율 (train : val : test = 7 : 1.5 : 1.5)
TEST_VAL_RATIO = 0.3
VAL_TEST_SPLIT = 0.5


# ────────────────────────────────────────────────
# 1. 데이터 로드 및 분할
# ────────────────────────────────────────────────
def load_and_split_data(folder_path: str):
    """AI Hub JSON 파일을 읽어 DataFrame 으로 변환하고 7:1.5:1.5 로 분할한다."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {folder_path}\n"
            f"AI Hub 에서 데이터셋을 내려받아 배치하거나, "
            f"환경변수 AIHUB_DATA_DIR 로 경로를 지정하세요."
        )

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {folder_path}")

    print(f"[로드] {folder_path}  ({len(json_files)}개 파일)")

    records, skipped = [], 0
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            records.append({
                "title_clean"  : j["labeledDataInfo"]["newTitle"],
                "content_clean": j["sourceDataInfo"]["newsContent"],
                # useType: 1=진짜, 0=가짜
                "label"        : int(j["sourceDataInfo"]["useType"]),
            })
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            skipped += 1
            continue

    if skipped:
        print(f"[경고] 파싱 실패로 건너뛴 파일: {skipped}개")

    df = pd.DataFrame(records).dropna(
        subset=["title_clean", "content_clean", "label"]
    )
    if df.empty:
        raise ValueError("유효한 레코드가 없습니다. JSON 구조를 확인하세요.")

    train_df, val_test_df = train_test_split(
        df, test_size=TEST_VAL_RATIO, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=VAL_TEST_SPLIT, random_state=SEED,
        stratify=val_test_df["label"]
    )

    print(f"[분할] 학습 {len(train_df)} / 검증 {len(val_df)} / 평가 {len(test_df)}")
    print(f"[분포] 진짜 {(df['label'] == 1).sum()} / 가짜 {(df['label'] == 0).sum()}")
    return train_df, val_df, test_df


# ────────────────────────────────────────────────
# 2. 평가 지표
# ────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    macro 평균을 사용한다.

    주의: experiments/ 의 다른 스크립트는 이진 F1(양성 클래스 기준)을 쓰므로
    본 스크립트의 F1 과 직접 비교할 수 없다.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy" : accuracy,
        "f1"       : f1,
        "precision": precision,
        "recall"   : recall,
    }


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  Model A — AI Hub 데이터 기반 KPF-BERT 파인튜닝")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    train_df, val_df, test_df = load_and_split_data(DATA_DIR)

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

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir                  = TMP_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 1,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "linear",
        warmup_ratio                = WARMUP_RATIO,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        fp16                        = torch.cuda.is_available(),
        seed                        = SEED,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        data_collator   = DataCollatorWithPadding(tokenizer),
    )

    print("\n[학습] 시작")
    trainer.train()

    print("\n" + "=" * 60)
    print("  테스트셋 평가 결과 (macro 평균)")
    print("=" * 60)
    results = trainer.evaluate(test_ds)
    for key, value in results.items():
        if key.startswith("eval_") and isinstance(value, float):
            print(f"  {key.replace('eval_', ''):<12} {value:.4f}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[저장] {OUTPUT_DIR}")
    print("[안내] serving/app.py 가 이 경로를 MODEL_A_PATH 로 참조한다.")


if __name__ == "__main__":
    main()
