"""
KLUE-BERT Fine-Tuning 하이퍼파라미터 탐색
데이터: FakeNews_train_v5.xlsx

탐색 순서 (K-Fold 없이 단순 Train/Val 분할 - 그리디 탐색 방식):
  STEP 1. LR 탐색      (Batch=8, Epoch=3, MaxLen=256 고정)
  STEP 2. Epoch 탐색   (최적 LR, Batch=8, MaxLen=256 고정)
  STEP 3. Batch 탐색   (최적 LR, 최적 Epoch, MaxLen=256 고정)
  STEP 4. MaxLen 탐색  (최적 LR, 최적 Epoch, 최적 Batch 고정)

탐색 범위:
  LR:     1e-5, 2e-5, 5e-5
  Epoch:  2, 3
  Batch:  8, 16
  MaxLen: 128, 256, 512
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ────────────────────────────────────────────────
# 고정 설정 (Configuration & Search Space)
# ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH   = os.path.join(BASE_DIR, "FakeNews_train_v5.xlsx")
# TRAIN_PATH = os.path.join(BASE_DIR, "korean_fake_news_db.csv")

MODEL_NAME   = "klue/bert-base"
WARMUP_RATIO = 0.1  # 학습 초기에 학습률을 서서히 올리는 구간 비율
PATIENCE     = 2    # 검증 성능이 연속으로 개선되지 않을 때 참아주는 횟수 (Early Stopping)
VAL_RATIO    = 0.15 # Train 85% / Val 15% 분할
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# 탐색 범위 (이 리스트 안에 있는 값들을 하나씩 테스트합니다)
LR_LIST     = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
EPOCH_LIST  = [3, 4, 5]
BATCH_LIST  = [8, 16]
MAXLEN_LIST = [128, 256, 384, 512]

print(f"[설정] Device : {DEVICE}")
print(f"[설정] 모델   : {MODEL_NAME}")
print(f"[설정] 데이터 : FakeNews_train_v5.xlsx")
print(f"[설정] 분할   : Train {int((1-VAL_RATIO)*100)}% / Val {int(VAL_RATIO*100)}%")


# ────────────────────────────────────────────────
# 1. 데이터 로드 및 분할
# ────────────────────────────────────────────────
def load_and_split(path: str):
    """
    모든 하이퍼파라미터 실험이 동일한 조건(동일한 Train/Val)에서 진행되도록
    맨 처음에 한 번만 데이터를 로드하고 분할하는 함수입니다.
    """
    df     = pd.read_excel(path)
    # 제목과 본문을 하나의 텍스트로 병합
    texts  = (df["title_clean"].fillna("") + " " + df["content_clean"].fillna("")).tolist()
    labels = df["is_fake"].values

    # stratify=labels: Train과 Val의 가짜/진짜 뉴스 비율이 원본과 동일하게 유지되도록 분리
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels,
        test_size=VAL_RATIO, stratify=labels, random_state=42
    )
    print(f"[데이터] Train: {len(tr_labels)}개  Val: {len(val_labels)}개")
    return tr_texts, val_texts, tr_labels, val_labels


# ────────────────────────────────────────────────
# 2. Dataset 클래스
# ────────────────────────────────────────────────
class NewsDataset(Dataset):
    """모델 학습에 필요한 형태로 텍스트를 변환해주는 PyTorch 데이터셋"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, 
            padding=True,          # 짧은 문장은 패딩으로 길이 맞춤
            truncation=True,       # max_length보다 길면 자름
            max_length=max_length, 
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ────────────────────────────────────────────────
# 3. 학습 및 검증 (핵심 실험 함수)
# ────────────────────────────────────────────────
def train_and_eval(tr_texts, val_texts, tr_labels, val_labels,
                   lr, epochs, batch_size, max_length):
    """
    넘겨받은 특정 하이퍼파라미터 조합으로 모델을 처음부터 끝까지 학습하고
    최고 Validation F1 Score를 반환합니다.
    (이전 실험의 학습 상태가 남지 않도록 매번 모델을 새로 불러옵니다.)
    """
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    model        = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    
    train_loader = DataLoader(
        NewsDataset(tr_texts, tr_labels, tokenizer, max_length),
        batch_size=batch_size, shuffle=True
    )
    val_loader   = DataLoader(
        NewsDataset(val_texts, val_labels, tokenizer, max_length),
        batch_size=batch_size, shuffle=False
    )
    
    optimizer    = AdamW(model.parameters(), lr=lr)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    best_f1, patience_cnt = 0.0, 0

    for epoch in range(1, epochs + 1):
        # [1] Training (학습)
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"    Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 기울기 폭발 방지
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        # [2] Validation (검증)
        model.eval()
        preds, trues = [], []
        with torch.no_grad(): # 역전파 비활성화 (메모리 절약)
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                )
                preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
                trues.extend(batch["labels"].numpy())

        # F1 Score 계산
        val_f1   = f1_score(trues, preds)
        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch {epoch}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

        # [3] Early Stopping (조기 종료 로직)
        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0  # 최고 성능 갱신 시 카운터 리셋
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    [Early Stopping] 학습 종료")
                break

    # 해당 파라미터 조합에서의 최고 F1 반환
    return best_f1


# ────────────────────────────────────────────────
# 4. 결과 출력 헬퍼 함수
# ────────────────────────────────────────────────
def print_results(title, results, best_key):
    """현재 탐색 단계(Step)의 모든 결과를 요약해서 예쁘게 출력합니다."""
    print("\n" + "="*50)
    print(f"  {title} 탐색 결과")
    print("="*50)
    for key, f1 in results.items():
        mark = " ← 최적" if key == best_key else ""
        print(f"  {title}={key}  F1={f1:.4f}{mark}")
    print(f"\n  최적 {title}: {best_key}")


# ────────────────────────────────────────────────
# 메인 실행부 (Greedy Hyperparameter Search)
# ────────────────────────────────────────────────
if __name__ == "__main__":

    # 데이터 로드 및 분할 (모든 단계에서 이 데이터를 공유합니다)
    tr_texts, val_texts, tr_labels, val_labels = load_and_split(TRAIN_PATH)

    # ────────────────────────────────
    # STEP 1. LR(학습률) 탐색
    # ────────────────────────────────
    print("\n" + "="*50)
    print("  STEP 1. Learning Rate 탐색")
    print("  고정: Batch=8, Epoch=3, MaxLen=256") # 나머지는 임의의 기본값으로 통일
    print("="*50)

    lr_results = {}
    for lr in LR_LIST:
        print(f"\n[LR={lr:.0e}] 학습 시작...")
        f1 = train_and_eval(
            tr_texts, val_texts, tr_labels, val_labels,
            lr=lr, epochs=3, batch_size=8, max_length=256
        )
        lr_results[lr] = f1
        print(f"  → 최고 val_F1: {f1:.4f}")

    # 가장 높은 F1을 기록한 LR 추출
    best_lr = max(lr_results, key=lr_results.get)
    print_results("LR", {f"{k:.0e}": v for k, v in lr_results.items()}, f"{best_lr:.0e}")

    # ────────────────────────────────
    # STEP 2. Epoch 탐색
    # ────────────────────────────────
    print("\n" + "="*50)
    print("  STEP 2. Epoch 탐색")
    print(f"  고정: LR={best_lr:.0e}, Batch=8, MaxLen=256") # Step 1에서 찾은 best_lr 고정
    print("="*50)

    epoch_results = {}
    for epoch in EPOCH_LIST:
        print(f"\n[Epoch={epoch}] 학습 시작...")
        f1 = train_and_eval(
            tr_texts, val_texts, tr_labels, val_labels,
            lr=best_lr, epochs=epoch, batch_size=8, max_length=256
        )
        epoch_results[epoch] = f1
        print(f"  → 최고 val_F1: {f1:.4f}")

    best_epoch = max(epoch_results, key=epoch_results.get)
    print_results("Epoch", epoch_results, best_epoch)

    # ────────────────────────────────
    # STEP 3. Batch Size 탐색
    # ────────────────────────────────
    print("\n" + "="*50)
    print("  STEP 3. Batch Size 탐색")
    print(f"  고정: LR={best_lr:.0e}, Epoch={best_epoch}, MaxLen=256") # Step 1, 2 결과 고정
    print("="*50)

    batch_results = {}
    for batch in BATCH_LIST:
        print(f"\n[Batch={batch}] 학습 시작...")
        f1 = train_and_eval(
            tr_texts, val_texts, tr_labels, val_labels,
            lr=best_lr, epochs=best_epoch, batch_size=batch, max_length=256
        )
        batch_results[batch] = f1
        print(f"  → 최고 val_F1: {f1:.4f}")

    best_batch = max(batch_results, key=batch_results.get)
    print_results("Batch", batch_results, best_batch)

    # ────────────────────────────────
    # STEP 4. Max Length 탐색
    # ────────────────────────────────
    print("\n" + "="*50)
    print("  STEP 4. Max Length 탐색")
    print(f"  고정: LR={best_lr:.0e}, Epoch={best_epoch}, Batch={best_batch}") # Step 1, 2, 3 결과 고정
    print("="*50)

    maxlen_results = {}
    for maxlen in MAXLEN_LIST:
        print(f"\n[MaxLen={maxlen}] 학습 시작...")
        f1 = train_and_eval(
            tr_texts, val_texts, tr_labels, val_labels,
            lr=best_lr, epochs=best_epoch, batch_size=best_batch, max_length=maxlen
        )
        maxlen_results[maxlen] = f1
        print(f"  → 최고 val_F1: {f1:.4f}")

    best_maxlen = max(maxlen_results, key=maxlen_results.get)
    print_results("MaxLen", maxlen_results, best_maxlen)

    # ────────────────────────────────
    # 최종 요약
    # ────────────────────────────────
    print("\n" + "="*50)
    print("  최종 최적 하이퍼파라미터")
    print("="*50)
    print(f"  Learning Rate : {best_lr:.0e}")
    print(f"  Epochs        : {best_epoch}")
    print(f"  Batch Size    : {best_batch}")
    print(f"  Max Length    : {best_maxlen}")
    print()
    print("  → klue_bert_ft_v5.py에 아래 값을 적용하세요!")
    print(f"  LEARNING_RATE = {best_lr}")
    print(f"  EPOCHS        = {best_epoch}")
    print(f"  BATCH_SIZE    = {best_batch}")
    print(f"  MAX_LENGTH    = {best_maxlen}")