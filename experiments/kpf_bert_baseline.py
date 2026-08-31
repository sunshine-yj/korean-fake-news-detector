import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH    = os.path.join(BASE_DIR, "FakeNews_train_merged.xlsx")
TEST_PATH     = os.path.join(BASE_DIR, "FakeNews_test_merged.xlsx")

MODEL_NAME    = "jinmang2/kpfbert"
MAX_LENGTH    = 512
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 1e-5
WARMUP_RATIO  = 0.1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[설정] Device : {DEVICE}")
print(f"[설정] 모델   : {MODEL_NAME}")
print(f"[설정] 방식   : KPF-BERT 단독 (수치 피처 없음) + AI Hub 데이터")


# ────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────
def load_dataset(path: str):
    df           = pd.read_excel(path)
    titles       = df["title_clean"].fillna("").tolist()
    contents     = df["content_clean"].fillna("").tolist()
    labels       = df["is_fake"].values
    prompt_types = np.array(df["prompt_type"].tolist(), dtype=object)
    print(f"  총 {len(labels)}개  (진짜: {(labels==0).sum()}, 가짜: {(labels==1).sum()})")
    return titles, contents, labels, prompt_types


# ────────────────────────────────────────────────
# 2. Dataset 클래스 (학습용)
# ────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, titles, contents, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            titles, contents,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels"        : self.labels[idx],
        }


# ────────────────────────────────────────────────
# 2-1. Dataset 클래스 (추론용)
# ────────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, titles, contents):
        self.titles   = titles
        self.contents = contents

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx], self.contents[idx]


# ────────────────────────────────────────────────
# 3. KPF-BERT 학습
# ────────────────────────────────────────────────
def train_model(titles, contents, labels):
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    model        = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    train_loader = DataLoader(
        NewsDataset(titles, contents, labels, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch}  loss={total_loss/len(train_loader):.4f}")

    return model, tokenizer


# ────────────────────────────────────────────────
# 4. 예측 함수
# ────────────────────────────────────────────────
def predict(model, tokenizer, titles, contents):
    dataset    = InferenceDataset(titles, contents)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_titles, batch_contents in tqdm(dataloader, desc="  예측", leave=False):
            inputs = tokenizer(
                list(batch_titles), list(batch_contents),
                padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            )
            out = model(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=inputs["attention_mask"].to(DEVICE),
            )
            preds.extend(out.logits.argmax(dim=-1).cpu().numpy())

    return np.array(preds)


# ────────────────────────────────────────────────
# 5. 성능 계산
# ────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    return {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

def print_metrics(label, n, m):
    print(f"{label:<22} {n:>6}  {m['Accuracy']:>9.4f}  {m['Precision']:>9.4f}  {m['Recall']:>9.4f}  {m['F1']:>9.4f}")


# # ────────────────────────────────────────────────
# # 6. 프롬프트별 성능 분석
# # ────────────────────────────────────────────────
# def analyze_prompt_results(y_true, y_pred, prompt_types):
#     prompt_labels = {
#         1.0: "프롬프트 1 (수치변경)",
#         2.0: "프롬프트 2 (고유명사)",
#         3.0: "프롬프트 3 (출처익명)",
#         4.0: "프롬프트 4 (인과왜곡)",
#         5.0: "프롬프트 5 (동사반전)",
#     }
#     results      = {}
#     real_idx_all = np.where(pd.isna(prompt_types))[0]

#     for pt, label in prompt_labels.items():
#         fake_idx = np.where(prompt_types == pt)[0]
#         n_fake   = len(fake_idx)
#         if n_fake == 0:
#             continue
#         np.random.seed(42)
#         real_idx = np.random.choice(real_idx_all, size=n_fake, replace=False)
#         idx      = np.concatenate([fake_idx, real_idx])
#         metrics  = calc_metrics(y_true[idx], y_pred[idx])
#         results[label] = (n_fake * 2, metrics)

#     return results


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1. 데이터 로드
    print("\n" + "="*62)
    print("  STEP 1. 데이터 로드")
    print("="*62)
    print("[Train]")
    tr_titles, tr_contents, train_y, _ = load_dataset(TRAIN_PATH)
    print("[Test]")
    te_titles, te_contents, test_y, pt_test = load_dataset(TEST_PATH)

    # STEP 2. KPF-BERT 학습
    print("\n" + "="*62)
    print("  STEP 2. KPF-BERT Fine-Tuning")
    print("="*62)
    kpf_model, kpf_tokenizer = train_model(tr_titles, tr_contents, train_y)

    # STEP 3. 예측
    print("\n" + "="*62)
    print("  STEP 3. Test 예측")
    print("="*62)
    test_preds = predict(kpf_model, kpf_tokenizer, te_titles, te_contents)

    # STEP 4. 성능 평가
    print("\n" + "="*62)
    print("  STEP 4. Test 전체 성능 평가")
    print("="*62)
    test_metrics = calc_metrics(test_y, test_preds)
    print(f"\n{'지표':<12} {'값':>8}")
    print("-"*25)
    for k, v in test_metrics.items():
        print(f"{k:<12} {v:>8.4f}")

    # # STEP 5. 프롬프트별 성능 분석
    # print("\n" + "="*74)
    # print("  STEP 5. 프롬프트별 성능 분석")
    # print("="*74)
    # print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    # print("-"*74)

    # prompt_results = analyze_prompt_results(test_y, test_preds, pt_test)
    # for label, (n, metrics) in prompt_results.items():
    #     print_metrics(label, n, metrics)

    # # 최종 요약
    # print("\n" + "="*74)
    # print("  최종 요약")
    # print("="*74)
    # print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    # print("-"*74)
    # print_metrics(f"전체 Test ({len(test_y)}개)", len(test_y), test_metrics)
    # for label, (n, metrics) in prompt_results.items():
    #     print_metrics(label, n, metrics)
    # print("="*74)

    print(f"\n[결과]")
    print(f"  KPF-BERT 단독 F1: {test_metrics['F1']:.4f}")

    # STEP 6. 모델 저장
    print("\n" + "="*74)
    print("  STEP 6. 모델 저장")
    print("="*74)

    SAVE_DIR = os.path.join(BASE_DIR, "saved_kpf_aihub_model")
    os.makedirs(SAVE_DIR, exist_ok=True)

    kpf_model.save_pretrained(SAVE_DIR)
    kpf_tokenizer.save_pretrained(SAVE_DIR)
    print(f"  [완료] KPF-BERT 모델 & 토크나이저 저장 -> {SAVE_DIR}")