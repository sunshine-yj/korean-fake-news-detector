"""
KPF-BERT + 번역 기반 LUNA 방식 수치 피처 결합 모델
학습: FakeNews_train_v5.xlsx (8,000개)
평가: FakeNews_test_v5.xlsx  (2,000개)

구조:
  KPF-BERT Fine-Tuning → 가짜일 확률
  + 한국어 → 영어 번역 (Helsinki opus-mt)
  + 영어 수치 피처 추출 (LUNA 아이디어 참고)
  → 로지스틱 회귀 메타 분류기

수치 피처 (영어 번역 기반, LUNA 아이디어):
  num_count   : 숫자 개수
  num_mean    : 숫자 평균
  num_max     : 숫자 최댓값
  num_std     : 숫자 표준편차
  has_percent : % / percent 포함 여부
  has_billion : billion 포함 여부
  has_million : million 포함 여부
  num_ratio   : 단어 대비 숫자 비율
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH    = os.path.join(BASE_DIR, "FakeNews_train_v5.xlsx")
TEST_PATH     = os.path.join(BASE_DIR, "FakeNews_test_v5.xlsx")

MODEL_NAME    = "jinmang2/kpfbert"
TRANS_MODEL   = "Helsinki-NLP/opus-mt-ko-en"
MAX_LENGTH    = 512
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
TRANS_BATCH   = 32
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CACHE   = os.path.join(BASE_DIR, "cache_luna_train.npy")
TEST_CACHE    = os.path.join(BASE_DIR, "cache_luna_test.npy")

print(f"[설정] Device    : {DEVICE}")
print(f"[설정] 모델      : {MODEL_NAME}")
print(f"[설정] 번역 모델 : {TRANS_MODEL}")
print(f"[설정] 방식      : KPF-BERT + 번역 + LUNA 방식 수치 피처")


# ────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────
def load_dataset(path: str):
    df           = pd.read_excel(path)
    titles       = df["title_clean"].fillna("").tolist()
    contents     = df["content_clean"].fillna("").tolist()
    texts        = [t + " " + c for t, c in zip(titles, contents)]
    labels       = df["is_fake"].values
    prompt_types = np.array(df["prompt_type"].tolist(), dtype=object)
    print(f"  총 {len(texts)}개  (진짜: {(labels==0).sum()}, 가짜: {(labels==1).sum()})")
    return titles, contents, texts, labels, prompt_types


# ────────────────────────────────────────────────
# 2. 한→영 번역 (Helsinki opus-mt, 캐시 활용)
# ────────────────────────────────────────────────
def translate_to_english(texts: list, cache_path: str) -> list:
    if os.path.exists(cache_path):
        print(f"  번역 캐시 로드: {os.path.basename(cache_path)}")
        return np.load(cache_path, allow_pickle=True).tolist()

    print(f"  Helsinki opus-mt 번역 시작...")
    tokenizer = MarianTokenizer.from_pretrained(TRANS_MODEL)
    model     = MarianMTModel.from_pretrained(TRANS_MODEL).to(DEVICE)
    model.eval()

    translated = []
    for i in tqdm(range(0, len(texts), TRANS_BATCH), desc="  한→영 번역"):
        batch   = [t[:500] for t in texts[i : i + TRANS_BATCH]]
        encoded = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            output = model.generate(**encoded)
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        translated.extend(decoded)

    np.save(cache_path, np.array(translated, dtype=object))
    print(f"  번역 완료: {len(translated)}개")
    return translated


# ────────────────────────────────────────────────
# 3. 영어 수치 피처 추출 (LUNA 아이디어)
# ────────────────────────────────────────────────
def extract_luna_numeric(eng_texts: list) -> np.ndarray:
    features = []
    for text in eng_texts:
        text    = str(text).lower()
        numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]

        num_count   = len(numbers)
        num_mean    = np.mean(numbers) if numbers else 0.0
        num_max     = np.max(numbers)  if numbers else 0.0
        num_std     = np.std(numbers)  if numbers else 0.0
        has_percent = 1.0 if '%' in text or 'percent' in text else 0.0
        has_billion = 1.0 if 'billion' in text else 0.0
        has_million = 1.0 if 'million' in text else 0.0
        num_ratio   = num_count / max(len(text.split()), 1)

        features.append([
            num_count, num_mean, num_max, num_std,
            has_percent, has_billion, has_million, num_ratio
        ])

    X = np.array(features)
    print(f"  수치 피처 shape: {X.shape}")
    return X


# ────────────────────────────────────────────────
# 4. Dataset 클래스
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
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ────────────────────────────────────────────────
# 5. KPF-BERT 학습
# ────────────────────────────────────────────────
def train_and_get_probs(titles, contents, labels):
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    model        = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    train_loader = DataLoader(
        NewsDataset(titles, contents, labels, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
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


def get_probs(model, tokenizer, titles, contents, labels):
    loader = DataLoader(
        NewsDataset(titles, contents, labels, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  확률 추출", leave=False):
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            prob = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(prob)
    return np.array(probs)


# ────────────────────────────────────────────────
# 6. 성능 계산 및 출력
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


# ────────────────────────────────────────────────
# 7. 프롬프트별 성능 분석
# ────────────────────────────────────────────────
def analyze_prompt_results(y_true, y_pred, prompt_types):
    prompt_labels = {
        1.0: "프롬프트 1 (수치변경)",
        2.0: "프롬프트 2 (고유명사)",
        3.0: "프롬프트 3 (출처익명)",
        4.0: "프롬프트 4 (인과왜곡)",
        5.0: "프롬프트 5 (동사반전)",
    }
    results      = {}
    real_idx_all = np.where(pd.isna(prompt_types))[0]

    for pt, label in prompt_labels.items():
        fake_idx = np.where(prompt_types == pt)[0]
        n_fake   = len(fake_idx)
        if n_fake == 0:
            continue
        np.random.seed(42)
        real_idx = np.random.choice(real_idx_all, size=n_fake, replace=False)
        idx      = np.concatenate([fake_idx, real_idx])
        metrics  = calc_metrics(y_true[idx], y_pred[idx])
        results[label] = (n_fake * 2, metrics)

    return results


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1. 데이터 로드
    print("\n" + "="*62)
    print("  STEP 1. 데이터 로드")
    print("="*62)
    print("[Train]")
    tr_titles, tr_contents, tr_texts, train_y, _ = load_dataset(TRAIN_PATH)
    print("[Test]")
    te_titles, te_contents, te_texts, test_y, pt_test = load_dataset(TEST_PATH)

    # STEP 2. 한→영 번역
    print("\n" + "="*62)
    print("  STEP 2. 한→영 번역 (Helsinki opus-mt)")
    print("="*62)
    print("[Train 번역]")
    tr_eng = translate_to_english(tr_texts, TRAIN_CACHE)
    print("[Test 번역]")
    te_eng = translate_to_english(te_texts, TEST_CACHE)

    # STEP 3. 영어 수치 피처 추출 (LUNA 방식)
    print("\n" + "="*62)
    print("  STEP 3. 영어 수치 피처 추출 (LUNA 아이디어)")
    print("="*62)
    print("[Train]")
    tr_numeric = extract_luna_numeric(tr_eng)
    print("[Test]")
    te_numeric = extract_luna_numeric(te_eng)

    scaler     = StandardScaler()
    tr_numeric = scaler.fit_transform(tr_numeric)
    te_numeric = scaler.transform(te_numeric)

    # STEP 4. KPF-BERT 학습
    print("\n" + "="*62)
    print("  STEP 4. KPF-BERT Fine-Tuning")
    print("="*62)
    kpf_model, kpf_tokenizer = train_and_get_probs(tr_titles, tr_contents, train_y)

    # STEP 5. 확률 추출
    print("\n" + "="*62)
    print("  STEP 5. 확률 추출")
    print("="*62)
    print("[Train]")
    tr_probs = get_probs(kpf_model, kpf_tokenizer, tr_titles, tr_contents, train_y)
    print("[Test]")
    te_probs = get_probs(kpf_model, kpf_tokenizer, te_titles, te_contents, test_y)

    # STEP 6. 메타 분류기
    print("\n" + "="*62)
    print("  STEP 6. 메타 분류기 (KPF-BERT 확률 + LUNA 수치 피처)")
    print("="*62)
    X_train = np.column_stack([tr_probs, tr_numeric])
    X_test  = np.column_stack([te_probs, te_numeric])

    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(X_train, train_y)
    test_preds = meta_clf.predict(X_test)

    # 피처 가중치 출력
    feature_names = [
        'kpf_prob', 'num_count', 'num_mean', 'num_max', 'num_std',
        'has_percent', 'has_billion', 'has_million', 'num_ratio'
    ]
    print("\n[피처 가중치]")
    for name, coef in zip(feature_names, meta_clf.coef_[0]):
        print(f"  {name:<15}: {coef:>8.4f}")

    # STEP 7. 성능 평가
    print("\n" + "="*62)
    print("  STEP 7. Test 전체 성능 평가 (2,000개)")
    print("="*62)
    test_metrics = calc_metrics(test_y, test_preds)
    print(f"\n{'지표':<12} {'값':>8}")
    print("-"*25)
    for k, v in test_metrics.items():
        print(f"{k:<12} {v:>8.4f}")

    # STEP 8. 프롬프트별 성능 분석
    print("\n" + "="*74)
    print("  STEP 8. 프롬프트별 성능 분석")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)

    prompt_results = analyze_prompt_results(test_y, test_preds, pt_test)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)

    print("\n" + "="*74)
    print("  최종 요약")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)
    print_metrics("전체 Test (2,000개)", len(test_y), test_metrics)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)
    print("="*74)

    print(f"\n[비교]")
    print(f"  KPF-BERT + LUNA 수치피처 F1   : {test_metrics['F1']:.4f}")
    print(f"  KPF-BERT + 한국어 수치피처 F1 : ???")
    print(f"  KPF-BERT 단독 F1              : ???")
