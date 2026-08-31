"""
KPF-BERT + 한국어 수치 피처 퓨전(Fusion) 딥러닝 모델 학습 및 평가
텍스트 임베딩(BERT)과 수치형 통계 데이터를 딥러닝 계층(Linear Layer)에서 병합하여
하나의 통합된 분류기로 가짜 뉴스를 판별하는 스크립트입니다.
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
# 1. 설정 (Configuration)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "FakeNews_train_v5.xlsx")
TEST_PATH  = os.path.join(BASE_DIR, "FakeNews_test_v5.xlsx")

# 이번에는 AutoModelForSequenceClassification 대신 순수 AutoModel을 불러와서
# 우리가 직접 분류기(Classifier) 헤드를 만들 것입니다.
MODEL_NAME = "jinmang2/kpfbert"

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] DEVICE = {DEVICE}")
print(f"[INFO] MODEL  = {MODEL_NAME}")


# =========================================================
# 2. 데이터 로드 (Data Loading)
# =========================================================
def load_dataset(path):
    """데이터를 로드하고 텍스트(제목, 본문)와 라벨을 반환합니다."""
    df = pd.read_excel(path)

    titles = df["title_clean"].fillna("")
    contents = df["content_clean"].fillna("")
    
    # 텍스트 통계를 뽑기 위해 임시로 제목과 본문을 공백으로 이어붙임
    texts = (titles + " " + contents).tolist()

    labels = df["is_fake"].astype(int).values
    prompt_types = np.array(df["prompt_type"].values, dtype=object)

    return titles.tolist(), contents.tolist(), texts, labels, prompt_types


# =========================================================
# 3. 수치 피처 추출 (Numeric Feature Engineering)
# =========================================================
def extract_numeric_features(texts):
    """텍스트 내에서 기사 특유의 통계적/수치적 특성을 7차원 벡터로 추출합니다."""
    features = []

    for text in texts:
        text = str(text)

        # 실수형태(예: 12, 3.14) 숫자를 모두 추출
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]

        num_count = len(numbers)
        num_mean  = np.mean(numbers) if numbers else 0
        num_max   = np.max(numbers) if numbers else 0
        num_std   = np.std(numbers) if numbers else 0

        # % 기호 및 한국어 특정 화폐/단위 표현 포함 여부
        has_percent = 1 if "%" in text else 0
        has_money   = 1 if re.search(r"\d+(억|만|천|원)", text) else 0
        
        # 기사 전체 단어 중 숫자가 차지하는 비율
        num_ratio   = num_count / max(len(text.split()), 1)

        features.append([
            num_count,
            num_mean,
            num_max,
            num_std,
            has_percent,
            has_money,
            num_ratio
        ])

    return np.array(features, dtype=np.float32)


# =========================================================
# 4. Dataset 클래스 (PyTorch Custom Dataset)
# =========================================================
class NewsDataset(Dataset):
    """텍스트 토큰과 수치형 피처를 동시에 모델에 공급하기 위한 커스텀 데이터셋"""
    def __init__(self, titles, contents, numeric, labels, tokenizer):

        # 제목과 본문을 [SEP] (Separation) 토큰으로 명확하게 구분하여 연결
        texts = [t + " [SEP] " + c for t, c in zip(titles, contents)]

        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt" # PyTorch Tensor 타입으로 반환
        )

        # 텍스트와 함께 수치형 피처(7차원 벡터)도 텐서로 변환하여 보관
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "numeric": self.numeric[idx],
            "labels": self.labels[idx]
        }


# =========================================================
# 5. Fusion Model (핵심: 텍스트 + 수치형 데이터 융합 신경망)
# =========================================================
class KPFNumericFusion(nn.Module):
    """
    KPF-BERT의 텍스트 특징 벡터(768차원)와 수치형 데이터(7차원)를 병합하여
    최종 가짜 뉴스 여부를 판별하는 멀티모달 딥러닝 모델
    """
    def __init__(self, model_name, num_dim):
        super().__init__()

        # 1. 텍스트 인코더: BERT 기본 모델 (분류기 헤드가 없는 뼈대 모델)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size # 기본 BERT는 보통 768차원

        # 2. 수치 데이터 인코더: 7차원 숫자를 32차원으로 맵핑(확장)하는 미니 신경망
        self.num_net = nn.Sequential(
            nn.Linear(num_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. 융합 분류기: 텍스트(768) + 수치(32) = 800차원을 입력받아 2개(진짜/가짜)로 출력
        self.classifier = nn.Sequential(
            nn.Linear(hidden + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, numeric, labels=None):
        # [Step A] 텍스트 데이터를 BERT에 통과시킴
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 문장 전체의 문맥을 요약하는 첫 번째 토큰([CLS])의 벡터를 추출 (형태: Batch_size x 768)
        cls = outputs.last_hidden_state[:, 0]
        
        # [Step B] 수치형 데이터를 미니 신경망에 통과시킴 (형태: Batch_size x 32)
        num = self.num_net(numeric)

        # [Step C] 두 벡터를 가로(dim=1)로 이어 붙임 (형태: Batch_size x 800)
        x = torch.cat([cls, num], dim=1)

        # [Step D] 이어 붙인 벡터를 최종 분류기에 통과시켜 예측값(Logits) 도출
        logits = self.classifier(x)

        loss = None
        # 학습 시 정답(labels)이 제공되면 크로스 엔트로피 오차 계산
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, logits


# =========================================================
# 6. 학습 (Training Loop)
# =========================================================
def train(model, loader):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = len(loader) * EPOCHS

    # 학습 초반 학습률을 서서히 올리고(Warmup) 이후 선형으로 감소시키는 스케줄러
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(loader):
            optimizer.zero_grad() # 기울기 초기화

            # 커스텀 모델의 forward 메서드 호출
            loss, _ = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["numeric"].to(DEVICE),
                batch["labels"].to(DEVICE)
            )

            loss.backward()  # 역전파
            optimizer.step() # 가중치 업데이트
            scheduler.step() # 학습률 업데이트

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")


# =========================================================
# 7. 평가 및 Metric 계산 (Evaluation)
# =========================================================
def predict(model, loader):
    model.eval() # 평가 모드 (Dropout 등 비활성화)
    y_true, y_pred = [], []

    with torch.no_grad(): # 평가 시에는 기울기 계산을 하지 않아 메모리 절약
        for batch in tqdm(loader, desc="  예측", leave=False):
            _, logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["numeric"].to(DEVICE),
                batch["labels"].to(DEVICE)
            )
            # Logits에서 가장 값이 큰 인덱스를 예측 클래스로 추출
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].cpu().numpy())

    return np.array(y_true), np.array(y_pred)

def calc_metrics(y_true, y_pred):
    """정밀도, 재현율 등 평가 지표 계산 (0으로 나누는 에러 방지 처리)"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def print_metrics(label, n, m):
    print(f"{label:<22} {n:>6}  {m['Accuracy']:>9.4f}  {m['Precision']:>9.4f}  {m['Recall']:>9.4f}  {m['F1']:>9.4f}")


# =========================================================
# 8. 프롬프트별 평가 (버그 수정 반영)
# =========================================================
def analyze_prompt(y_true, y_pred, prompt_types):
    """생성 방식별 모델 성능 분석 (진짜 뉴스와 가짜 뉴스를 1:1로 추출해 공정하게 평가)"""
    prompt_labels = {
        1.0: "프롬프트 1 (수치변경)",
        2.0: "프롬프트 2 (고유명사)",
        3.0: "프롬프트 3 (출처익명)",
        4.0: "프롬프트 4 (인과왜곡)",
        5.0: "프롬프트 5 (동사반전)",
    }
    results = {}
    np.random.seed(42)

    # 진짜 뉴스 전체 인덱스 확보 (결측치 또는 0)
    real_idx_all = np.where(pd.isna(prompt_types) | (prompt_types == 0))[0]

    for pt, label in prompt_labels.items():
        fake_idx = np.where(prompt_types == pt)[0]
        n_fake = len(fake_idx)

        if n_fake == 0:
            continue

        # 가짜 뉴스 개수만큼 진짜 뉴스 추출 (1:1 매칭)
        real_idx = np.random.choice(real_idx_all, size=n_fake, replace=False)
        idx = np.concatenate([fake_idx, real_idx])

        metrics = calc_metrics(y_true[idx], y_pred[idx])
        results[label] = (n_fake * 2, metrics)

    return results


# =========================================================
# 9. MAIN (Execution)
# =========================================================
if __name__ == "__main__":

    print("\n" + "="*62)
    print("  STEP 1. 데이터 및 수치 피처 로드")
    print("="*62)
    tr_t, tr_c, tr_texts, tr_y, _ = load_dataset(TRAIN_PATH)
    te_t, te_c, te_texts, te_y, pt = load_dataset(TEST_PATH)

    tr_num = extract_numeric_features(tr_texts)
    te_num = extract_numeric_features(te_texts)

    # 모델이 숫자의 크기에 압도되지 않도록 평균 0, 분산 1로 스케일링
    scaler = StandardScaler()
    tr_num = scaler.fit_transform(tr_num)
    te_num = scaler.transform(te_num)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = NewsDataset(tr_t, tr_c, tr_num, tr_y, tokenizer)
    test_ds  = NewsDataset(te_t, te_c, te_num, te_y, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("\n" + "="*62)
    print("  STEP 2. KPF-BERT Fusion 딥러닝 모델 학습")
    print("="*62)
    # 모델 인스턴스화 (입력되는 수치형 데이터가 7개이므로 num_dim=7)
    model = KPFNumericFusion(MODEL_NAME, num_dim=7)
    train(model, train_loader)

    print("\n" + "="*62)
    print("  STEP 3. Test 전체 성능 평가 (2,000개)")
    print("="*62)
    y_true, y_pred = predict(model, test_loader)
    test_metrics = calc_metrics(y_true, y_pred)
    
    print(f"\n{'지표':<12} {'값':>8}")
    print("-"*25)
    for k, v in test_metrics.items():
        print(f"{k:<12} {v:>8.4f}")

    print("\n" + "="*74)
    print("  STEP 4. 프롬프트별 성능 분석 (1:1 클래스 균형)")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)

    prompt_results = analyze_prompt(y_true, y_pred, pt)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)

    print("\n" + "="*74)
    print("  최종 요약")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)
    print_metrics("전체 Test (2,000개)", len(te_y), test_metrics)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)
    print("="*74)

    print(f"\n[비교]")
    print(f"  퓨전 딥러닝 (KPF+수치) Test F1: {test_metrics['F1']:.4f}")

    # =========================================================
    # 모델 및 스케일러 저장 추가
    # =========================================================
    import joblib
    SAVE_DIR = os.path.join(BASE_DIR, "saved_fusion_model")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 커스텀 딥러닝 모델이므로 state_dict(학습된 가중치 딕셔너리)를 직접 저장
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fusion_model.pt"))
    tokenizer.save_pretrained(SAVE_DIR)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    print(f"\n✅ 학습된 퓨전 모델과 스케일러가 [{SAVE_DIR}] 폴더에 안전하게 저장되었습니다!")