"""
KLUE-BERT + Voting 우리 데이터 학습 및 평가
학습: FakeNews_train_v5.xlsx (8,000개)
평가: FakeNews_test_v5.xlsx  (2,000개) - 프롬프트별 균등 200개

실험 흐름:
  STEP 1. Train 데이터 로드 및 KLUE-BERT 임베딩 (캐싱 적용으로 속도 향상)
  STEP 2. K-Fold(k=5) 교차 검증 (모델의 일반화 성능 확인)
  STEP 3. 전체 Train으로 최종 모델 학습 (앙상블 모델)
  STEP 4. Test 전체 성능 평가
  STEP 5. Test 안에서 프롬프트별 성능 분석
"""

import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# 강력한 머신러닝 모델 및 앙상블 기법을 위한 scikit-learn 라이브러리들
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 트리 기반의 강력한 부스팅 모델들
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ────────────────────────────────────────────────
# 설정 (Configuration)
# ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH   = os.path.join(BASE_DIR, "FakeNews_train_v5.xlsx")
TEST_PATH    = os.path.join(BASE_DIR, "FakeNews_test_v5.xlsx")

MODEL_NAME   = "klue/bert-base"
MAX_LENGTH   = 256  # BERT 토크나이저 최대 길이
BATCH_SIZE   = 16   # 임베딩 추출 시 메모리 초과를 막기 위한 배치 사이즈
K_FOLDS      = 5    # 교차 검증 분할 수
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# 임베딩 추출은 시간이 매우 오래 걸리므로, 한 번 추출한 결과(.npy)를 저장해두는 경로
TRAIN_CACHE  = os.path.join(BASE_DIR, "cache_v5_train.npy")
TEST_CACHE   = os.path.join(BASE_DIR, "cache_v5_test.npy")
# 학습된 최종 머신러닝 앙상블 모델을 저장할 경로
SAVE_PATH = os.path.join(BASE_DIR, "saved_model/klue_bert_voting_v5.pkl")

print(f"[설정] Device : {DEVICE}")
print(f"[설정] 모델   : {MODEL_NAME}")
print(f"[설정] 학습   : FakeNews_train_v5.xlsx (8,000개)")
print(f"[설정] 평가   : FakeNews_test_v5.xlsx  (2,000개)")


# ────────────────────────────────────────────────
# 1. 데이터 로드 (Data Loading)
# ────────────────────────────────────────────────
def load_dataset(path: str):
    """엑셀 파일에서 데이터를 읽어와 제목과 본문을 하나의 텍스트로 합칩니다."""
    df           = pd.read_excel(path)
    # 결측치(NaN)를 빈 문자열로 채우고, 제목과 본문 사이에 공백을 넣어 병합
    texts        = (df["title_clean"].fillna("") + " " + df["content_clean"].fillna("")).tolist()
    labels       = df["is_fake"].values
    prompt_types = np.array(df["prompt_type"].tolist(), dtype=object)
    
    print(f"  총 {len(texts)}개  (진짜: {(labels==0).sum()}, 가짜: {(labels==1).sum()})")
    return texts, labels, prompt_types


# ────────────────────────────────────────────────
# 2. KLUE-BERT 임베딩 추출 (Feature Extraction)
# ────────────────────────────────────────────────
def extract_embeddings(texts: list, cache_path: str) -> np.ndarray:
    """
    KLUE-BERT를 통과시켜 각 문장을 대표하는 고차원 벡터(임베딩)를 추출합니다.
    매번 딥러닝 연산을 하면 너무 느리므로, 캐시(cache) 파일이 있으면 바로 로드합니다.
    """
    # 이미 추출해 둔 캐시 파일(.npy)이 존재하면 연산 생략 후 즉시 로드
    if os.path.exists(cache_path):
        print(f"  캐시 로드: {os.path.basename(cache_path)}")
        return np.load(cache_path)

    # 텍스트를 벡터로 바꾸기 위해 사전 학습된 모델과 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() # 추론 모드 (Dropout 적용 안 함)

    embeddings = []
    # 데이터를 BATCH_SIZE만큼 쪼개서 처리 (GPU 메모리 보호)
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  임베딩 추출"):
        batch   = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        
        with torch.no_grad(): # 역전파(기울기 계산)를 하지 않아 속도/메모리 최적화
            out = model(**encoded)
            
        # out.last_hidden_state 형태: [Batch_size, Seq_length, Hidden_size(768)]
        # [:, 0, :] -> 문장 전체의 의미를 요약하는 첫 번째 토큰([CLS] 토큰)의 벡터만 추출
        embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    # 리스트에 담긴 배치별 결과물을 세로로 쌓아서 하나의 큰 행렬(N x 768)로 만듦
    X = np.vstack(embeddings)
    # 다음 실행 시 시간을 절약하기 위해 넘파이 배열 캐시로 저장
    np.save(cache_path, X)
    print(f"  임베딩 완료: shape={X.shape}")
    return X


# ────────────────────────────────────────────────
# 3. Voting 분류기 (Ensemble Classifier)
# ────────────────────────────────────────────────
def build_voting_classifier():
    """
    서로 다른 원리를 가진 6개의 강력한 머신러닝 모델을 결합합니다.
    voting="soft"는 각 모델이 예측한 '가짜 뉴스일 확률'을 평균 내어 최종 결과를 정하는 방식입니다.
    """
    return VotingClassifier(estimators=[
        # 1. 서포트 벡터 머신 (SVM)
        ("svm",        SVC(kernel="rbf", probability=True, C=1.0, random_state=42)),
        # 2. XGBoost (대표적인 트리 부스팅 알고리즘)
        ("xgb",        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     eval_metric="logloss", random_state=42, verbosity=0)),
        # 3. LightGBM (속도가 빠르고 성능이 좋은 트리 부스팅)
        ("lgbm",       LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       random_state=42, verbose=-1)),
        # 4. CatBoost (범주형 데이터 처리에 강하고 과적합을 잘 방지하는 부스팅)
        ("catboost",   CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                           random_state=42, verbose=0)),
        # 5. Random Forest (여러 개의 결정 트리를 만드는 배깅 알고리즘)
        ("rf",         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        # 6. Extra Trees (랜덤 포레스트보다 더 무작위성을 주어 과적합을 방지)
        ("extratrees", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ], voting="soft", n_jobs=1)


# ────────────────────────────────────────────────
# 4. 성능 계산 및 출력 (Metrics Calculation)
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
# 5. 프롬프트별 성능 분석 (Prompt-Specific Analysis)
# ────────────────────────────────────────────────
def analyze_prompt_results(y_true, y_pred, prompt_types):
    """각종 조작 기법(프롬프트)에 대해 모델이 얼마나 잘 방어하는지 1:1 비율로 섞어 확인합니다."""
    prompt_labels = {
        1.0: "프롬프트 1 (수치변경)",
        2.0: "프롬프트 2 (고유명사)",
        3.0: "프롬프트 3 (출처익명)",
        4.0: "프롬프트 4 (인과왜곡)",
        5.0: "프롬프트 5 (동사반전)",
    }
    results      = {}
    # 결측치(NaN)를 '진짜 뉴스'로 간주하여 인덱스 확보
    real_idx_all = np.where(pd.isna(prompt_types))[0]

    for pt, label in prompt_labels.items():
        fake_idx = np.where(prompt_types == pt)[0]
        n_fake   = len(fake_idx)
        if n_fake == 0:
            continue
            
        # 정확도 왜곡을 막기 위해 가짜 뉴스 개수만큼의 진짜 뉴스를 랜덤하게 뽑아 1:1 세트 구성
        np.random.seed(42)
        real_idx = np.random.choice(real_idx_all, size=n_fake, replace=False)
        idx      = np.concatenate([fake_idx, real_idx])
        
        metrics  = calc_metrics(y_true[idx], y_pred[idx])
        results[label] = (n_fake * 2, metrics)

    return results


# ────────────────────────────────────────────────
# 메인 (Pipeline Execution)
# ────────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1. 데이터 로드 및 임베딩
    print("\n" + "="*60)
    print("  STEP 1. 데이터 로드 및 KLUE-BERT 임베딩")
    print("="*60)
    print("[Train]")
    train_texts, train_y, _ = load_dataset(TRAIN_PATH)
    # BERT를 통과한 768차원의 Train 피처 벡터 
    X_train = extract_embeddings(train_texts, TRAIN_CACHE)

    print("[Test]")
    test_texts, test_y, pt_test = load_dataset(TEST_PATH)
    # BERT를 통과한 768차원의 Test 피처 벡터 
    X_test = extract_embeddings(test_texts, TEST_CACHE)

    # STEP 2. K-Fold 교차 검증
    # 모델이 특정 데이터 셋에만 우연히 성능이 좋은 것인지(과적합) 검증하기 위해 5번 쪼개서 평가
    print("\n" + "="*60)
    print("  STEP 2. K-Fold 교차 검증 (Train 내부)")
    print("="*60)
    skf        = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    clf_kfold  = build_voting_classifier()
    
    # K-Fold 진행 및 성능 지표 계산
    cv_results = cross_validate(
        clf_kfold, X_train, train_y,
        cv=skf,
        scoring=["accuracy", "precision", "recall", "f1"],
        verbose=1, # 진행 상황 표시
    )
    print(f"\n[K-Fold 검증 성능]")
    # 5번의 테스트 결과의 평균(mean)과 표준편차(std) 출력
    print(f"  Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"  Recall   : {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
    print(f"  F1-score : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

    # STEP 3. 전체 Train으로 최종 모델 학습
    print("\n" + "="*60)
    print("  STEP 3. 전체 Train으로 최종 모델 학습")
    print("="*60)
    if os.path.exists(SAVE_PATH):
        # 학습에 시간이 걸리므로 이미 저장된 파일이 있으면 불러옵니다.
        print("[로드] 저장된 모델 불러오는 중...")
        clf = joblib.load(SAVE_PATH)
        print("[로드] 완료!")
    else:
        # 앙상블 모델 초기화 후 8000개 Train 데이터 전체를 사용해 최종 학습
        clf = build_voting_classifier()
        print("[학습] 최종 모델 학습 중...")
        clf.fit(X_train, train_y)
        print("[학습] 완료!")
        
        # 모델 저장 (추후 재사용 가능하도록 .pkl 파일로 직렬화)
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        joblib.dump(clf, SAVE_PATH)
        print(f"[저장] 완료: {SAVE_PATH}")

    # STEP 4. Test 전체 성능 평가
    print("\n" + "="*60)
    print("  STEP 4. Test 전체 성능 평가 (2,000개)")
    print("="*60)
    
    # 평가 데이터를 모델에 넣어 최종 가짜뉴스 여부 추론
    test_preds   = clf.predict(X_test)
    test_metrics = calc_metrics(test_y, test_preds)

    print(f"\n{'지표':<12} {'값':>8}")
    print("-"*25)
    for k, v in test_metrics.items():
        print(f"{k:<12} {v:>8.4f}")

    # STEP 5. 프롬프트별 성능 분석
    print("\n" + "="*74)
    print("  STEP 5. 프롬프트별 성능 분석 (Test 2,000개 기준)")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)

    prompt_results = analyze_prompt_results(test_y, test_preds, pt_test)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)

    # 최종 요약 (전체 결과 통합 출력)
    print("\n" + "="*74)
    print("  최종 요약")
    print("="*74)
    print(f"{'유형':<22} {'샘플수':>6}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print("-"*74)
    print_metrics("전체 Test", len(test_y), test_metrics)
    for label, (n, metrics) in prompt_results.items():
        print_metrics(label, n, metrics)
    print("="*74)

    # 교차 검증 시의 성능과 최종 평가 셋에서의 성능을 비교해 과적합 여부 파악
    print("\n[비교]")
    print(f"  K-Fold 검증 F1               : {cv_results['test_f1'].mean():.4f}")
    print(f"  우리 데이터 학습 → Test F1   : {test_metrics['F1']:.4f}")