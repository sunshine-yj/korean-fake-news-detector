# 실험

본 연구에서 수행한 모델 실험의 목적, 설정, 결과를 정리한다.

```
experiments/
├── klue_bert_hparam_search.py    하이퍼파라미터 탐색
├── klue_bert_voting.py           논문 재현 베이스라인 (전이 실패 검증)
├── kpf_bert_baseline.py          KPF-BERT 단독
├── kpf_bert_numeric_fusion.py    KPF-BERT + 한국어 수치 피처 (단일 최고 F1)
└── kpf_bert_luna.py              KPF-BERT + 번역 기반 수치 피처

../src/train/
├── train_aihub.py                Model A (최종 앙상블 구성)
└── train_custom.py               Model B (최종 앙상블 구성)
```

---

## 1. 결과 요약

| # | 실험 | 학습 데이터 | F1 | 비고 |
|---|---|---|---:|---|
| 1 | KLUE-BERT Voting (K-Fold) | 원 논문 데이터 | 0.9739 | 재현 성공 |
| 2 | KLUE-BERT Voting (전이) | 원 논문 데이터 → 본 연구 데이터 | **0.0587** | **핵심 발견** |
| 3 | KLUE-BERT 파인튜닝 | 본 연구 데이터 | 0.9488 | |
| 4 | KPF-BERT 파인튜닝 | 본 연구 데이터 | 0.9555 | |
| 5 | KPF-BERT + 수치 피처 | 본 연구 데이터 | 0.9569 | |
| 6 | KPFNumericFusion | 본 연구 데이터 | **0.9650** | 단일 최고 |
| 7 | Model A (KPF-BERT) | AI Hub | 0.9697 ※ | 앙상블 구성 |
| 8 | Model B (KPF-BERT) | 본 연구 데이터 | 0.9575 | 앙상블 구성 |
| 9 | 최종 앙상블 (A:B = 2:8) | — | 0.9559 | 서비스 적용 |

> ※ **지표 불일치 주의.** 실험 7(`train_aihub.py`)은 `average='macro'`로 F1을 계산하고, 나머지 실험은 이진 F1(양성 클래스 기준)을 사용한다. 또한 실험 7의 평가 데이터는 AI Hub 자체 테스트 분할이므로, 다른 실험(본 연구 테스트셋 2,000건)과 직접 비교할 수 없다.

### 최종 모델 선택

단일 최고 F1은 KPFNumericFusion(0.9650)이나, **최종 시스템은 앙상블(0.9559)을 채택**했다. 서로 다른 분포의 데이터로 학습한 두 모델을 결합해 미지의 입력에 대한 견고성을 확보하기 위함이다. 서비스 환경에서는 학습 데이터와 다른 성격의 기사가 입력될 수 있으므로, 테스트셋 F1 최대화보다 분포 외 입력에 대한 안정성을 우선했다.

---

## 2. 핵심 실험 — 전이 실패 검증

`klue_bert_voting.py`

본 연구의 중심 근거가 되는 실험이다. 선행연구의 KLUE-BERT + Voting 앙상블 구조를 재현한 뒤, 동일 모델을 본 연구에서 구축한 LLM 생성 데이터셋에 적용했다.

### 구조

```
텍스트 → KLUE-BERT [CLS] 임베딩 (768차원)
           ↓
    Soft Voting 앙상블
    ├─ SVM (RBF)
    ├─ XGBoost
    ├─ LightGBM
    ├─ CatBoost
    ├─ Random Forest
    └─ Extra Trees
           ↓
        판정
```

### 결과

| 조건 | F1 |
|---|---:|
| 원 논문 데이터 K-Fold 검증 | 0.9739 |
| 본 연구 데이터 적용 | **0.0587** |

재현 자체는 성공했으나, 데이터셋이 바뀌자 성능이 사실상 무작위 이하로 붕괴했다. 이는 기존 모델이 "가짜뉴스의 언어적 특징"이 아니라 **특정 데이터셋의 표면적 패턴**을 학습했음을 시사한다.

BERT를 고정된 특징 추출기로만 사용하고 분류는 트리 기반 모델에 맡기는 구조는, 임베딩 분포가 달라질 때 대응할 수단이 없다. 파인튜닝 방식(실험 3, F1 0.9488)이 동일 데이터에서 정상 동작한다는 점이 이를 뒷받침한다.

---

## 3. 실험별 상세

### 3.1 klue_bert_hparam_search.py — 하이퍼파라미터 탐색

**그리디 순차 탐색.** 베이지안 최적화나 그리드 서치가 아니라, 한 번에 하나의 하이퍼파라미터만 탐색하고 최적값을 고정한 뒤 다음으로 넘어가는 방식이다.

```
STEP 1. LR      탐색  (Batch=8, Epoch=3, MaxLen=256 고정)
STEP 2. Epoch   탐색  (최적 LR 적용)
STEP 3. Batch   탐색  (최적 LR·Epoch 적용)
STEP 4. MaxLen  탐색  (최적 LR·Epoch·Batch 적용)
```

| 파라미터 | 탐색 범위 |
|---|---|
| Learning Rate | 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 |
| Epoch | 3, 4, 5 |
| Batch Size | 8, 16 |
| Max Length | 128, 256, 384, 512 |

전체 조합(120가지)을 모두 시도하면 단일 GPU에서 현실적이지 않아, 순차 탐색으로 4+3+2+4=13회 학습으로 축소했다. 하이퍼파라미터 간 상호작용을 포착하지 못하는 한계가 있으나, 자원 제약 하에서의 실용적 선택이다.

Early Stopping(patience=2)과 Train/Val 85:15 분할을 적용했다.

### 3.2 klue_bert_voting.py — 논문 재현

2절 참조. 임베딩 추출 결과를 `.npy`로 캐싱해 반복 실행 시간을 단축했다.

### 3.3 kpf_bert_baseline.py — KPF-BERT 단독

한국언론진흥재단이 뉴스 코퍼스로 사전학습한 KPF-BERT(`jinmang2/kpfbert`)를 파인튜닝한다. 제목과 본문을 `[SEP]`로 구분해 문장 쌍으로 입력한다.

```python
tokenizer(titles, contents, ...)   # sentence pair 입력
```

| 설정 | 값 |
|---|---|
| Max Length | 512 |
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 1e-5 |
| Warmup Ratio | 0.1 |

> **주의:** 본 스크립트는 `FakeNews_train_merged.xlsx`를 사용한다. 다른 실험이 쓰는 `FakeNews_train_v5.xlsx`와의 관계 확인 필요.

### 3.4 kpf_bert_numeric_fusion.py — KPFNumericFusion

단일 실험 중 최고 F1(0.9650)을 기록했다. 유형 1(수치 변경)에 대한 취약점을 보완하기 위해, 한국어 텍스트에서 수치 표현을 직접 추출해 BERT 표현과 융합한다.

#### 구조

```
제목 [SEP] 본문 ──→ KPF-BERT ──→ [CLS] (768)
                                          ├─→ concat (800) ─→ 분류기 ─→ 판정
수치 피처 (7) ──→ Linear(32)+ReLU ────────┘
```

#### 추출 피처 (7차원)

| 피처 | 설명 |
|---|---|
| `num_count` | 숫자 개수 |
| `num_mean` | 숫자 평균 |
| `num_max` | 숫자 최댓값 |
| `num_std` | 숫자 표준편차 |
| `has_percent` | `%` 포함 여부 |
| `has_money` | 억·만·천·원 단위 포함 여부 |
| `num_ratio` | 단어 대비 숫자 비율 |

`StandardScaler`로 정규화해 수치 크기가 학습을 지배하지 않도록 했다.

`AutoModelForSequenceClassification` 대신 `AutoModel`을 사용해 분류 헤드를 직접 구성한 것이 특징이다. 학습된 스케일러는 추론 시에도 필요하므로 모델과 함께 저장한다.

### 3.5 kpf_bert_luna.py — 번역 기반 수치 피처

LUNA(영어권 연구)의 수치 피처 접근을 한국어에 적용하기 위해, 본문을 영어로 번역(`Helsinki-NLP/opus-mt-ko-en`)한 뒤 피처를 추출하는 방식이다.

#### 3.4와의 차이

| | 3.4 (한국어 직접) | 3.5 (번역 기반) |
|---|---|---|
| 전처리 | 없음 | 한→영 번역 |
| 화폐 표현 | 억, 만, 천, 원 | billion, million |
| 결합 방식 | 신경망 계층 융합 | 로지스틱 회귀 메타 분류기 |

번역 단계에서 본문을 500자로 절단하므로 정보 손실이 발생하며, 번역 오류가 수치 표현을 왜곡할 수 있다. **한국어 표현을 직접 다루는 3.4가 더 나은 성능을 보였고, 이는 번역 경유가 불필요함을 보여준다.**

메타 분류기의 피처 가중치를 출력하도록 구현되어 있어, 어떤 수치 피처가 판별에 기여했는지 확인할 수 있다.

---

## 4. 최종 모델 (../src/train/)

### 4.1 train_aihub.py — Model A

AI Hub 낚시성 기사 데이터셋으로 KPF-BERT를 파인튜닝한다. JSON 파일을 직접 파싱하며, 7:1.5:1.5로 train/val/test를 분할한다.

```python
'label': int(j['sourceDataInfo']['useType'])   # 1=진짜, 0=가짜
```

> **라벨 방향 주의.** Model A는 **0이 가짜**다. Model B(1이 가짜)와 반대이므로, 앙상블 시 서로 다른 인덱스에서 확률을 추출해야 한다. `../serving/README.md` 참조.

| 설정 | 값 |
|---|---|
| Learning Rate | 1e-5 |
| Weight Decay | 0.01 |
| Scheduler | linear |
| Epochs / Batch | 3 / 16 |
| 평가 | epoch마다 (`load_best_model_at_end`) |

AI Hub 데이터는 라이선스상 본 저장소에 포함하지 않는다. [AI Hub](https://aihub.or.kr/)에서 직접 내려받아 `FakeNews_origin_dataset2/` 폴더에 JSON 파일을 배치해야 한다.

### 4.2 train_custom.py — Model B

본 연구 데이터셋(`FakeNews_train_v5.xlsx`, 8,000건)으로 KPF-BERT를 파인튜닝한다.

| 설정 | 값 | Model A와 차이 |
|---|---|---|
| Learning Rate | 2e-5 | 1e-5 |
| Weight Decay | 0.1 | 0.01 |
| Scheduler | cosine | linear |
| 검증 | 없음 | epoch마다 |

검증 분할 없이 전체 데이터로 학습한다. 앙상블 구성 요소로서 데이터를 최대한 활용하기 위한 선택이며, 하이퍼파라미터는 별도 실험에서 확정한 값을 적용했다.

Weight Decay를 0.1로 높인 것은 표면적 패턴에 대한 과적합을 억제하기 위함이다. 2절의 전이 실패가 바로 그 과적합의 결과이므로, 동일한 실패를 반복하지 않기 위한 조치다.

---

## 5. 실험 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 5070 |
| 프레임워크 | PyTorch, HuggingFace Transformers |
| 부스팅 모델 | XGBoost, LightGBM, CatBoost |
| 데이터 | `FakeNews_train_v5.xlsx` (8,000) / `FakeNews_test_v5.xlsx` (2,000) |

---

## 6. 평가 방법

### 6.1 유형별 성능 분석

테스트셋의 유형별 가짜 기사(각 200건)에 **동일 개수의 진짜 기사를 무작위 추출해 1:1로 구성**한 뒤 지표를 계산한다. 클래스 불균형에 따른 지표 왜곡을 막기 위함이다.

```python
np.random.seed(42)
real_idx = np.random.choice(real_idx_all, size=n_fake, replace=False)
idx = np.concatenate([fake_idx, real_idx])
```

**한계:** 모든 실험이 동일한 시드를 사용하므로 유형 간 비교는 공정하나, 유형별 절대 성능은 추출된 특정 진짜 기사 200건에 의존한다. 다른 시드에서는 수치가 달라질 수 있다.

### 6.2 지표

정확도, 정밀도, 재현율, F1을 산출한다. 양성 클래스는 가짜(1)이며, `train_aihub.py`만 macro 평균을 사용한다(1절 각주 참조).

---

## 7. 알려진 한계

**① 하이퍼파라미터 상호작용 미탐색**
순차 탐색 방식이므로 파라미터 간 상호작용을 포착하지 못한다. 전체 조합 탐색 시 다른 최적점이 존재할 수 있다.

**② 평가 지표 불일치**
`train_aihub.py`가 macro F1, 나머지가 binary F1을 사용해 1절 표의 직접 비교에 제약이 있다.

**③ 유형별 분석의 시드 의존성**
6.1 참조.

**④ 학습 데이터 파일 불일치**
`kpf_bert_baseline.py`가 `_merged`, 나머지가 `_v5` 파일을 사용한다. 두 파일의 관계 확인 필요.

**⑤ 앙상블 가중치 탐색 근거 미기록**
2:8 비율의 결정 과정(탐색 범위, 각 비율별 성능)이 코드에 남아 있지 않다.

**⑥ 단일 시드 실험**
각 실험을 단일 시드로 1회 실행했다. 성능 차이가 통계적으로 유의한지 검증하려면 복수 시드 반복이 필요하다. 특히 실험 4~6의 F1 차이(0.9555 / 0.9569 / 0.9650)는 시드 변동 범위 내일 가능성을 배제할 수 없다.
