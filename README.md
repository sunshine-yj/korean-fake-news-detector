# Korean Fake News Detector

> LLM으로 생성된 한국어 가짜뉴스에 대한 기존 탐지 모델의 한계를 검증하고, 이를 극복하는 KPF-BERT 기반 탐지 시스템을 구축한 연구 프로젝트

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![n8n](https://img.shields.io/badge/n8n-workflow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. 연구 배경

기존 한국어 가짜뉴스 탐지 연구는 대부분 특정 시점에 수집된 공개 데이터셋(AI Hub 등)에 의존한다. 그러나 최근 대형 언어모델(LLM)의 발전으로, **문체와 형식이 실제 기사와 구분되지 않으면서 사실관계만 정교하게 조작된** 새로운 유형의 가짜뉴스가 등장했다.

본 연구는 다음 두 가지 질문에서 출발한다.

1. 기존 데이터셋으로 학습된 탐지 모델이 LLM 생성 가짜뉴스에도 유효한가?
2. 유효하지 않다면, 어떤 조작 유형이 특히 탐지하기 어려운가?

## 2. 핵심 결과

기존 연구(KLUE-BERT Voting)를 그대로 재현했을 때 원 논문 데이터셋에서는 **F1 0.9739**를 기록했으나, 동일 모델을 본 연구에서 구축한 LLM 생성 데이터셋에 적용하자 **F1 0.0587**로 붕괴했다.

이는 기존 모델이 "가짜뉴스의 언어적 특징"이 아니라 **특정 데이터셋의 표면적 패턴**을 학습했음을 시사한다.

| 모델 | 학습 데이터 | F1 |
|---|---|---|
| KLUE-BERT Voting (논문 재현, K-Fold) | 원 논문 데이터 | 0.9739 |
| KLUE-BERT Voting (전이 적용) | 원 논문 데이터 → 본 연구 데이터 | **0.0587** |
| KLUE-BERT 파인튜닝 | 본 연구 데이터 | 0.9488 |
| KPF-BERT 파인튜닝 | 본 연구 데이터 | 0.9555 |
| KPF-BERT + 한국어 수치 피처 | 본 연구 데이터 | 0.9569 |
| KPFNumericFusion (End-to-End) | 본 연구 데이터 | **0.9650** |
| 최종 앙상블 (Model A:B = 2:8) | AI Hub + 본 연구 데이터 | 0.9559 |

> 최종 시스템은 단일 최고 F1(KPFNumericFusion) 대신 **앙상블 구성**을 채택했다. 서로 다른 분포의 데이터로 학습한 두 모델을 결합해 미지의 입력에 대한 견고성을 확보하기 위함이며, 가중치 비율은 실험을 통해 2:8로 결정했다.

## 3. 주요 기여

### 3.1 5가지 조작 유형 기반 데이터셋 구축

실제 기사를 원본으로 삼아, LLM에게 **단일 조작 유형만** 적용하도록 통제한 가짜뉴스를 생성했다. 유형이 혼재된 기존 데이터셋과 달리, 유형별 탐지 성능을 독립적으로 측정할 수 있다.

| # | 유형 | 조작 내용 | 선행연구 존재 |
|---|---|---|---|
| 1 | 수치 변경 | 통계·금액·인원 등 수치를 왜곡 | O |
| 2 | 고유명사 치환 | 인물·기관·지역명을 다른 대상으로 교체 | O |
| 3 | 출처 익명화 | 명시적 출처를 익명 표현으로 대체 | **X (본 연구)** |
| 4 | 인과 왜곡 | 사건 간 인과관계를 재구성 | O |
| 5 | 동사 반전 | 서술어의 의미를 반대로 전환 | **X (본 연구)** |

### 3.2 조작 유형별 탐지 난이도 규명

모든 유형이 동일하게 어렵지 않다는 점, 그리고 **그 난이도에 구조적 이유가 있다는 점**이 본 연구의 주요 발견이다.

- **유형 3(출처 익명화)**: 가장 탐지가 쉽다. 익명 표현으로의 전환이 문체적 변화를 남기므로 언어모델이 포착 가능하다.
- **유형 1(수치 변경) · 유형 5(동사 반전)**: 가장 어렵다. 문법적으로 완전하고 문체 변화가 없어, **외부 사실 검증 없이는 원리적으로 판별이 어렵다.**

이 난이도 구배는 텍스트 기반 탐지 모델의 성능 상한을 설명하며, 향후 팩트체킹 모듈 결합의 필요성을 뒷받침한다.

### 3.3 한국어 수치 표현 피처 융합

한국어 텍스트에서 수치 표현을 **번역 없이 직접 추출**해 BERT 표현과 융합하는 구조를 설계했다. 유형 1에 대한 취약점을 보완하기 위한 접근이다.

### 3.4 자동화 데이터 수집·생성 파이프라인

n8n 기반으로 RSS 수집 → 본문 파싱 → 전처리 → DB 적재 → 가짜뉴스 생성까지 전 과정을 자동화했다.

## 4. 시스템 구성

```
[뉴스 수집]                [가짜뉴스 생성]           [탐지 모델]              [서비스]
 n8n RSS/API      →       n8n + GPT-5-mini    →    KPF-BERT 앙상블    →    FastAPI
      ↓                          ↓                       ↑                     ↓
  MySQL (ARTICLES) ────────────────────────────────────────            Chrome Extension
```

### 4.1 데이터 수집 파이프라인

연합뉴스, SBS, 동아일보, 한겨레, JTBC 등의 RSS 피드와 네이버 뉴스 API를 통해 실제 기사를 수집한다.

![RSS 수집 파이프라인](docs/images/pipeline-rss.png)
![API 수집 파이프라인](docs/images/pipeline-api.png)

### 4.2 가짜뉴스 생성 파이프라인

수집된 실제 기사를 원본으로, 5가지 유형 프롬프트를 라운드로빈 방식으로 순환 적용해 균형 잡힌 분포를 유지한다.

![가짜뉴스 생성 파이프라인](docs/images/pipeline-generator.png)

### 4.3 탐지 모델

```
입력 기사
   ├─ Model A (KPF-BERT, AI Hub 학습)        F1 0.9697  ─┐
   │                                                      ├─ 2:8 가중 앙상블 → 판정
   └─ Model B (KPF-BERT, 본 연구 데이터 학습)  F1 0.9575  ─┘
```

### 4.4 서비스 인터페이스

Chrome 확장 프로그램 **'AI 뉴스 정제기'** 를 통해 사용자가 열람 중인 기사를 실시간으로 검증한다. FastAPI 백엔드가 추론을 담당하며, React 기반 프론트엔드에서 결과를 시각화한다.

## 5. 데이터셋

**FakeNews_balanced_v5** — 총 10,000건 (진짜 5,000 / 가짜 5,000)

| 구분 | 건수 |
|---|---|
| Train | 8,000 |
| Test | 2,000 (조작 유형별 200건 × 5유형 + 진짜 1,000건) |

전체 명세, 생성 조건, 라이선스 및 이용 제한은 [`data/README.md`](data/README.md)를 참조.

> ⚠️ **본 리포지토리의 가짜뉴스 데이터는 연구 목적으로 인위 생성된 합성 텍스트입니다.** 실제 사실이 아니며, 어떠한 형태로든 사실 정보로 유통되어서는 안 됩니다.

## 6. 디렉토리 구조

```
.
├── docs/           연구 문서 (데이터셋 설계, 실험 로그, 결과 분석, 트러블슈팅)
├── pipeline/       n8n 워크플로 JSON 및 조작 유형별 프롬프트
├── database/       MySQL 스키마 및 분석 쿼리
├── data/           데이터셋 명세, 샘플, 통계
├── src/            전처리 · 피처 · 모델 · 학습 · 평가 · 앙상블
├── experiments/    실험 설정(YAML) 및 결과 지표
├── serving/        FastAPI 추론 서버
├── extension/      Chrome 확장 프로그램
└── frontend/       React 프론트엔드
```

## 7. 시작하기

### 7.1 환경 설정

```bash
git clone https://github.com/sunshine-yj/korean-fake-news-detector.git
cd korean-fake-news-detector

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env    # DB 접속 정보 및 API 키 입력
```

### 7.2 데이터베이스 초기화

```bash
mysql -u root -p < database/schema.sql
mysql -u root -p fake_news_db < database/seed_sources.sql
```

### 7.3 n8n 파이프라인 실행

n8n 관리 화면에서 `pipeline/workflows/` 내 JSON을 Import한 뒤, Credential을 직접 등록한다. (보안상 리포지토리에는 포함하지 않음)

### 7.4 학습

```bash
python -m src.train.finetune --config experiments/configs/kpfbert_base.yaml
python -m src.train.optuna_search --config experiments/configs/optuna.yaml
```

### 7.5 평가 및 추론

```bash
python -m src.eval.evaluate --checkpoint <path> --by-type
uvicorn serving.api.main:app --reload
```

## 8. 실험 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 5070 |
| 프레임워크 | PyTorch, HuggingFace Transformers |
| 하이퍼파라미터 탐색 | Optuna |
| 데이터베이스 | MySQL 8.0 |
| 워크플로 자동화 | n8n (Docker) |
| 생성 모델 | GPT-5-mini |

## 9. 참고문헌

- 이상민 외, "(제목)", (학회명), (권), (호), pp. (면), (날짜).
- 고상훈, 안현철, "(제목)", (학회명), (권), (호), pp. (면), 2024.
- 박성수, 이건창, "(제목)", (학회명), (권), (호), pp. (면), 2018.
- MegaFake: A Theory-Driven Dataset of Fake News Generated by Large Language Models, 2024.
- Satapara et al., (제목), 2024.

## 10. 라이선스 및 이용 안내

- **코드**: MIT License
- **데이터**: 비상업적 연구 목적에 한해 이용 가능. 수집된 기사 원문은 각 언론사에 저작권이 있으며 본 리포지토리에는 포함되지 않는다.

## 11. 인용

```bibtex
@misc{korean_fake_news_detector_2026,
  title  = {LLM 생성 한국어 가짜뉴스 탐지를 위한 조작 유형별 데이터셋 및 KPF-BERT 기반 탐지 시스템},
  author = {(성명)},
  school = {선문대학교},
  year   = {2026},
  note   = {https://github.com/sunshine-yj/korean-fake-news-detector}
}
```
