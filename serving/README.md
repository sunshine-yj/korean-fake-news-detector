# 서빙

Flask 기반 추론 API. 두 개의 KPF-BERT 모델을 가중 앙상블해 가짜뉴스 여부를 판정한다.

크롬 확장 프로그램 'AI 뉴스 정제기'가 n8n 워크플로를 경유해 이 API를 호출한다.

---

## 1. 앙상블 구조

```
제목 + 본문
   │
   ├─→ Model B (본 연구 데이터 학습) ──→ P(가짜) × 0.8 ─┐
   │                                                      ├─→ 합산 ─→ ≥0.5 → 가짜
   └─→ Model A (AI Hub 학습) ─────────→ P(가짜) × 0.2 ─┘
```

가중치는 실험을 통해 A:B = 2:8로 결정했다. 본 연구 데이터로 학습한 Model B에 더 큰 비중을 두되, 다른 분포로 학습한 Model A를 보조적으로 결합해 미지의 입력에 대한 견고성을 확보한다.

---

## 2. 라벨 방향 — 수정 시 반드시 확인

**두 모델의 라벨 의미가 반대다.**

| 모델 | 학습 데이터 | 가짜 = 클래스 | 확률 추출 |
|---|---|---:|---|
| Model A | AI Hub (`useType`: 1=진짜) | **0** | `probs[0][0]` |
| Model B | 본 연구 (`is_fake`: 1=가짜) | **1** | `probs[0][1]` |

```python
def get_aihub_prob(title, content):
    ...
    return probs[0][0].item()   # Model A: 0번이 가짜

def get_custom_prob(title, content):
    ...
    return probs[0][1].item()   # Model B: 1번이 가짜
```

인덱스를 통일하면 두 모델이 서로 상쇄되어 앙상블이 조용히 무력화된다. 오류 메시지가 나지 않고 성능만 떨어지므로 발견이 어렵다. 코드 수정 시 이 부분을 먼저 확인할 것.

---

## 3. 실행

### 3.1 모델 준비

학습된 모델 두 개가 필요하다. 저장소에는 가중치를 포함하지 않으므로 직접 학습하거나 별도 배포본을 내려받아야 한다.

```bash
python ../src/train/train_aihub.py     # → ./model_aihub
python ../src/train/train_custom.py    # → ./model_custom
```

### 3.2 환경변수

```bash
cp ../.env.example .env
```

```
MODEL_A_PATH=./model_aihub
MODEL_B_PATH=./model_custom
ENSEMBLE_WEIGHT_A=0.2
ENSEMBLE_WEIGHT_B=0.8
API_PORT=5000
```

### 3.3 서버 실행

```bash
pip install -r ../requirements.txt
python app.py
```

기본 포트는 5000이다. n8n의 `extension-inference` 워크플로가 `http://127.0.0.1:5000/predict`를 호출하므로 포트를 바꾸면 워크플로도 함께 수정해야 한다.

---

## 4. API

### POST /predict

**요청**

```json
{
  "title": "기사 제목",
  "content": "기사 본문"
}
```

**응답**

```json
{
  "predicted_class": 1,
  "label": "가짜뉴스",
  "confidence": 0.0312,
  "custom_model_prob": 0.9821,
  "aihub_model_prob": 0.9435
}
```

| 필드 | 설명 |
|---|---|
| `predicted_class` | 0=진짜, 1=가짜 |
| `label` | 판정 결과 문자열 |
| `confidence` | **진짜뉴스 확률** (`1 - 앙상블 가짜 확률`) |
| `custom_model_prob` | Model B의 가짜 확률 |
| `aihub_model_prob` | Model A의 가짜 확률 |

> ⚠️ **`confidence` 필드명 주의.** 이 값은 일반적인 의미의 "판정 확신도"가 아니라 **진짜뉴스일 확률**이다. 가짜로 판정된 경우 값이 0에 가깝게 나온다. 클라이언트에서 확신도로 표시하면 "가짜뉴스인데 확신도 3%"처럼 오해를 유발할 수 있다.
>
> 클라이언트가 확신도를 필요로 한다면 다음과 같이 수정한다.
>
> ```python
> "confidence": round(ensemble_prob if predicted_class == 1 else real_prob, 4)
> ```

### GET /health

```json
{ "status": "ok" }
```

---

## 5. 전처리 위치

이 API는 **정제된 텍스트를 입력으로 가정한다.** HTML 파싱, 광고·바이라인 제거 등의 전처리는 n8n의 `extension-inference` 워크플로가 담당한다.

```
크롬 확장 → n8n Webhook
              ├─ HTML 다운로드
              ├─ 범용 셀렉터 추출
              ├─ 1차 정규식 전처리
              └─ 2차 LLM 정제 (Gemini)
                    ↓
              Flask /predict  ← 여기부터
```

상세는 [`../pipeline/README.md`](../pipeline/README.md) 5절 참조.

---

## 6. 알려진 한계

**① 개발 서버로 실행**
`app.run(debug=True)`는 Flask 내장 개발 서버다. 연구용 로컬 실행을 전제로 하며, 다중 요청 처리나 보안이 필요한 환경에는 적합하지 않다. 배포 시 gunicorn 등 WSGI 서버를 사용해야 한다.

**② CORS 전체 허용**
`CORS(app)`으로 모든 출처를 허용한다. 크롬 확장에서의 호출을 위한 설정이나, 외부 노출 시 출처를 제한해야 한다.

**③ 입력 길이 제한**
`max_length=512` 토큰을 초과하는 본문은 절단된다. 긴 기사의 후반부는 판정에 반영되지 않는다.

**④ 모델 상시 메모리 적재**
두 모델을 시작 시 로드해 상주시킨다. 응답 속도에는 유리하나 메모리 사용량이 크다.

**⑤ 배치 처리 미지원**
요청당 1건만 처리한다.
