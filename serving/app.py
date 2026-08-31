"""
가짜뉴스 판별 추론 API (Flask)

KPF-BERT 기반 두 모델을 가중 앙상블하여 기사의 진위를 판정한다.
크롬 확장 프로그램 'AI 뉴스 정제기'가 n8n 워크플로를 경유해 호출한다.

라벨 방향 주의:
    두 모델은 학습 데이터의 라벨 규약이 반대다.

        Model A (AI Hub, useType)  : 0번 클래스 = 가짜
        Model B (본 연구, is_fake)  : 1번 클래스 = 가짜

    따라서 확률 추출 인덱스가 서로 다르다. 인덱스를 통일하면 두 모델이
    상쇄되어 앙상블이 무력화되며, 오류 없이 성능만 떨어지므로 발견이
    어렵다. 수정 시 README.md 2절을 먼저 확인할 것.

실행:
    python app.py                       # 기본 포트 5000

모델 준비:
    학습된 모델 두 개가 필요하다. 저장소에는 가중치를 포함하지 않는다.

        python ../src/train/train_aihub.py     # -> model_aihub
        python ../src/train/train_custom.py    # -> model_custom

    기본값은 이 파일과 같은 위치의 model_aihub / model_custom 이며,
    환경변수로 다른 경로를 지정할 수 있다.
"""

import os

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
CORS(app)


# ------------------------------------------------
# 설정
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model B - 본 연구 데이터 학습 (1번 클래스 = 가짜)
CUSTOM_MODEL_DIR = os.getenv(
    "MODEL_B_PATH", os.path.join(BASE_DIR, "model_custom")
)
# Model A - AI Hub 데이터 학습 (0번 클래스 = 가짜)
AIHUB_MODEL_DIR = os.getenv(
    "MODEL_A_PATH", os.path.join(BASE_DIR, "model_aihub")
)

# 앙상블 가중치 (실험을 통해 A:B = 2:8 로 결정)
W_CUSTOM = float(os.getenv("ENSEMBLE_WEIGHT_B", 0.8))
W_AIHUB  = float(os.getenv("ENSEMBLE_WEIGHT_A", 0.2))

MAX_LENGTH = 512
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", 5000))
DEBUG      = os.getenv("FLASK_DEBUG", "false").lower() == "true"


# ------------------------------------------------
# 모델 로드
# ------------------------------------------------
def load_model(model_dir: str, label: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"{label} 모델을 찾을 수 없습니다: {model_dir}\n"
            f"학습을 먼저 수행하거나, 환경변수로 경로를 지정하세요."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print(f"  [완료] {label}  ({model_dir})")
    return tokenizer, model


print("모델 로딩 중...")
custom_tokenizer, custom_model = load_model(CUSTOM_MODEL_DIR, "Model B (본 연구 데이터)")
aihub_tokenizer,  aihub_model  = load_model(AIHUB_MODEL_DIR,  "Model A (AI Hub)")
print(f"모델 로딩 완료  (앙상블 가중치 A:B = {W_AIHUB} : {W_CUSTOM})")


# ------------------------------------------------
# 추론
# ------------------------------------------------
def _predict_proba(tokenizer, model, title: str, content: str, fake_index: int) -> float:
    """지정한 인덱스의 클래스 확률을 반환한다."""
    inputs = tokenizer(
        title, content,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.nn.functional.softmax(logits, dim=-1)
    return probs[0][fake_index].item()


def get_custom_prob(title: str, content: str) -> float:
    """Model B - 1번 클래스가 가짜뉴스."""
    return _predict_proba(custom_tokenizer, custom_model, title, content, fake_index=1)


def get_aihub_prob(title: str, content: str) -> float:
    """Model A - 0번 클래스가 가짜뉴스 (useType 1=진짜)."""
    return _predict_proba(aihub_tokenizer, aihub_model, title, content, fake_index=0)


# ------------------------------------------------
# 엔드포인트
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({"error": "JSON 파싱 실패"}), 400

    title   = data.get("title", "")
    content = data.get("content", "")

    if not title and not content:
        return jsonify({"error": "제목 또는 본문을 입력해주세요."}), 400

    try:
        custom_prob = get_custom_prob(title, content)
        aihub_prob  = get_aihub_prob(title, content)

        # 가중 평균
        fake_prob       = (custom_prob * W_CUSTOM) + (aihub_prob * W_AIHUB)
        real_prob       = 1 - fake_prob
        predicted_class = 1 if fake_prob >= 0.5 else 0
        label           = "가짜뉴스" if predicted_class == 1 else "진짜뉴스"

        return jsonify({
            "predicted_class"  : predicted_class,
            "label"            : label,
            # 판정된 클래스에 대한 확신도
            "confidence"       : round(fake_prob if predicted_class == 1 else real_prob, 4),
            # 개별 확률 (클라이언트에서 필요 시 사용)
            "fake_prob"        : round(fake_prob, 4),
            "real_prob"        : round(real_prob, 4),
            "custom_model_prob": round(custom_prob, 4),
            "aihub_model_prob" : round(aihub_prob, 4),
        })

    except Exception as e:
        print(f"[오류] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "weights": {"model_a": W_AIHUB, "model_b": W_CUSTOM},
    })


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)