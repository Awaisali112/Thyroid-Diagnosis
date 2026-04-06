"""
ThyroScan KBS – Flask Backend (3-class: Negative / Hypothyroid / Hyperthyroid)
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, os, numpy as np

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
CORS(app)

BASE   = os.path.dirname(__file__)
MODEL  = joblib.load(os.path.join(BASE, "models/thyroid_model.pkl"))
SCALER = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
LE     = joblib.load(os.path.join(BASE, "models/label_encoder.pkl"))

with open(os.path.join(BASE, "models/model_meta.json")) as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]
KB            = META["knowledge_base"]
CLASS_MAP     = META["class_mapping"]   # {"0":"Negative","1":"Hypothyroid","2":"Hyperthyroid"}

NORMAL_RANGES = {
    "TSH": (0.35, 4.5,  "mIU/L"),
    "T3":  (1.2,  3.1,  "nmol/L"),
    "TT4": (60,   140,  "ng/dL"),
    "T4U": (0.8,  1.8,  "ratio"),
    "FTI": (65,   155,  "index"),
}

def flag_lab(name, value):
    if name not in NORMAL_RANGES:
        return None
    lo, hi, unit = NORMAL_RANGES[name]
    status = "LOW" if value < lo else "HIGH" if value > hi else "NORMAL"
    return {"status": status, "normal": f"{lo}–{hi} {unit}"}


def apply_rules(data):
    rules = []
    tsh = data.get("TSH", 0)
    t3  = data.get("T3",  0)
    tt4 = data.get("TT4", 0)

    if tsh > 4.5:
        rules.append(f"⚠ TSH elevated ({tsh:.2f} mIU/L > 4.5) — hypothyroid indicator")
    elif tsh < 0.35:
        rules.append(f"⚠ TSH suppressed ({tsh:.2f} mIU/L < 0.35) — hyperthyroid indicator")
    else:
        rules.append(f"✅ TSH normal ({tsh:.2f} mIU/L, range 0.35–4.5)")

    if t3 < 1.2:
        rules.append(f"⚠ T3 low ({t3:.2f} nmol/L < 1.2) — hypothyroid indicator")
    elif t3 > 3.1:
        rules.append(f"⚠ T3 elevated ({t3:.2f} nmol/L > 3.1) — hyperthyroid indicator")

    if tt4 < 60:
        rules.append(f"⚠ TT4 low ({tt4:.1f} ng/dL < 60) — hypothyroid indicator")
    elif tt4 > 140:
        rules.append(f"⚠ TT4 elevated ({tt4:.1f} ng/dL > 140) — hyperthyroid indicator")

    if data.get("pregnant", 0) == 1:
        rules.append("⚠ Pregnancy — thyroid requirements increase significantly")

    return rules


def get_recommendations(prediction):
    recs = {
        "Negative": [
            "No thyroid intervention currently indicated",
            "Routine annual screening if risk factors are present",
            "Maintain healthy iodine intake through diet",
            "Report new symptoms (fatigue, weight change) to your doctor",
        ],
        "Hypothyroid": [
            "Consult an endocrinologist promptly for confirmatory workup",
            "Levothyroxine replacement therapy is the standard treatment",
            "Repeat thyroid function tests 6–8 weeks after starting treatment",
            "Monitor TSH every 6–12 months once stable",
            "Check for associated autoimmune conditions (TPO antibodies)",
        ],
        "Borderline": [
            "Results are inconclusive — consult a doctor for a full thyroid evaluation",
            "Repeat thyroid function tests (TSH, Free T4, T3) are recommended",
            "Multiple abnormal lab values detected despite low model confidence",
            "Do not self-medicate; a physician must interpret these results",
        ],
        "Hyperthyroid": [
            "Urgent endocrinology referral recommended",
            "Consider anti-thyroid medications (methimazole or PTU)",
            "Radioactive iodine therapy or surgery may be indicated",
            "Beta-blockers for symptomatic relief of palpitations/tremors",
            "Thyroid uptake scan to identify the underlying cause",
        ],
    }
    return recs.get(prediction, [])


@app.route("/favicon.svg")
def favicon():
    return send_from_directory(os.path.join(BASE, "static"), "favicon.svg", mimetype="image/svg+xml")


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_accuracy": META["accuracy"],
        "classes": META["classes"],
        "n_samples": META.get("n_samples"),
    })


@app.route("/api/meta")
def meta():
    return jsonify({
        "accuracy":           META["accuracy"],
        "cv_mean":            META["cv_mean"],
        "cv_std":             META["cv_std"],
        "classes":            META["classes"],
        "feature_names":      FEATURE_NAMES,
        "feature_importance": META["feature_importance"],
        "n_samples":          META.get("n_samples"),
        "datasets":           META.get("datasets", []),
    })


@app.route("/api/kb")
def kb():
    return jsonify(KB)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    missing = [f for f in FEATURE_NAMES if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = [float(body[f]) for f in FEATURE_NAMES]
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    X        = np.array(values).reshape(1, -1)
    X_scaled = SCALER.transform(X)
    raw_pred  = int(MODEL.predict(X_scaled)[0])
    proba     = MODEL.predict_proba(X_scaled)[0]

    prediction = CLASS_MAP[str(raw_pred)]
    confidence = round(float(max(proba)) * 100, 1)

    prob_dict = {}
    for cls_idx, cls_int in enumerate(MODEL.classes_):
        label = CLASS_MAP[str(int(cls_int))]
        prob_dict[label] = round(float(proba[cls_idx]) * 100, 1)

    data_  = dict(zip(FEATURE_NAMES, values))
    rules  = apply_rules(data_)
    flags  = {k: flag_lab(k, data_[k]) for k in NORMAL_RANGES if k in data_}

    # Borderline safety override:
    # If the model is uncertain (< 65% confidence) AND the rule engine
    # flagged 2 or more abnormal values, the case is too risky to call
    # "Negative" — escalate to a borderline warning instead.
    abnormal_rules = [r for r in rules if "⚠" in r]
    if confidence < 65 and len(abnormal_rules) >= 2:
        prediction = "Borderline"
        rules.append("⚠ Low model confidence + multiple abnormal values — borderline result, further evaluation recommended")

    recs   = get_recommendations(prediction)

    return jsonify({
        "prediction":      prediction,
        "confidence":      confidence,
        "probabilities":   prob_dict,
        "rules_triggered": rules,
        "recommendations": recs,
        "lab_flags":       flags,
        "kb_info":         KB.get(prediction, {}),
        "input_summary":   data_,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
