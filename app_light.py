# app_light.py
import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_ROOT = PROJECT_ROOT / "models" / "light"
SUPPORTED = {"english", "hindi", "gujarati"}
DEFAULT_LANGUAGE = "english"
MAX_TEXT_LEN = 20000

app = Flask(__name__)
CORS(app)

_LOADED = {}

def light_model_files(language: str):
    base = MODELS_ROOT / f"{language}_light_model"
    return {
        "base": base,
        "classifier": base / "classifier.joblib",
        "tfidf": base / "tfidf_vectorizer.joblib",
        "metrics": base / "metrics.json",
        "metadata": base / "metadata.json",
        "logfile": base / "prediction_log.txt",
        "fallback": base / "model.joblib",
    }

class PipelineWrapper:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X)
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.vstack([1.0 - probs_pos, probs_pos]).T
        preds = self.classifier.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[np.arange(len(preds)), preds.astype(int)] = 1.0
        return probs

    def decision_function(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.decision_function(X)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

def _load_json(p: Path):
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

def load_light_model(language: str):
    language = language.lower()
    if language in _LOADED:
        return _LOADED[language]

    files = light_model_files(language)
    clf_p = files["classifier"]
    tfidf_p = files["tfidf"]
    metrics_p = files["metrics"]
    meta_p = files["metadata"]
    fallback_p = files["fallback"]

    if clf_p.exists() and tfidf_p.exists():
        vec = joblib.load(str(tfidf_p))
        clf = joblib.load(str(clf_p))
        pipeline = PipelineWrapper(vec, clf)
        meta = _load_json(meta_p) or {}
        metrics = _load_json(metrics_p) or {}
        threshold = float(meta.get("threshold", 0.5))
        info = {
            "pipeline": pipeline,
            "threshold": threshold,
            "meta": meta,
            "metrics": metrics,
            "path": str(clf_p),
            "files": files,
        }
        _LOADED[language] = info
        return info

    if fallback_p.exists():
        obj = joblib.load(str(fallback_p))
        if isinstance(obj, dict) and "pipeline" in obj:
            pipeline = obj["pipeline"]
            threshold = float(obj.get("threshold", 0.5))
            meta = obj.get("meta", {})
            metrics = obj.get("metrics", {})
        else:
            pipeline = obj
            threshold = 0.5
            meta = {}
            metrics = {}
        info = {"pipeline": pipeline, "threshold": threshold, "meta": meta, "metrics": metrics, "path": str(fallback_p), "files": files}
        _LOADED[language] = info
        return info

    raise FileNotFoundError(f"Light model for '{language}' not found under: {files['base']}")

_re_devanagari = re.compile(r"[\u0900-\u097F]")
_re_gujarati = re.compile(r"[\u0A80-\u0AFF]")

def detect_language_from_text(text: str) -> str:
    if text is None:
        return DEFAULT_LANGUAGE
    s = str(text)
    if _re_gujarati.search(s):
        return "gujarati"
    if _re_devanagari.search(s):
        return "hindi"
    return "english"

def _get_test_accuracy(metrics):
    if isinstance(metrics, dict):
        for k in ("test", "test_accuracy", "test_acc", "accuracy", "val_accuracy"):
            if k in metrics:
                v = metrics[k]
                if isinstance(v, dict) and "accuracy" in v:
                    try:
                        return float(v["accuracy"])
                    except Exception:
                        pass
                else:
                    try:
                        return float(v)
                    except Exception:
                        pass
    return None

def _log_prediction(lang, prob_real, prob_fake, label_real, label_fake, model_path, acc):
    now = datetime.utcnow().isoformat() + "Z"
    final_label = 1 if label_real == 1 else 0
    label_text = "REAL" if final_label == 1 else "FAKE"

    sep = "=" * 60
    print(sep, flush=True)
    print(f"[{now}] Prediction | {lang.upper()} | model: {model_path}", flush=True)
    print(f"→ Probability (Real): {prob_real:.3f} | (Fake): {prob_fake:.3f}", flush=True)
    print(f"→ Model predicted: {final_label} ({label_text})", flush=True)
    print(f"→ Accuracy: {acc if acc is not None else 'N/A'}", flush=True)
    print(sep + "\n", flush=True)
    sys.stdout.flush()

    try:
        files = light_model_files(lang)
        logpath = files["logfile"]
        logpath.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": now,
            "lang": lang,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "predicted_label": final_label,
            "predicted_text": label_text,
            "accuracy": acc,
            "model": model_path
        }
        with open(logpath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write log file: {e}", flush=True)

def predict_with_light(language: str, text: str):
    info = load_light_model(language)
    pipeline = info["pipeline"]
    threshold = float(info.get("threshold", 0.5))
    metrics = info.get("metrics", {})
    acc = _get_test_accuracy(metrics)

    if not isinstance(text, str):
        text = str(text)

    try:
        probs = pipeline.predict_proba([text])[0]
        prob_real = float(probs[1])
    except Exception:
        try:
            score = pipeline.decision_function([text])[0]
            prob_real = float(1.0 / (1.0 + np.exp(-score)))
        except Exception:
            pred = pipeline.predict([text])[0]
            prob_real = float(pred)

    prob_fake = 1.0 - prob_real
    label_real = int(prob_real >= threshold)
    label_fake = 1 - label_real

    _log_prediction(language, prob_real, prob_fake, label_real, label_fake, info.get("path"), acc)

    return {
        "language_used": language,
        "probability_real": prob_real,
        "probability_fake": prob_fake,
        "label_real": label_real,
        "label_fake": label_fake,
        "threshold": threshold,
        "model_path": info.get("path"),
        "test_accuracy": acc,
        "meta": info.get("meta", {}),
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded_models": list(_LOADED.keys()), "models_root": str(MODELS_ROOT)})

@app.route("/models", methods=["GET"])
def list_models():
    out = {}
    for lang in sorted(list(SUPPORTED)):
        base = MODELS_ROOT / f"{lang}_light_model"
        out[lang] = {"exists": base.exists(), "path": str(base)}
        if base.exists():
            out[lang]["classifier"] = str(base / "classifier.joblib") if (base / "classifier.joblib").exists() else None
            out[lang]["tfidf"] = str(base / "tfidf_vectorizer.joblib") if (base / "tfidf_vectorizer.joblib").exists() else None
            out[lang]["metrics"] = str(base / "metrics.json") if (base / "metrics.json").exists() else None
            out[lang]["metadata"] = str(base / "metadata.json") if (base / "metadata.json").exists() else None
    return jsonify(out)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid or empty json body"}), 400

    text = data.get("text") or data.get("texts") or ""
    if not text or not str(text).strip():
        return jsonify({"error": "provide non-empty 'text' in request body"}), 400
    if len(text) > MAX_TEXT_LEN:
        return jsonify({"error": f"text too long (max {MAX_TEXT_LEN} characters)"}), 400

    lang = detect_language_from_text(text)
    if lang not in SUPPORTED:
        lang = DEFAULT_LANGUAGE

    try:
        res = predict_with_light(lang, text)
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500
    except Exception as e:
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500

    # Map model prediction to frontend format
    # label_real == 1 means model says REAL
    if res["label_real"] == 1:
        result_text = "REAL NEWS"
    else:
        result_text = "FAKE NEWS"

    return jsonify({"result": result_text})

@app.route("/reload", methods=["POST"])
def reload_models():
    _LOADED.clear()
    loaded, errors = [], {}
    for lang in SUPPORTED:
        try:
            load_light_model(lang)
            loaded.append(lang)
        except Exception as e:
            errors[lang] = str(e)
    return jsonify({"reloaded": loaded, "errors": errors})

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    app.run(host="0.0.0.0", port=5000, debug=False)
