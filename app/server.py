# app/server.py
from __future__ import annotations
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from pathlib import Path

from src.config import load_config
from src.inference.serve import Predictor

app = Flask(__name__, static_folder="static", template_folder="templates")

CFG = load_config()
PREDICTOR = Predictor(CFG)

# --- Basic CORS (no external deps) ---
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_dir": str(Path(CFG["paths"]["model_dir"]) / CFG["serve"]["model_subdir"]),
        "max_length": CFG["serve"]["max_length"],
        "thresholds": {
            "general": CFG["serve"]["general_threshold"],
            "per_class": CFG["serve"]["per_class_thresholds"]
        }
    }), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    # Accept single object or list
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        return jsonify({"error": "Payload must be object or array"}), 400

    # Optional batch limit
    max_batch = int(CFG["serve"]["max_batch_size"])
    if len(items) > max_batch:
        return jsonify({"error": f"Batch too large. Max {max_batch}."}), 400

    # minimal schema: text required
    for i, it in enumerate(items):
        if not isinstance(it, dict) or "text" not in it:
            return jsonify({"error": f"Item {i} missing 'text'."}), 400

    results = PREDICTOR.predict_payload(items)
    # Output friendly response (omit internal fields you don't need)
    return jsonify({
        "results": [{
            "text": r["text"],
            "language": r["language"],
            "sentiment": r["pred"],
            "confidence": r["confidence"],
            "probs": {
                "negative": r["p_negative"],
                "neutral": r["p_neutral"],
                "positive": r["p_positive"]
            },
            "low_confidence": r["low_confidence"],
            "advisory": r["advisory"]
        } for r in results],
        "disclaimer": (
            "This model performs best on positive reviews. "
            "Neutral/negative predictions may be lower confidence due to class imbalance. "
            "Low-confidence responses include an advisory for human review."
        )
    }), 200

@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Feedback payload must be an object"}), 400

    for field in ["text", "language", "pred", "confidence", "true_label"]:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Log feedback (for future re-training)
    PREDICTOR.log_feedback({
        "ts": data.get("ts"),
        "text": data["text"],
        "language": data["language"],
        "pred": data["pred"],
        "confidence": data["confidence"],
        "true_label": data["true_label"],
        "notes": data.get("notes", "")
    })
    return jsonify({"status": "ok"}), 200

# --- Minimal single-file UI for quick local testing ---
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Restaurant Sentiment Demo</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; line-height: 1.4; }
.card { border: 1px solid #e3e3e3; border-radius: 12px; padding: 16px; margin-top: 16px; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #f1f1f1; margin-left: 6px; }
.low { background: #fff3cd; } .ok { background: #e2f0d9; }
label { display:block; font-weight:600; margin-top:12px;}
textarea { width: 100%; min-height: 100px; }
input[type=text] { width: 200px; }
button { padding: 10px 16px; border-radius: 8px; border: 1px solid #ddd; background: #fafafa; cursor:pointer; }
pre { white-space: pre-wrap; word-wrap: break-word; }
</style>
</head>
<body>
  <h2>Multilingual Restaurant Review Sentiment</h2>
  <p>This demo exposes the model via a REST API. It also flags low-confidence cases and lets you submit feedback.</p>
  <div class="card">
    <label>Language (en/id, optional):</label>
    <input id="lang" type="text" placeholder="en or id" />
    <label>Review text:</label>
    <textarea id="text" placeholder="Type a restaurant review..."></textarea>
    <div style="margin-top:10px;">
      <button onclick="sendPredict()">Predict</button>
    </div>
    <div id="result"></div>
  </div>

  <script>
    async function sendPredict() {
      const text = document.getElementById('text').value.trim();
      const language = document.getElementById('lang').value.trim();
      if (!text) { alert('Please enter review text.'); return; }
      const payload = { text, language };
      const resp = await fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      const r = data.results && data.results[0];
      const el = document.getElementById('result');
      if (!r) { el.innerHTML = '<p>Unexpected response.</p>'; return; }
      const badgeClass = r.low_confidence ? 'badge low' : 'badge ok';
      el.innerHTML = `
        <h3>Prediction</h3>
        <p><strong>Sentiment:</strong> ${r.sentiment} <span class="${badgeClass}">${r.low_confidence ? 'low confidence' : 'ok'}</span></p>
        <p><strong>Confidence:</strong> ${(r.confidence*100).toFixed(1)}%</p>
        <p><strong>Probabilities:</strong> neg ${(r.probs.negative*100).toFixed(1)}% · neu ${(r.probs.neutral*100).toFixed(1)}% · pos ${(r.probs.positive*100).toFixed(1)}%</p>
        ${r.advisory ? `<p><em>Advisory:</em> ${r.advisory}</p>` : ''}
        <details style="margin-top:8px;">
          <summary>Submit Feedback (optional)</summary>
          <div style="margin-top:8px;">
            <label>True label (negative/neutral/positive)</label>
            <input id="true_label" type="text" placeholder="e.g., neutral" />
            <label>Notes</label>
            <input id="notes" type="text" placeholder="optional notes" style="width: 60%;" />
            <div style="margin-top:8px;">
              <button onclick="sendFeedback('${text.replace(/'/g,"&#39;")}', '${language}', '${r.sentiment}', ${r.confidence})">Send Feedback</button>
            </div>
          </div>
        </details>
        <p style="margin-top:10px;color:#666;">${data.disclaimer}</p>
      `;
    }

    async function sendFeedback(text, language, pred, confidence) {
      const true_label = document.getElementById('true_label').value.trim().toLowerCase();
      const notes = document.getElementById('notes').value.trim();
      if (!['negative','neutral','positive'].includes(true_label)) {
        alert('True label must be negative, neutral, or positive.'); return;
      }
      const payload = { text, language, pred, confidence, true_label, notes };
      const resp = await fetch('/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (resp.ok) alert('Thanks! Feedback saved.');
      else alert('Failed to save feedback.');
    }
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)