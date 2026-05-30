from __future__ import annotations

import os
import secrets

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from genai_classifier import classify_with_genai
from gmail_integration import (
    GmailSetupError,
    apply_priority_label,
    build_oauth_flow,
    disconnect_gmail,
    gmail_status,
    list_emails,
    save_credentials,
)
from ml_model import load_or_train_model, predict_priority, train_model


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "email-priority-dev-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
if os.getenv("FLASK_ENV") == "production":
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_SAMESITE="Lax")


def gmail_token_id() -> str:
    token_id = session.get("gmail_token_id")
    if not token_id:
        token_id = secrets.token_urlsafe(24)
        session["gmail_token_id"] = token_id
    return token_id


@app.get("/")
def index():
    _, metrics = load_or_train_model()
    return render_template("index.html", metrics=metrics)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/gmail/status")
def gmail_status_route():
    return jsonify(gmail_status(gmail_token_id()))


@app.get("/gmail/connect")
def gmail_connect():
    try:
        flow = build_oauth_flow(url_for("gmail_oauth_callback", _external=True))
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        session["gmail_oauth_state"] = state
        return redirect(authorization_url)
    except GmailSetupError as exc:
        return render_template("gmail_setup.html", message=str(exc)), 400


@app.get("/gmail/oauth2callback")
def gmail_oauth_callback():
    try:
        flow = build_oauth_flow(
            url_for("gmail_oauth_callback", _external=True),
            state=session.get("gmail_oauth_state"),
        )
        flow.fetch_token(authorization_response=request.url)
        save_credentials(gmail_token_id(), flow.credentials)
        session.pop("gmail_oauth_state", None)
        return redirect(url_for("index", gmail="connected"))
    except Exception as exc:
        return render_template("gmail_setup.html", message=f"Gmail connection failed: {exc}"), 400


@app.post("/gmail/disconnect")
def gmail_disconnect():
    disconnect_gmail(gmail_token_id())
    return jsonify(gmail_status(gmail_token_id()))


@app.post("/gmail/classify")
def gmail_classify():
    payload = request.get_json(silent=True) or {}
    limit = int(payload.get("limit", 25) or 25)
    query = str(payload.get("query", "in:inbox")).strip() or "in:inbox"
    mode = str(payload.get("mode", "ml")).strip().lower()
    apply_labels = bool(payload.get("apply_labels", False))

    if mode not in {"ml", "genai"}:
        return jsonify({"error": "Gmail batch mode must be ml or genai."}), 400

    try:
        token_id = gmail_token_id()
        emails = list_emails(token_id=token_id, limit=limit, query=query)
        rows = []
        for email in emails:
            if mode == "genai":
                prediction = classify_with_genai(email.subject, email.body)
                result = {
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "source": prediction.source,
                }
            else:
                prediction = predict_priority(email.subject, email.body)
                result = {
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "source": "ml",
                }

            if apply_labels:
                apply_priority_label(token_id, email.id, result["label"])

            rows.append(
                {
                    "id": email.id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "date": email.date,
                    "snippet": email.snippet,
                    "prediction": result,
                }
            )

        return jsonify(
            {
                "count": len(rows),
                "query": query,
                "mode": mode,
                "applied_labels": apply_labels,
                "emails": rows,
            }
        )
    except GmailSetupError as exc:
        return jsonify({"error": str(exc), "status": gmail_status(gmail_token_id())}), 400
    except Exception as exc:
        return jsonify({"error": f"Gmail classification failed: {exc}"}), 500


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    subject = str(payload.get("subject", "")).strip()
    body = str(payload.get("body", "")).strip()
    mode = str(payload.get("mode", "both")).strip().lower()

    if not subject and not body:
        return jsonify({"error": "Enter a subject or body to classify."}), 400

    response = {}
    if mode in {"ml", "both"}:
        ml_prediction = predict_priority(subject, body)
        response["ml"] = {
            "label": ml_prediction.label,
            "confidence": ml_prediction.confidence,
            "probabilities": ml_prediction.probabilities,
        }

    if mode in {"genai", "both"}:
        genai_prediction = classify_with_genai(subject, body)
        response["genai"] = {
            "label": genai_prediction.label,
            "confidence": genai_prediction.confidence,
            "rationale": genai_prediction.rationale,
            "source": genai_prediction.source,
        }

    return jsonify(response)


@app.post("/train")
def train():
    metrics = train_model()
    return jsonify(metrics)


if __name__ == "__main__":
    load_or_train_model()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
