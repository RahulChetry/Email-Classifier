from __future__ import annotations

import base64
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CREDENTIALS_PATH = ROOT / "credentials.json"
TOKEN_DIR = ROOT / "instance" / "gmail_tokens"
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
IMPORTANT_LABEL = "AI Important"
NORMAL_LABEL = "AI Normal"


class GmailSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class GmailEmail:
    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    body: str


def dependencies_available() -> bool:
    try:
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401

        return True
    except ImportError:
        return False


def _token_path(token_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", token_id)
    if not safe_id:
        raise GmailSetupError("Missing Gmail session token.")
    return TOKEN_DIR / f"{safe_id}.json"


def _client_config() -> dict[str, Any] | None:
    raw = os.getenv("GOOGLE_CLIENT_CONFIG_JSON")
    encoded = os.getenv("GOOGLE_CLIENT_CONFIG_BASE64")
    if encoded:
        raw = base64.b64decode(encoded).decode("utf-8")
    if raw:
        return json.loads(raw)
    return None


def credentials_available() -> bool:
    return CREDENTIALS_PATH.exists() or _client_config() is not None


def gmail_status(token_id: str | None = None) -> dict[str, Any]:
    return {
        "dependencies": dependencies_available(),
        "credentials": credentials_available(),
        "connected": bool(token_id and _token_path(token_id).exists()),
        "scope": SCOPES[0],
    }


def _require_dependencies() -> None:
    if not dependencies_available():
        raise GmailSetupError(
            "Install Gmail dependencies with: pip install -r requirements.txt"
        )


def _require_credentials() -> None:
    if not credentials_available():
        raise GmailSetupError(
            "Missing Google OAuth credentials. Add credentials.json locally, or set GOOGLE_CLIENT_CONFIG_JSON in production."
        )


def build_oauth_flow(redirect_uri: str, state: str | None = None):
    _require_dependencies()
    _require_credentials()
    if redirect_uri.startswith("http://127.0.0.1") or redirect_uri.startswith("http://localhost"):
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    from google_auth_oauthlib.flow import Flow

    config = _client_config()
    if config:
        return Flow.from_client_config(
            config,
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            state=state,
        )

    return Flow.from_client_secrets_file(str(CREDENTIALS_PATH), scopes=SCOPES, redirect_uri=redirect_uri, state=state)


def save_credentials(token_id: str, creds) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    _token_path(token_id).write_text(creds.to_json(), encoding="utf-8")


def disconnect_gmail(token_id: str) -> None:
    path = _token_path(token_id)
    if path.exists():
        path.unlink()


def get_gmail_service(token_id: str):
    _require_dependencies()
    _require_credentials()

    token_path = _token_path(token_id)
    if not token_path.exists():
        raise GmailSetupError("Gmail is not connected yet.")

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_credentials(token_id, creds)

    if not creds.valid:
        raise GmailSetupError("Gmail token is invalid. Disconnect and connect Gmail again.")

    return build("gmail", "v1", credentials=creds)


def _header(headers: list[dict[str, str]], name: str) -> str:
    for item in headers:
        if item.get("name", "").lower() == name.lower():
            return item.get("value", "")
    return ""


def _decode_data(data: str) -> str:
    padded = data + ("=" * (-len(data) % 4))
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="ignore")


def _plain_text_from_payload(payload: dict[str, Any]) -> str:
    collected: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if data and mime_type in {"text/plain", "text/html"}:
            text = _decode_data(data)
            if mime_type == "text/html":
                text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html.unescape(text)
            collected.append(text)

        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    text = " ".join(collected)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]


def list_emails(token_id: str, limit: int = 25, query: str = "in:inbox") -> list[GmailEmail]:
    service = get_gmail_service(token_id)
    limit = max(1, min(int(limit), 500))
    messages: list[dict[str, str]] = []
    request = service.users().messages().list(
        userId="me",
        q=query or "in:inbox",
        maxResults=min(limit, 100),
    )

    while request is not None and len(messages) < limit:
        response = request.execute()
        messages.extend(response.get("messages", []))
        if len(messages) >= limit:
            break
        request = service.users().messages().list_next(request, response)

    emails: list[GmailEmail] = []
    for message in messages[:limit]:
        full = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        payload = full.get("payload", {})
        headers = payload.get("headers", [])
        snippet = html.unescape(full.get("snippet", ""))
        body = _plain_text_from_payload(payload) or snippet
        emails.append(
            GmailEmail(
                id=full["id"],
                thread_id=full.get("threadId", ""),
                subject=_header(headers, "Subject") or "(No subject)",
                sender=_header(headers, "From"),
                date=_header(headers, "Date"),
                snippet=snippet,
                body=body,
            )
        )

    return emails


def _label_id(service, label_name: str) -> str:
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label.get("name") == label_name:
            return label["id"]

    created = service.users().labels().create(
        userId="me",
        body={
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        },
    ).execute()
    return created["id"]


def apply_priority_label(token_id: str, message_id: str, label: str) -> None:
    service = get_gmail_service(token_id)
    important_id = _label_id(service, IMPORTANT_LABEL)
    normal_id = _label_id(service, NORMAL_LABEL)
    add_id = important_id if label == "important" else normal_id
    remove_id = normal_id if label == "important" else important_id
    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"addLabelIds": [add_id], "removeLabelIds": [remove_id]},
    ).execute()
