# Email Priority Classifier

Classifies emails as `important` or `normal` using two approaches:

- Traditional ML: TF-IDF features plus logistic regression.
- GenAI: an optional OpenAI-compatible chat-completion call, with a local fallback when no API key is configured.

## Dataset

The included dataset is a reproducible synthetic dataset with 10,000 rows:

- `5,000` important emails
- `5,000` normal emails

Generate it again or create a larger one with:

```powershell
python scripts/generate_dataset.py --rows 10000
```

For a real production classifier, replace or extend `data/emails.csv` with labeled emails from your own domain.

## Run

```powershell
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Optional GenAI API

Set an API key before running the app:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:GENAI_MODEL="gpt-4o-mini"
python app.py
```

For another OpenAI-compatible provider, set:

```powershell
$env:GENAI_API_URL="https://your-provider.example/v1/chat/completions"
$env:GENAI_API_KEY="your_api_key"
```

## Gmail Integration

The app can connect to Gmail, classify matching messages, and optionally apply Gmail labels:

- `AI Important`
- `AI Normal`

Setup:

1. In Google Cloud Console, enable the Gmail API.
2. Configure the OAuth consent screen.
3. For local development, create an OAuth client with application type `Desktop app`.
4. Download the file, rename it to `credentials.json`, and place it in this project folder.
5. Run the app and click `Connect Gmail`.

The Gmail integration uses:

```text
https://www.googleapis.com/auth/gmail.modify
```

This lets the app read messages and apply/remove Gmail labels. Your private `credentials.json` and `token.json` files are ignored by git.

## Public Deployment

This project is deployment-ready for Python web hosts such as Render, Railway, Fly.io, or any service that can run Gunicorn.

Included deployment files:

- `Procfile` - starts the Flask app with Gunicorn.
- `render.yaml` - Render blueprint configuration.
- `runtime.txt` - Python runtime hint.

For Render:

1. Push this project to GitHub.
2. Create a new Render Web Service from the repository.
3. Use build command:

```bash
pip install -r requirements.txt
```

4. Use start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

5. Add environment variables:

```text
FLASK_ENV=production
FLASK_SECRET_KEY=<generate-a-long-random-secret>
GOOGLE_CLIENT_CONFIG_JSON=<your Google OAuth web client JSON>
OPENAI_API_KEY=<optional>
GENAI_MODEL=gpt-4o-mini
```

For Gmail on a public URL, create a Google OAuth client with application type `Web application`, not `Desktop app`. Add this authorized redirect URI:

```text
https://YOUR-DEPLOYED-DOMAIN/gmail/oauth2callback
```

Then paste the downloaded OAuth JSON into the `GOOGLE_CLIENT_CONFIG_JSON` environment variable. The app stores Gmail tokens separately per visitor session under `instance/gmail_tokens/`; for a serious production system, replace this with a database-backed token store.

## Files

- `data/emails.csv` - generated labeled email dataset with 10,000 rows by default.
- `scripts/generate_dataset.py` - reproducible synthetic dataset generator.
- `ml_model.py` - training and prediction code for traditional ML.
- `genai_classifier.py` - GenAI prompt, API call, JSON parsing, and fallback logic.
- `gmail_integration.py` - Gmail OAuth, message fetch, text extraction, and label helpers.
- `app.py` - Flask routes and API.
- `templates/index.html` and `static/styles.css` - UI.

## API

```http
POST /predict
Content-Type: application/json

{
  "subject": "Server outage impacting checkout",
  "body": "Checkout API is returning 500 errors for all customers.",
  "mode": "both"
}
```

`mode` can be `ml`, `genai`, or `both`.
