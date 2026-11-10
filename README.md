# Premier League match predictor

A Streamlit app for exploring Premier League match data and estimating match
outcomes from recent team form.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run season_2526.py
```

Keep `season-2526.csv` in the same directory as `season_2526.py`.

## Deploy from GitHub

GitHub Pages only hosts static files; this app runs Python and Streamlit. Deploy
it with [Streamlit Community Cloud](https://share.streamlit.io/) connected to
this GitHub repository:

1. Push this project to GitHub.
2. In Streamlit Community Cloud, choose **Create app** and select the repository
   and branch.
3. Set the main file path to `season_2526.py`, then deploy.
4. If enabling email delivery, add the SMTP settings below in the app's
   **Settings → Secrets**. Do not commit credentials to GitHub.

The app's dependencies are listed in `requirements.txt`.

### Optional email delivery

Contact-form delivery is disabled until SMTP secrets are configured in
Streamlit Community Cloud:

```toml
SMTP_HOST = "smtp.example.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-smtp-username"
SMTP_PASSWORD = "your-smtp-password"
SMTP_SENDER = "your-sending-address@example.com"
CONTACT_RECIPIENT = "your-inbox@example.com"
```
