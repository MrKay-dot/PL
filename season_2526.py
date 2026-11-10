from pathlib import Path
import re
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    page_title="Premier League match predictor",
    page_icon=":material/sports_soccer:",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().with_name("season-2526.csv")
BACKGROUND_IMAGE = (
    "https://images.unsplash.com/photo-1574629810360-7efbbe195018"
    "?auto=format&fit=crop&w=1920&q=85"
)
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
MODEL_NAMES = (
    "Logistic regression",
    "Random forest",
    "Most-common baseline",
)
FORM_STATS = ("GF", "GA", "SF", "SA", "SOT", "SOTA", "Pts")
FEATURE_COLUMNS = tuple(
    f"{side}_form_{stat}"
    for side in ("home", "away")
    for stat in FORM_STATS
)
SMTP_SECRET_KEYS = (
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_SENDER",
    "CONTACT_RECIPIENT",
)

st.html(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(
                120deg,
                rgba(7, 18, 12, 0.88),
                rgba(7, 18, 12, 0.72) 55%,
                rgba(7, 18, 12, 0.86)
            ),
            url("{BACKGROUND_IMAGE}") center / cover fixed no-repeat;
    }}
    </style>
    """
)


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Season data was not found at {DATA_PATH}. "
            "Place season-2526.csv beside season_2526.py."
        )

    matches = pd.read_csv(DATA_PATH)
    required = {
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "FTR", "HS", "AS", "HST", "AST",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"The season CSV is missing required columns: {', '.join(missing)}")

    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    if matches["Date"].isna().any():
        raise ValueError("The season CSV contains missing or invalid match dates.")
    if matches[list(required - {"Date"})].isna().any().any():
        raise ValueError("The season CSV contains missing values in required match fields.")
    if not matches["FTR"].isin(OUTCOME_LABELS).all():
        raise ValueError("FTR must contain only H, D, or A match results.")

    return matches.sort_values("Date").reset_index(drop=True)


def build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches.index,
            "Date": matches["Date"],
            "team": matches["HomeTeam"],
            "side": "H",
            "GF": matches["FTHG"],
            "GA": matches["FTAG"],
            "SF": matches["HS"],
            "SA": matches["AS"],
            "SOT": matches["HST"],
            "SOTA": matches["AST"],
            "Pts": matches["FTR"].map({"H": 3, "D": 1, "A": 0}),
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches.index,
            "Date": matches["Date"],
            "team": matches["AwayTeam"],
            "side": "A",
            "GF": matches["FTAG"],
            "GA": matches["FTHG"],
            "SF": matches["AS"],
            "SA": matches["HS"],
            "SOT": matches["AST"],
            "SOTA": matches["HST"],
            "Pts": matches["FTR"].map({"A": 3, "D": 1, "H": 0}),
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "Date", "match_id"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def analyze_season() -> dict[str, Any]:
    matches = load_matches()
    history = build_team_history(matches)
    form = history.groupby("team", sort=False)[list(FORM_STATS)].transform(
        lambda values: values.shift(1).rolling(5, min_periods=3).mean()
    )
    form.columns = [f"form_{column}" for column in FORM_STATS]
    history = pd.concat([history, form], axis=1)

    home_form = (
        history.loc[history["side"] == "H"]
        .set_index("match_id")[form.columns]
        .add_prefix("home_")
    )
    away_form = (
        history.loc[history["side"] == "A"]
        .set_index("match_id")[form.columns]
        .add_prefix("away_")
    )
    features = pd.concat([home_form, away_form], axis=1).reindex(matches.index)
    model_data = pd.concat(
        [matches[["Date", "HomeTeam", "AwayTeam", "FTR"]], features], axis=1
    ).dropna(subset=[*FEATURE_COLUMNS, "FTR"])

    split_at = int(len(model_data) * 0.8)
    if split_at < 2 or len(model_data) - split_at == 0:
        raise ValueError("Not enough completed matches to create a time-based evaluation.")

    X = model_data.loc[:, FEATURE_COLUMNS]
    y = model_data["FTR"]
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    if y_train.nunique() < 2:
        raise ValueError("The training period needs at least two match outcome classes.")

    model_templates = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=42
        ),
        "Most-common baseline": DummyClassifier(strategy="most_frequent"),
    }
    evaluations: dict[str, Any] = {}
    for name, template in model_templates.items():
        evaluation_model = clone(template).fit(X_train, y_train)
        predicted = evaluation_model.predict(X_test)
        full_model = clone(template).fit(X, y)

        holdout = model_data.iloc[split_at:][
            ["Date", "HomeTeam", "AwayTeam", "FTR"]
        ].copy()
        holdout["Predicted"] = predicted
        holdout["Actual outcome"] = holdout.pop("FTR").map(OUTCOME_LABELS)
        holdout["Predicted outcome"] = holdout.pop("Predicted").map(OUTCOME_LABELS)

        evaluations[name] = {
            "train_accuracy": accuracy_score(y_train, evaluation_model.predict(X_train)),
            "test_accuracy": accuracy_score(y_test, predicted),
            "confusion_matrix": confusion_matrix(
                y_test, predicted, labels=["H", "D", "A"]
            ),
            "holdout": holdout,
            "model": full_model,
        }

    latest_five = history.groupby("team", sort=False, group_keys=False).tail(5)
    team_form = latest_five.groupby("team")[list(FORM_STATS)].mean()

    return {
        "matches": matches,
        "model_data": model_data,
        "team_form": team_form,
        "evaluations": evaluations,
    }


def fixture_features(home_team: str, away_team: str, team_form: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, float] = {}
    for side, team in (("home", home_team), ("away", away_team)):
        for stat in FORM_STATS:
            values[f"{side}_form_{stat}"] = float(team_form.loc[team, stat])
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def youtube_embed_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None

    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }:
        return None

    video_id: str | None = None
    path_parts = parsed.path.strip("/").split("/")
    if host == "youtu.be" and path_parts:
        video_id = path_parts[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif len(path_parts) == 2 and path_parts[0] in {"live", "embed", "shorts"}:
        video_id = path_parts[1]

    if video_id is None or re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) is None:
        return None
    return f"https://www.youtube-nocookie.com/embed/{video_id}"


def send_contact_message(
    name: str,
    sender_email: str,
    body: str,
    settings: dict[str, Any],
) -> None:
    email_message = EmailMessage()
    email_message["From"] = settings["SMTP_SENDER"]
    email_message["To"] = settings["CONTACT_RECIPIENT"]
    email_message["Reply-To"] = sender_email
    email_message["Subject"] = "Website contact message"
    email_message.set_content(f"From: {name}\nEmail: {sender_email}\n\n{body}")

    host = settings["SMTP_HOST"]
    port = settings["SMTP_PORT"]
    timeout = 20
    if port == 465:
        with smtplib.SMTP_SSL(
            host, port, timeout=timeout, context=ssl.create_default_context()
        ) as server:
            server.login(settings["SMTP_USERNAME"], settings["SMTP_PASSWORD"])
            server.send_message(email_message)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(settings["SMTP_USERNAME"], settings["SMTP_PASSWORD"])
            server.send_message(email_message)


def render_home() -> None:
    st.title("Premier League match predictor", icon=":material/sports_soccer:")
    st.caption(
        "Explore the season and estimate match outcomes from each team's recent form."
    )

    with st.container(border=True):
        st.subheader("Live match")
        st.caption(
            "Add a public, embeddable YouTube live-stream link. Only YouTube links "
            "are accepted; the stream must allow embedding."
        )
        with st.form("live_feed_form"):
            live_feed_input = st.text_input(
                "YouTube live-stream URL",
                placeholder="https://www.youtube.com/live/...",
                key="live_feed_url_input",
            )
            load_live_feed = st.form_submit_button(
                "Load live stream", icon=":material/play_circle:"
            )

        if load_live_feed:
            st.session_state["live_feed_url"] = live_feed_input.strip()

        live_feed_url = st.session_state.get("live_feed_url", "")
        if live_feed_url:
            embed_url = youtube_embed_url(live_feed_url)
            if embed_url is None:
                st.error(
                    "Enter a valid HTTPS YouTube watch, live, or youtu.be link."
                )
            else:
                st.iframe(
                    embed_url,
                    height=440,
                    alt="Live football match video",
                )
        else:
            st.info(
                "No live stream URL added yet. Paste a YouTube link above to display it."
            )

    analysis = analyze_season()
    matches = analysis["matches"]
    model_data = analysis["model_data"]
    evaluations = analysis["evaluations"]
    best_model_name = max(
        MODEL_NAMES, key=lambda name: evaluations[name]["test_accuracy"]
    )

    with st.container(border=True):
        st.subheader("Predict a fixture")
        st.write(
            "Select two teams to compare their recent scoring, shots, and points form."
        )
        teams = sorted(analysis["team_form"].index.tolist())
        with st.form("fixture_prediction"):
            home_team = st.selectbox("Home team", teams, key="prediction_home")
            away_options = [team for team in teams if team != home_team]
            away_team = st.selectbox("Away team", away_options, key="prediction_away")
            model_name = st.selectbox(
                "Prediction model",
                MODEL_NAMES,
                index=MODEL_NAMES.index(best_model_name),
            )
            submitted = st.form_submit_button(
                "Predict match", type="primary", icon=":material/auto_awesome:"
            )

        if submitted:
            model = evaluations[model_name]["model"]
            fixture = fixture_features(home_team, away_team, analysis["team_form"])
            probabilities = model.predict_proba(fixture)[0]
            st.session_state["latest_prediction"] = {
                "home_team": home_team,
                "away_team": away_team,
                "model_name": model_name,
                "probabilities": dict(zip(model.classes_, probabilities)),
            }

    prediction = st.session_state.get("latest_prediction")
    if prediction:
        st.subheader(f"{prediction['home_team']} vs {prediction['away_team']}")
        probability_columns = st.columns(3)
        for column, outcome in zip(probability_columns, ("H", "D", "A")):
            probability = float(prediction["probabilities"].get(outcome, 0.0))
            column.metric(OUTCOME_LABELS[outcome], f"{probability:.1%}")
        st.caption(
            f"Model: {prediction['model_name']}. Probabilities are estimates, "
            "not guarantees or betting advice."
        )

    most_recent = matches.iloc[-1]
    with st.container(horizontal=True):
        st.metric("Completed matches", f"{len(matches):,}", border=True)
        st.metric("Usable for model", f"{len(model_data):,}", border=True)
        st.metric("Teams", f"{analysis['team_form'].shape[0]:,}", border=True)
        st.metric("Top holdout model", best_model_name, border=True)

    st.caption(
        "Latest result in the file: "
        f"{most_recent['HomeTeam']} {int(most_recent['FTHG'])}–"
        f"{int(most_recent['FTAG'])} {most_recent['AwayTeam']} "
        f"({most_recent['Date']:%d %b %Y})."
    )


def render_dashboard() -> None:
    st.title("Dashboard", icon=":material/dashboard:")
    st.caption("Model performance and season results at a glance.")

    analysis = analyze_season()
    evaluations = analysis["evaluations"]
    best_model_name = max(
        MODEL_NAMES, key=lambda name: evaluations[name]["test_accuracy"]
    )
    best_evaluation = evaluations[best_model_name]
    matches = analysis["matches"]

    with st.container(horizontal=True):
        st.metric("Season matches", f"{len(matches):,}", border=True)
        st.metric(
            "Holdout accuracy",
            f"{best_evaluation['test_accuracy']:.1%}",
            border=True,
        )
        st.metric(
            "Best holdout model",
            best_model_name,
            border=True,
        )
        st.metric(
            "Teams",
            f"{analysis['team_form'].shape[0]:,}",
            border=True,
        )

    model_scores = pd.DataFrame(
        [
            {
                "Model": name,
                "Test accuracy": evaluations[name]["test_accuracy"],
            }
            for name in MODEL_NAMES
        ]
    )
    outcomes = (
        matches["FTR"]
        .map(OUTCOME_LABELS)
        .value_counts()
        .rename_axis("Outcome")
        .rename("Matches")
        .reset_index()
    )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Time-based model comparison")
            st.bar_chart(
                model_scores,
                x="Model",
                y="Test accuracy",
                alt="Holdout accuracy for each prediction model",
            )
    with right:
        with st.container(border=True):
            st.subheader("Season outcomes")
            st.bar_chart(
                outcomes,
                x="Outcome",
                y="Matches",
                alt="Number of home wins, draws, and away wins",
            )

    st.subheader(f"Recent holdout predictions — {best_model_name}")
    recent = best_evaluation["holdout"].tail(15).copy()
    recent["Date"] = recent["Date"].dt.strftime("%d %b %Y")
    st.dataframe(
        recent,
        hide_index=True,
        alt="Most recent match predictions in the time-based test period",
    )
    st.caption(
        "The holdout contains the final 20% of usable matches in date order. "
        "It is not a guarantee of future performance."
    )


def render_overview() -> None:
    st.title("Season overview", icon=":material/insights:")
    st.caption("Review the source data, match outcomes, and model approach.")

    analysis = analyze_season()
    matches = analysis["matches"]
    outcomes = matches["FTR"].map(OUTCOME_LABELS).value_counts()
    first_date, last_date = matches["Date"].min(), matches["Date"].max()
    with st.container(horizontal=True):
        st.metric("Matches in file", f"{len(matches):,}", border=True)
        st.metric("Season coverage", f"{first_date:%d %b} – {last_date:%d %b %Y}", border=True)
        st.metric("Home wins", f"{outcomes.get('Home win', 0):,}", border=True)
        st.metric("Draws", f"{outcomes.get('Draw', 0):,}", border=True)

    st.subheader("How predictions are made")
    st.markdown(
        """
        - For each team, features are the average of its previous five matches.
        - Each historical feature is shifted so the match being predicted is never
          included in that team's form.
        - Models are evaluated on the final 20% of usable matches, kept in date order.
        - The selected model is then retrained on all available matches for fixture
          predictions.
        """
    )

    st.subheader("Latest matches in the source file")
    recent = matches.tail(20)[
        ["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam", "FTR"]
    ].copy()
    recent["Date"] = recent["Date"].dt.strftime("%d %b %Y")
    recent["Result"] = recent.pop("FTR").map(OUTCOME_LABELS)
    st.dataframe(
        recent.iloc[::-1].reset_index(drop=True),
        hide_index=True,
        alt="Latest completed matches from the season CSV",
    )


def render_about() -> None:
    st.title("About us", icon=":material/info:")
    st.write(
        "This match-prediction demo turns the season CSV into a simple way to "
        "compare teams and explore model performance."
    )
    with st.container(border=True):
        st.subheader("What the app does")
        st.write(
            "It uses historical match results, goals, shots, shots on target, and "
            "points to estimate the chance of a home win, draw, or away win."
        )
    with st.container(border=True):
        st.subheader("What to keep in mind")
        st.write(
            "Predictions are based only on the supplied season file and recent "
            "five-match form. The app does not account for injuries, lineups, "
            "transfers, or live information, and probabilities are not guarantees."
        )


def render_contact() -> None:
    st.title("Contact us", icon=":material/mail:")
    st.write(
        "Send us a question or feedback using the form below."
    )
    try:
        configured_secrets = st.secrets.to_dict()
    except StreamlitSecretNotFoundError:
        configured_secrets = {}

    missing_secrets = [
        key for key in SMTP_SECRET_KEYS if not str(configured_secrets.get(key, "")).strip()
    ]
    settings: dict[str, Any] | None = None
    settings_error: str | None = None
    if not missing_secrets:
        try:
            port = int(configured_secrets["SMTP_PORT"])
        except (TypeError, ValueError):
            settings_error = "SMTP_PORT must be a valid port number."
        else:
            if not 1 <= port <= 65535:
                settings_error = "SMTP_PORT must be between 1 and 65535."
            else:
                settings = {
                    **configured_secrets,
                    "SMTP_PORT": port,
                }

    with st.form("contact_message_form"):
        name = st.text_input("Your name", max_chars=120)
        sender_email = st.text_input("Your email address", max_chars=254)
        message = st.text_area("Message", height=180, max_chars=5000)
        submitted = st.form_submit_button(
            "Send message", type="primary", icon=":material/send:"
        )

    if submitted:
        clean_name = " ".join(name.split())
        clean_email = sender_email.strip()
        clean_message = message.strip()
        if not clean_name:
            st.error("Please enter your name.")
        elif re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_email) is None:
            st.error("Please enter a valid email address.")
        elif not clean_message:
            st.error("Please enter a message.")
        elif settings_error:
            st.error(settings_error)
        elif missing_secrets:
            st.error(
                "Message delivery is not configured. Add the SMTP settings shown "
                "below to your local Streamlit secrets file."
            )
        elif settings is None:
            st.error("Email settings could not be loaded.")
        else:
            try:
                send_contact_message(clean_name, clean_email, clean_message, settings)
            except (OSError, smtplib.SMTPException) as error:
                st.error(f"Could not send your message: {error}")
            else:
                st.success("Your message was sent successfully.")

    if missing_secrets or settings_error:
        st.info(
            "To enable email delivery, add these values to "
            "`.streamlit/secrets.toml`. Keep this file private and do not commit it."
        )
        st.code(
            'SMTP_HOST = "smtp.example.com"\n'
            "SMTP_PORT = 587\n"
            'SMTP_USERNAME = "your-smtp-username"\n'
            'SMTP_PASSWORD = "your-smtp-password"\n'
            'SMTP_SENDER = "your-sending-address@example.com"\n'
            'CONTACT_RECIPIENT = "your-inbox@example.com"',
            language="toml",
        )


pages = [
    st.Page(
        render_home,
        title="Home",
        icon=":material/home:",
        help="Predict a fixture and see a quick summary of the loaded season.",
    ),
    st.Page(
        render_dashboard,
        title="Dashboard",
        icon=":material/dashboard:",
        help="Compare model holdout performance and explore the season results.",
    ),
    st.Page(
        render_overview,
        title="Overview",
        icon=":material/insights:",
        help="Review the season data and learn how match predictions are built.",
    ),
    st.Page(
        render_about,
        title="About us",
        icon=":material/info:",
        help="Learn what the predictor uses and what its estimates leave out.",
    ),
    st.Page(
        render_contact,
        title="Contact us",
        icon=":material/mail:",
        help="Find out how to get in touch or share feedback about this demo.",
    ),
]

page = st.navigation(pages, position="top")
page.run()

st.divider()
st.caption("© 2026 Premier League match predictor")
