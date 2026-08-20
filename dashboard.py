"""
================================================================
  FANTASY HOCKEY DASHBOARD  |  10-Cat Weekly Yahoo League
  Single-file Streamlit app -- no other files needed except
  requirements.txt.
  HOW TO RUN:
    1. pip install -r requirements.txt
    2. streamlit run dashboard.py
================================================================
"""
import base64
import hashlib
import secrets as pysecrets
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fantasy Hockey Dashboard", page_icon="\U0001F3D2", layout="wide",
                    initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
NHL_API = "https://api.nhle.com/stats/rest/en"
SEASONS_OF_HISTORY = 4          # + current season
MIN_GP_SKATER = 20
MIN_GP_GOALIE = 10

# label -> underlying column name (each Yahoo category maps to exactly one
# NHL stat here, so there's no sub-weighting like some baseball categories need)
SKATER_CATS = {"G": "goals", "A": "assists", "+/-": "plusMinus", "PIM": "penaltyMinutes",
               "PPP": "ppPoints", "SOG": "shots", "HIT": "hits", "FW": "faceoffWins"}
GOALIE_CATS = {"W": "wins", "SV%": "savePct"}


def current_season_id():
    import datetime
    today = datetime.date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    return int(f"{start_year}{start_year + 1}")


def season_list():
    latest = current_season_id()
    latest_start = int(str(latest)[:4])
    return [int(f"{latest_start - i}{latest_start - i + 1}") for i in range(SEASONS_OF_HISTORY, -1, -1)]


# ─────────────────────────────────────────────────────────────
#  NHL DATA (public API, no auth) -- cached 4h
# ─────────────────────────────────────────────────────────────
def _paginate(report, season, sort_field):
    out, start = [], 0
    cayenne = f"seasonId={season} and gameTypeId=2"
    while True:
        sort = urllib.parse.quote(f'[{{"property":"{sort_field}","direction":"DESC"}}]')
        url = (f"{NHL_API}/{report}?isAggregate=false&isGame=false&sort={sort}"
               f"&start={start}&limit=100&cayenneExp={urllib.parse.quote(cayenne)}")
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", [])
        out.extend(rows)
        start += 100
        if start >= data.get("total", len(out)) or not rows:
            break
    return out


@st.cache_data(ttl=14400, show_spinner="\U0001F3D2 Loading NHL stats...")
def load_data():
    seasons = season_list()
    skater_frames, goalie_frames = [], []

    for season in seasons:
        summary = pd.DataFrame(_paginate("skater/summary", season, "points"))
        realtime = pd.DataFrame(_paginate("skater/realtime", season, "hits"))
        faceoffs = pd.DataFrame(_paginate("skater/faceoffwins", season, "totalFaceoffWins"))
        if summary.empty:
            continue
        df = summary[["playerId", "skaterFullName", "teamAbbrevs", "positionCode", "gamesPlayed",
                      "goals", "assists", "points", "plusMinus", "penaltyMinutes", "ppPoints",
                      "shots"]].rename(columns={"skaterFullName": "Name", "teamAbbrevs": "Team",
                                                 "positionCode": "Position"})
        if not realtime.empty:
            df = df.merge(realtime[["playerId", "hits"]], on="playerId", how="left")
        else:
            df["hits"] = 0
        if not faceoffs.empty:
            df = df.merge(faceoffs[["playerId", "totalFaceoffWins"]].rename(
                columns={"totalFaceoffWins": "faceoffWins"}), on="playerId", how="left")
        else:
            df["faceoffWins"] = 0
        df[["hits", "faceoffWins"]] = df[["hits", "faceoffWins"]].fillna(0)
        df["Season"] = season
        skater_frames.append(df)

        gsummary = pd.DataFrame(_paginate("goalie/summary", season, "wins"))
        if not gsummary.empty:
            gdf = gsummary[["playerId", "goalieFullName", "teamAbbrevs", "gamesPlayed", "wins",
                            "losses", "savePct", "goalsAgainstAverage", "shutouts"]].rename(
                columns={"goalieFullName": "Name", "teamAbbrevs": "Team"})
            gdf["Season"] = season
            goalie_frames.append(gdf)

    skaters_all = pd.concat(skater_frames, ignore_index=True) if skater_frames else pd.DataFrame()
    goalies_all = pd.concat(goalie_frames, ignore_index=True) if goalie_frames else pd.DataFrame()
    latest = seasons[-1]
    return skaters_all, goalies_all, latest, seasons


# ─────────────────────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────────────────────
def _z(s):
    mu, sd = s.mean(), s.std()
    return pd.Series(0.0, index=s.index) if not sd else (s - mu) / sd


def score(df, cat_map, weights):
    df = df.copy()
    z_cols = []
    for label, col in cat_map.items():
        df[f"z_{label}"] = _z(df[col].fillna(0)).round(2)
        z_cols.append(f"z_{label}")
    total_w = sum(abs(weights.get(c.replace("z_", ""), 1.0)) for c in z_cols) or 1.0
    df["composite"] = sum(df[c] * weights.get(c.replace("z_", ""), 1.0) for c in z_cols) / total_w
    df["composite"] = df["composite"].round(2)
    df["rank"] = df["composite"].rank(ascending=False).astype(int)
    return df.sort_values("composite", ascending=False)


def default_weights(cat_map):
    return {label: 1.0 for label in cat_map}


def style_z(val):
    try:
        v = float(val)
        if v > 1.0: return "background-color:#1a472a"
        if v > 0.5: return "background-color:#2d5a3d"
        if v < -1.0: return "background-color:#5c1a1a"
        if v < -0.5: return "background-color:#7b2d2d"
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
try:
    skaters_all, goalies_all, LATEST, SEASONS = load_data()
except Exception as e:
    st.error(f"⚠️ Could not load NHL data: {e}")
    st.info("The NHL stats API may be temporarily unavailable. Try reloading in a minute.")
    st.stop()

if skaters_all.empty:
    st.error("No skater data returned from the NHL API. Try again shortly.")
    st.stop()

skaters_latest = skaters_all[skaters_all["Season"] == LATEST].copy()
skaters_latest = skaters_latest[skaters_latest["gamesPlayed"] >= MIN_GP_SKATER]
goalies_latest = goalies_all[goalies_all["Season"] == LATEST].copy() if not goalies_all.empty else pd.DataFrame()
if not goalies_latest.empty:
    goalies_latest = goalies_latest[goalies_latest["gamesPlayed"] >= MIN_GP_GOALIE]

if "skater_weights" not in st.session_state:
    st.session_state.skater_weights = default_weights(SKATER_CATS)
if "goalie_weights" not in st.session_state:
    st.session_state.goalie_weights = default_weights(GOALIE_CATS)

skaters_scored = score(skaters_latest, SKATER_CATS, st.session_state.skater_weights)
goalies_scored = score(goalies_latest, GOALIE_CATS, st.session_state.goalie_weights) if not goalies_latest.empty else pd.DataFrame()

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("\U0001F3D2 Hockey Dashboard")
st.sidebar.caption(f"Season {LATEST} · standard Yahoo 10-cat: G,A,+/-,PIM,PPP,SOG,HIT,FW,W,SV%")
page = st.sidebar.radio("Navigate", ["\U0001F4CB Draft Board", "\U0001F50D Player Deep Dive",
                                      "⚙️ Weight Dashboard", "\U0001F3C6 My Yahoo League"])
if st.sidebar.button("\U0001F504 Refresh NHL data"):
    st.cache_data.clear()
    st.rerun()

# ═════════════════════════════════════════════════════════════
#  PAGE: DRAFT BOARD
# ═════════════════════════════════════════════════════════════
if page == "\U0001F4CB Draft Board":
    st.title("\U0001F4CB Draft Board")
    st.caption(f"Composite z-score rankings — {LATEST} season. "
               f"Skaters need {MIN_GP_SKATER}+ GP, goalies need {MIN_GP_GOALIE}+ GP.")
    ptype = st.radio("Player type", ["Skaters", "Goalies"], horizontal=True, label_visibility="collapsed")
    df = skaters_scored.copy() if ptype == "Skaters" else goalies_scored.copy()
    cat_map = SKATER_CATS if ptype == "Skaters" else GOALIE_CATS

    if df.empty:
        st.info("No qualifying players yet this season.")
    else:
        c1, c2 = st.columns(2)
        teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
        team_f = c1.selectbox("Team", teams)
        if ptype == "Skaters":
            pos_f = c2.selectbox("Position", ["All"] + sorted(df["Position"].dropna().unique().tolist()))
        else:
            pos_f = "All"
        if team_f != "All":
            df = df[df["Team"] == team_f]
        if pos_f != "All":
            df = df[df["Position"] == pos_f]

        z_cols = [f"z_{c}" for c in cat_map]
        sort_by = st.selectbox("Sort by", ["composite"] + z_cols)
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df.index += 1

        show = ["Name", "Team"] + (["Position"] if ptype == "Skaters" else []) + ["composite"] + \
               list(cat_map.keys()) + z_cols
        show_cols = ["Name", "Team"] + (["Position"] if ptype == "Skaters" else []) + ["composite"] + \
                    list(cat_map.values()) + z_cols
        display_df = df[["Name", "Team"] + (["Position"] if ptype == "Skaters" else [])].copy()
        display_df["composite"] = df["composite"]
        for label, col in cat_map.items():
            display_df[label] = df[col]
        for zc in z_cols:
            display_df[zc] = df[zc]

        styled = (display_df.style
                  .map(style_z, subset=z_cols)
                  .background_gradient(subset=["composite"], cmap="RdYlGn")
                  .format({c: "{:.2f}" for c in ["composite"] + z_cols}))
        st.dataframe(styled, width="stretch", height=560)

        st.markdown("---")
        st.markdown("#### \U0001F4CA Category Scarcity")
        st.caption("Median / 75th / 90th percentile ('elite') value, qualifying players only.")
        rows = []
        for label, col in cat_map.items():
            s = df[col].dropna()
            rows.append({"Category": label, "Median": round(s.quantile(.5), 3),
                         "Good (P75)": round(s.quantile(.75), 3), "Elite (P90)": round(s.quantile(.9), 3)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════
#  PAGE: PLAYER DEEP DIVE
# ═════════════════════════════════════════════════════════════
elif page == "\U0001F50D Player Deep Dive":
    st.title("\U0001F50D Player Deep Dive")
    ptype = st.radio("Player type", ["Skater", "Goalie"], horizontal=True, label_visibility="collapsed")
    all_df = skaters_all if ptype == "Skater" else goalies_all
    scored_df = skaters_scored if ptype == "Skater" else goalies_scored
    cat_map = SKATER_CATS if ptype == "Skater" else GOALIE_CATS

    if all_df.empty:
        st.info("No data available.")
    else:
        name = st.selectbox("Select player", sorted(all_df["Name"].dropna().unique().tolist()))
        hist = all_df[all_df["Name"] == name].sort_values("Season")
        rec = scored_df[scored_df["Name"] == name] if not scored_df.empty else pd.DataFrame()

        if hist.empty:
            st.warning("No history found.")
        else:
            latest_row = hist.iloc[-1]
            st.markdown(f"## {name}  `{latest_row.get('Team', '-')}`"
                        f"{'  ' + str(latest_row.get('Position', '')) if ptype == 'Skater' else ''}")

            cols = st.columns(len(cat_map) + 1)
            for i, (label, col) in enumerate(cat_map.items()):
                val = latest_row.get(col, np.nan)
                disp = f"{val:.3f}" if label == "SV%" and pd.notna(val) else (str(int(val)) if pd.notna(val) else "-")
                cols[i].metric(label, disp)
            cols[-1].metric("GP", int(latest_row.get("gamesPlayed", 0)))

            if not rec.empty:
                r = rec.iloc[0]
                st.markdown("---")
                st.markdown(f"### Category Value · {LATEST}")
                st.caption(f"Composite **{r['composite']:.2f}** · ranked #{int(r['rank'])} of "
                           f"{len(scored_df)} qualifying {'skaters' if ptype == 'Skater' else 'goalies'}")
                zvals = [max(-3, min(3, float(r[f"z_{c}"]))) for c in cat_map]
                fig = go.Figure(go.Bar(x=list(cat_map.keys()), y=zvals,
                                        marker_color=["#21C354" if v > 0.3 else "#FFA500" if v > -0.3 else "#FF4B4B"
                                                      for v in zvals],
                                        text=[f"{v:+.2f}" for v in zvals], textposition="outside"))
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                fig.update_layout(template="plotly_dark", height=260, margin=dict(t=10, b=30))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Doesn't meet the minimum games-played threshold for ranking this season yet.")

            st.markdown("---")
            st.markdown("### \U0001F4C8 Season Trends")
            n_cats = len(cat_map)
            grid = st.columns(min(n_cats, 4))
            for i, (label, col) in enumerate(cat_map.items()):
                with grid[i % len(grid)]:
                    sub = hist[["Season", col]].dropna()
                    fig2 = px.line(sub, x="Season", y=col, markers=True, template="plotly_dark",
                                    title=label, color_discrete_sequence=["#4fc3f7"])
                    fig2.update_layout(height=220, margin=dict(l=10, r=10, t=36, b=10),
                                        xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig2, width="stretch")

# ═════════════════════════════════════════════════════════════
#  PAGE: WEIGHT DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Weight Dashboard":
    st.title("⚙️ Weight Dashboard")
    st.caption("Re-weight how much each category counts toward the composite score. Applies instantly everywhere.")
    tab_sk, tab_gl = st.tabs(["\U0001F3D2 Skaters", "\U0001F945 Goalies"])

    with tab_sk:
        cols = st.columns(len(SKATER_CATS))
        for i, label in enumerate(SKATER_CATS):
            st.session_state.skater_weights[label] = cols[i].slider(
                label, 0.0, 3.0, float(st.session_state.skater_weights[label]), 0.25, key=f"sw_{label}")
        if st.button("Reset skater weights"):
            st.session_state.skater_weights = default_weights(SKATER_CATS)
            st.rerun()

        preview = score(skaters_latest, SKATER_CATS, st.session_state.skater_weights)
        default_scored = score(skaters_latest, SKATER_CATS, default_weights(SKATER_CATS))
        default_rank = dict(zip(default_scored["Name"], default_scored["rank"]))
        preview["rank_change"] = preview["Name"].map(default_rank) - preview["rank"]
        st.markdown("#### Live Preview — Top 15")
        show = preview[["rank", "Name", "Team", "composite", "rank_change"]].head(15)
        st.dataframe(show.style.map(
            lambda v: ("color:#21C354" if isinstance(v, (int, np.integer)) and v > 0 else
                       "color:#FF4B4B" if isinstance(v, (int, np.integer)) and v < 0 else ""),
            subset=["rank_change"]), width="stretch", hide_index=True)

    with tab_gl:
        if goalies_latest.empty:
            st.info("No goalie data available.")
        else:
            cols = st.columns(len(GOALIE_CATS))
            for i, label in enumerate(GOALIE_CATS):
                st.session_state.goalie_weights[label] = cols[i].slider(
                    label, 0.0, 3.0, float(st.session_state.goalie_weights[label]), 0.25, key=f"gw_{label}")
            if st.button("Reset goalie weights"):
                st.session_state.goalie_weights = default_weights(GOALIE_CATS)
                st.rerun()

            preview = score(goalies_latest, GOALIE_CATS, st.session_state.goalie_weights)
            default_scored = score(goalies_latest, GOALIE_CATS, default_weights(GOALIE_CATS))
            default_rank = dict(zip(default_scored["Name"], default_scored["rank"]))
            preview["rank_change"] = preview["Name"].map(default_rank) - preview["rank"]
            st.markdown("#### Live Preview")
            show = preview[["rank", "Name", "Team", "composite", "rank_change"]].head(15)
            st.dataframe(show.style.map(
                lambda v: ("color:#21C354" if isinstance(v, (int, np.integer)) and v > 0 else
                           "color:#FF4B4B" if isinstance(v, (int, np.integer)) and v < 0 else ""),
                subset=["rank_change"]), width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════
#  PAGE: MY YAHOO LEAGUE  (OAuth2 PKCE flow for Streamlit Cloud)
# ═════════════════════════════════════════════════════════════
elif page == "\U0001F3C6 My Yahoo League":
    YAHOO_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
    YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
    YAHOO_API_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

    # ---- EDIT THIS after you deploy to Streamlit Cloud, and register the
    # ---- same URL as a Redirect URI on your Yahoo app at
    # ---- https://developer.yahoo.com/apps/
    REDIRECT_URI = "https://REPLACE-WITH-YOUR-APP.streamlit.app/"

    try:
        CONSUMER_KEY = st.secrets["yahoo"]["consumer_key"]
        CONSUMER_SECRET = st.secrets["yahoo"]["consumer_secret"]
    except Exception:
        st.error("⚠️ Yahoo credentials not found in Streamlit secrets. Go to your app's "
                 "**Settings → Secrets** and add:\n\n```toml\n[yahoo]\nconsumer_key = \"...\"\n"
                 "consumer_secret = \"...\"\n```")
        st.stop()

    def _pkce_pair():
        verifier = pysecrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        return verifier, challenge

    def _make_state(verifier):
        import hmac as _hmac
        sig = _hmac.new(CONSUMER_SECRET.encode(), verifier.encode(), hashlib.sha256).hexdigest()[:16]
        return base64.urlsafe_b64encode(f"{verifier}||{sig}".encode()).decode().rstrip("=")

    def _decode_state(state):
        import hmac as _hmac
        try:
            padded = state + "=" * (-len(state) % 4)
            verifier, sig = base64.urlsafe_b64decode(padded).decode().split("||", 1)
            expected = _hmac.new(CONSUMER_SECRET.encode(), verifier.encode(), hashlib.sha256).hexdigest()[:16]
            if _hmac.compare_digest(sig, expected):
                return verifier
        except Exception:
            pass
        return None

    def _auth_url(state, challenge):
        params = {"client_id": CONSUMER_KEY, "redirect_uri": REDIRECT_URI, "response_type": "code",
                  "scope": "fspt-r", "state": state, "code_challenge": challenge,
                  "code_challenge_method": "S256"}
        return YAHOO_AUTH_URL + "?" + urllib.parse.urlencode(params)

    def _exchange_code(code, verifier):
        creds = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()
        r = requests.post(YAHOO_TOKEN_URL, headers={"Authorization": f"Basic {creds}",
                          "Content-Type": "application/x-www-form-urlencoded"},
                          data={"grant_type": "authorization_code", "code": code,
                                "redirect_uri": REDIRECT_URI, "code_verifier": verifier}, timeout=15)
        if r.status_code != 200:
            return {"error": f"Token exchange failed ({r.status_code}): {r.text[:400]}"}
        d = r.json()
        d["issued_at"] = time.time()
        return d

    def _refresh_token(refresh_tok):
        creds = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()
        r = requests.post(YAHOO_TOKEN_URL, headers={"Authorization": f"Basic {creds}",
                          "Content-Type": "application/x-www-form-urlencoded"},
                          data={"grant_type": "refresh_token", "refresh_token": refresh_tok,
                                "redirect_uri": REDIRECT_URI}, timeout=15)
        if r.status_code != 200:
            return {"error": f"Refresh failed ({r.status_code}): {r.text[:400]}"}
        d = r.json()
        d["issued_at"] = time.time()
        return d

    def _api(path):
        tok = st.session_state.get("yahoo_token", {})
        if not tok:
            return {"error": "not_authenticated"}
        age = time.time() - tok.get("issued_at", 0)
        if age > tok.get("expires_in", 3600) - 300:
            new_tok = _refresh_token(tok.get("refresh_token", ""))
            if "error" not in new_tok:
                st.session_state["yahoo_token"] = new_tok
                tok = new_tok
            else:
                return {"error": "Token expired and refresh failed. Please reconnect."}
        headers = {"Authorization": f"Bearer {tok['access_token']}", "Accept": "application/json"}
        url = f"{YAHOO_API_BASE}{path}" + ("&" if "?" in path else "?") + "format=json"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
        return r.json()

    for k in ["yahoo_token", "yahoo_league_key", "yahoo_my_team_key", "yahoo_my_team_name",
              "yahoo_auth_url", "yahoo_stat_map"]:
        if k not in st.session_state:
            st.session_state[k] = None

    qp = st.query_params
    if "code" in qp and st.session_state["yahoo_token"] is None:
        verifier = _decode_state(qp.get("state", ""))
        if verifier:
            with st.spinner("Exchanging authorization code..."):
                tok = _exchange_code(qp["code"], verifier)
            if "error" in tok:
                st.error(f"OAuth error: {tok['error']}")
            else:
                st.session_state["yahoo_token"] = tok
                st.session_state["yahoo_auth_url"] = None
            st.query_params.clear()
            st.rerun()
        else:
            st.warning("Could not verify OAuth state. Try connecting again.")
            st.query_params.clear()

    st.title("\U0001F3C6 My Yahoo League")

    if st.session_state["yahoo_token"] is None:
        st.markdown("### Connect your Yahoo Fantasy account")
        if st.session_state.get("yahoo_auth_url") is None:
            verifier, challenge = _pkce_pair()
            state = _make_state(verifier)
            st.session_state["yahoo_auth_url"] = _auth_url(state, challenge)
        st.link_button("\U0001F517 Connect Yahoo Fantasy", st.session_state["yahoo_auth_url"], type="primary")
        st.caption("You'll be redirected to Yahoo, then brought back here automatically.")

    else:
        col1, col2 = st.columns([5, 1])
        col1.markdown("### ✅ Connected to Yahoo Fantasy")
        if col2.button("Disconnect"):
            for k in ["yahoo_token", "yahoo_league_key", "yahoo_my_team_key", "yahoo_my_team_name",
                      "yahoo_auth_url"]:
                st.session_state[k] = None
            st.rerun()

        # Discover the real stat_id -> display_name map from Yahoo itself,
        # instead of guessing IDs (they aren't published anywhere reliable).
        if st.session_state.get("yahoo_stat_map") is None:
            sc = _api("/game/nhl/stat_categories")
            stat_map = {}
            try:
                cats = sc["fantasy_content"]["game"][1]["stat_categories"]["stats"]
                for k, v in cats.items():
                    if k == "count":
                        continue
                    s = v["stat"]
                    stat_map[str(s["stat_id"])] = s.get("display_name") or s.get("name")
            except Exception:
                pass
            st.session_state["yahoo_stat_map"] = stat_map or None

        if st.session_state["yahoo_league_key"] is None:
            with st.spinner("Loading your leagues..."):
                lg_resp = _api("/users;use_login=1/games;game_keys=nhl/leagues")
            leagues = []
            try:
                games = lg_resp["fantasy_content"]["users"]["0"]["user"][1]["games"]
                for gk, gv in games.items():
                    if gk == "count":
                        continue
                    for lk, lv in gv["game"][1].get("leagues", {}).items():
                        if lk == "count":
                            continue
                        lg = lv["league"][0]
                        leagues.append({"key": lg.get("league_key", ""), "name": lg.get("name", ""),
                                        "teams": lg.get("num_teams", "?"), "season": lg.get("season", "")})
            except Exception as e:
                st.error(f"Could not parse leagues: {e}")
                st.stop()
            if not leagues:
                st.info("No active NHL leagues found on this Yahoo account.")
                st.stop()
            st.markdown("#### Select your league")
            for lg in leagues:
                if st.button(f"**{lg['name']}** — {lg['season']} · {lg['teams']} teams", key=f"lg_{lg['key']}"):
                    st.session_state["yahoo_league_key"] = lg["key"]
                    teams_resp = _api(f"/league/{lg['key']}/teams")
                    try:
                        for k, v in teams_resp["fantasy_content"]["league"][1]["teams"].items():
                            if k == "count":
                                continue
                            t = v["team"][0]
                            is_mine = int(next((x.get("is_owned_by_current_login", 0)
                                                for x in t if isinstance(x, dict)), 0))
                            if is_mine:
                                st.session_state["yahoo_my_team_key"] = next(
                                    (x["team_key"] for x in t if isinstance(x, dict) and "team_key" in x), None)
                                st.session_state["yahoo_my_team_name"] = next(
                                    (x["name"] for x in t if isinstance(x, dict) and "name" in x), "My Team")
                    except Exception:
                        pass
                    st.rerun()
        else:
            league_key = st.session_state["yahoo_league_key"]
            my_team_key = st.session_state["yahoo_my_team_key"]
            my_team_name = st.session_state["yahoo_my_team_name"] or "My Team"
            st.markdown(f"**League:** `{league_key}` · **My team:** {my_team_name}")
            if st.button("← Switch league"):
                st.session_state["yahoo_league_key"] = None
                st.rerun()

            tab_roster, tab_standings, tab_matchup = st.tabs(["\U0001F465 My Roster", "\U0001F4CA Standings",
                                                                "⚔️ This Week"])

            with tab_roster:
                if not my_team_key:
                    st.warning("Could not auto-detect your team on this league.")
                else:
                    r = _api(f"/team/{my_team_key}/roster/players")
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        try:
                            entries = r["fantasy_content"]["team"][1]["roster"]["0"]["players"]
                            rows = []
                            for k, v in entries.items():
                                if k == "count":
                                    continue
                                p0 = v["player"][0]
                                rows.append({
                                    "Name": next((x["name"]["full"] for x in p0 if isinstance(x, dict) and "name" in x), "?"),
                                    "Pos": next((x["display_position"] for x in p0 if isinstance(x, dict) and "display_position" in x), ""),
                                    "NHL Team": next((x["editorial_team_abbr"] for x in p0 if isinstance(x, dict) and "editorial_team_abbr" in x), ""),
                                    "Status": next((x.get("status", "Active") for x in p0 if isinstance(x, dict) and "status" in x), "Active") or "Active",
                                })
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                        except Exception as e:
                            st.warning(f"Could not parse roster: {e}")

            with tab_standings:
                sd = _api(f"/league/{league_key}/standings")
                if "error" in sd:
                    st.error(sd["error"])
                else:
                    try:
                        rows = []
                        teams = sd["fantasy_content"]["league"][1]["standings"][0]["teams"]
                        for k, v in teams.items():
                            if k == "count":
                                continue
                            t = v["team"]
                            info = t[0]
                            standings = t[2]["team_standings"] if len(t) > 2 else {}
                            name = next((x["name"] for x in info if isinstance(x, dict) and "name" in x), "?")
                            outcome = standings.get("outcome_totals", {})
                            rows.append({"Rank": standings.get("rank", "?"),
                                        "Team": ("\U0001F7E2 " if name == my_team_name else "") + name,
                                        "W": outcome.get("wins", "?"), "L": outcome.get("losses", "?"),
                                        "T": outcome.get("ties", "?"), "Win%": outcome.get("percentage", "?")})
                        sdf = pd.DataFrame(rows)
                        try:
                            sdf["Rank"] = sdf["Rank"].astype(int)
                            sdf = sdf.sort_values("Rank")
                        except Exception:
                            pass
                        st.dataframe(sdf, width="stretch", hide_index=True)
                    except Exception as e:
                        st.warning(f"Could not parse standings: {e}")

            with tab_matchup:
                if not my_team_key:
                    st.warning("Could not auto-detect your team on this league.")
                else:
                    mu_d = _api(f"/team/{my_team_key}/matchups")
                    if "error" in mu_d:
                        st.error(mu_d["error"])
                    else:
                        try:
                            stat_map = st.session_state.get("yahoo_stat_map") or {}
                            matchups_raw = mu_d["fantasy_content"]["team"][1]["matchups"]
                            cur_mu = None
                            for mk, mv in matchups_raw.items():
                                if mk == "count":
                                    continue
                                mu = mv.get("matchup", mv)
                                if str(mu.get("is_current_week", "0")) == "1":
                                    cur_mu = mu
                                    break
                            if cur_mu is None:
                                st.info("No active matchup (season may not have started).")
                            else:
                                week = cur_mu.get("week", "?")
                                st.markdown(f"**Week {week}**")
                                mu_teams = cur_mu["0"]["teams"]
                                summaries = []
                                for tk in ["0", "1"]:
                                    tm = mu_teams[tk]["team"]
                                    t_info = tm[0]
                                    name = next((x["name"] for x in t_info if isinstance(x, dict) and "name" in x), "?")
                                    tkey = next((x["team_key"] for x in t_info if isinstance(x, dict) and "team_key" in x), None)
                                    stats, winners = {}, {}
                                    if len(tm) > 1:
                                        for sk, sv in tm[1].get("team_stats", {}).get("stats", {}).items():
                                            if sk == "count":
                                                continue
                                            s = sv.get("stat", {})
                                            stats[s.get("stat_id", "")] = s.get("value", "-")
                                        for sk, sv in tm[1].get("team_stats", {}).get("stat_winners", {}).items():
                                            if sk == "count":
                                                continue
                                            w = sv.get("stat_winner", {})
                                            winners[w.get("stat_id", "")] = w.get("winner_team_key", "")
                                    summaries.append({"name": name, "key": tkey, "stats": stats, "winners": winners})
                                me = next((s for s in summaries if s["key"] == my_team_key), summaries[0])
                                opp = next((s for s in summaries if s["key"] != my_team_key), summaries[-1])
                                rows, my_score, opp_score = [], 0, 0
                                for sid in me["stats"]:
                                    cat_name = stat_map.get(sid, f"stat {sid}")
                                    winner = me["winners"].get(sid, "")
                                    if winner == my_team_key:
                                        result, my_score = "W", my_score + 1
                                    elif winner:
                                        result, opp_score = "L", opp_score + 1
                                    else:
                                        result = "T"
                                    rows.append({"Category": cat_name, f"Me": me["stats"].get(sid, "-"),
                                                f"Opp ({opp['name']})": opp["stats"].get(sid, "-"), "Result": result})
                                c1, c2 = st.columns(2)
                                c1.metric("My Score", my_score)
                                c2.metric("Opp Score", opp_score)
                                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                        except Exception as e:
                            st.warning(f"Could not parse matchup: {e}")