# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Player Performance Dashboard")

# ----------------------------
# Helpers: load data (pickles or csv fallback)
# ----------------------------
@st.cache_data
def load_data():
    # Prefer pickles created by train.py
    if os.path.exists("model/players.pkl") and os.path.exists("model/gk.pkl"):
        players = joblib.load("model/players.pkl")
        goalkeepers = joblib.load("model/gk.pkl")
    else:
        # fallback: try to load CSV and compute final_score on the fly
        if not os.path.exists("players_20.csv"):
            st.error("Missing data files. Please provide players.pkl & gk.pkl or players_20.csv.")
            return pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv("players_20.csv")
        # keep required columns safely
        cols = [
            "short_name", "long_name", "age", "height_cm", "weight_kg",
            "nationality", "club", "overall", "potential",
            "dribbling", "passing", "shooting",
            "value_eur", "wage_eur", "player_positions"
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[cols].copy()
        # Separate GK and outfield
        mask_gk = df["player_positions"].astype(str).str.contains("GK", na=False)
        goalkeepers = df[mask_gk].copy()
        players = df[~mask_gk].copy()

        # fill numeric columns
        for c in ["overall", "potential", "dribbling", "passing", "shooting"]:
            players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0)

        # compute final_score
        players["final_score"] = (
            players["overall"] * 0.5 +
            players["potential"] * 0.5 +
            players["dribbling"] * 0.1 +
            players["passing"] * 0.1 +
            players["shooting"] * 0.1
        )

        # Ensure numeric columns in GK
        for c in ["overall", "potential"]:
            goalkeepers[c] = pd.to_numeric(goalkeepers[c], errors="coerce").fillna(0)

    # Ensure consistent dtypes & no NAs for display
    for df in (players, goalkeepers):
        if "short_name" in df.columns:
            df["short_name"] = df["short_name"].astype(str)
        if "long_name" in df.columns:
            df["long_name"] = df["long_name"].astype(str)
        if "player_positions" in df.columns:
            df["player_positions"] = df["player_positions"].astype(str)
    return players, goalkeepers


players, goalkeepers = load_data()

# If data not loaded, stop
if players.empty and goalkeepers.empty:
    st.stop()

# Combined frame for some operations
df_all = pd.concat([players, goalkeepers], ignore_index=True, sort=False)

# ----------------------------
# UI Sidebar
# ----------------------------
st.title("⚽ Player & Goalkeeper Performance Dashboard")
st.sidebar.header("Options")

menu = st.sidebar.radio(
    "Choose view:",
    [
        "Top Players (final_score)",
        "Top 10 Goalkeepers",
        "Player Search",
        "Goalkeeper Search",
        "Player Profile Page",
        "Filters (Club / Nation / Age / Position)"
    ],
)

# ----------------------------
# Top Players by final_score
# ----------------------------
if menu == "Top Players (final_score)":
    st.subheader("🏆 Top 25 Outfield Players by Final Score")
    if "final_score" not in players.columns:
        st.warning("final_score not found in players data. Re-run train.py or use players_20.csv fallback.")
    top_players = players.sort_values(by="final_score", ascending=False).head(25).reset_index(drop=True)
    st.dataframe(top_players, use_container_width=True)

    with st.expander("Download top players as CSV"):
        st.download_button("Download CSV", top_players.to_csv(index=False), file_name="top_players_final_score.csv")

# ----------------------------
# Top 10 Goalkeepers
# ----------------------------
elif menu == "Top 10 Goalkeepers":
    st.subheader("🧤 Top 10 Goalkeepers by Overall Rating")
    top_gk = goalkeepers.sort_values(by="overall", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(top_gk, use_container_width=True)

    with st.expander("Download top goalkeepers as CSV"):
        st.download_button("Download CSV", top_gk.to_csv(index=False), file_name="top_goalkeepers.csv")

# ----------------------------
# Player Search (Outfield + GK combined)
# ----------------------------
elif menu == "Player Search":
    st.subheader("🔎 Search Any Player (Outfield & GK)")

    q = st.text_input("Enter player name (partial or full, case-insensitive)")
    if q:
        # search both datasets
        mask_players = players["short_name"].str.contains(q, case=False, na=False)
        mask_gk = goalkeepers["short_name"].str.contains(q, case=False, na=False)
        res = pd.concat([players[mask_players], goalkeepers[mask_gk]], ignore_index=True, sort=False)

        if res.empty:
            st.warning("No player found. Try different spelling or a shorter query.")
        else:
            st.success(f"Found {len(res)} player(s)")
            st.dataframe(res.reset_index(drop=True), use_container_width=True)

# ----------------------------
# Goalkeeper Search (clean)
# ----------------------------
elif menu == "Goalkeeper Search":
    st.subheader("🧤 Search Goalkeepers")

    q = st.text_input("Enter goalkeeper name (partial or full, case-insensitive)", key="gk_search")
    if q:
        res = goalkeepers[goalkeepers["short_name"].str.contains(q, case=False, na=False)]
        if res.empty:
            st.warning("No goalkeeper found.")
        else:
            st.success(f"Found {len(res)} goalkeeper(s)")
            st.dataframe(res.reset_index(drop=True), use_container_width=True)

# ----------------------------
# Player Profile Page (charts + info)
# ----------------------------
elif menu == "Player Profile Page":
    st.subheader("📋 Player Profile Page")

    q = st.text_input("Enter player name (partial or full, case-insensitive) for profile", key="profile_search")
    if q:
        matches = df_all[df_all["short_name"].str.contains(q, case=False, na=False)].copy()

        if matches.empty:
            st.warning("No player found.")
        else:
            # If multiple matches allow user to pick
            matches["display_label"] = matches["short_name"] + " — " + matches.get("club", "").fillna("") + " — " + matches.get("player_positions", "")
            pick = st.selectbox("Select player", matches["display_label"].tolist())
            player = matches[matches["display_label"] == pick].iloc[0]

            # Basic info & layout
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {player.get('long_name', player.get('short_name'))}")
                st.write(f"**Short name:** {player.get('short_name', '')}")
                st.write(f"**Club:** {player.get('club', 'N/A')}")
                st.write(f"**Nationality:** {player.get('nationality', 'N/A')}")
                st.write(f"**Positions:** {player.get('player_positions', 'N/A')}")
                st.write(f"**Age:** {player.get('age', 'N/A')}")
                st.write(f"**Height (cm):** {player.get('height_cm', 'N/A')}")
                st.write(f"**Weight (kg):** {player.get('weight_kg', 'N/A')}")
            with col2:
                st.write("")  # spacing
                st.metric("Overall", player.get("overall", "N/A"))
                st.metric("Potential", player.get("potential", "N/A"))
                if "final_score" in player.index:
                    st.metric("Final Score", round(player.get("final_score", 0), 2))

            # Attribute bar chart (using Streamlit native)
            st.markdown("#### 🔢 Key Attributes (Bar)")
            bar_attrs = {
                "Overall": float(player.get("overall") or 0),
                "Potential": float(player.get("potential") or 0),
                "Dribbling": float(player.get("dribbling") or 0),
                "Passing": float(player.get("passing") or 0),
                "Shooting": float(player.get("shooting") or 0),
            }
            bar_df = pd.DataFrame.from_dict(bar_attrs, orient="index", columns=["score"])
            st.bar_chart(bar_df["score"])

            # Radar chart (Plotly)
            st.markdown("#### 🕸 Skill Radar")
            radar_categories = ["Dribbling", "Passing", "Shooting", "Overall", "Potential"]
            radar_values = [
                float(player.get("dribbling") or 0),
                float(player.get("passing") or 0),
                float(player.get("shooting") or 0),
                float(player.get("overall") or 0),
                float(player.get("potential") or 0),
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=radar_values, theta=radar_categories, fill="toself", name=player.get("short_name")))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Trend chart: simple two-point line overall vs potential
            st.markdown("#### 📈 Mini Trend (Overall vs Potential)")
            trend_df = pd.DataFrame({
                "metric": ["Overall", "Potential"],
                "value": [float(player.get("overall") or 0), float(player.get("potential") or 0)]
            }).set_index("metric")
            st.line_chart(trend_df)

            # Raw data / download
            st.markdown("#### Raw player record")
            st.write(player.to_frame().T)
            csv = player.to_frame().T.to_csv(index=False)
            st.download_button("Download player CSV", csv, file_name=f"{player.get('short_name')}_profile.csv")

# ----------------------------
# Filters (club / nationality / age / position)
# ----------------------------
elif menu == "Filters (Club / Nation / Age / Position)":
    st.subheader("🔎 Advanced Filters (Outfield & GK)")

    df = df_all.copy()

    # Prepare filter choices (limit size for performance)
    clubs = ["All"] + sorted(df["club"].dropna().unique().tolist())
    nations = ["All"] + sorted(df["nationality"].dropna().unique().tolist())
    positions = ["All"] + sorted(df["player_positions"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        club = st.selectbox("Club", clubs)
        nation = st.selectbox("Nationality", nations)
    with col2:
        position = st.selectbox("Position (contains)", positions)
        age_min, age_max = st.slider("Age range", 15, 50, (16, 40))
    with col3:
        sort_by = st.selectbox("Sort by", ["final_score", "overall", "potential", "age", "value_eur"], index=1)
        ascending = st.checkbox("Ascending order", value=False)

    if club != "All":
        df = df[df["club"] == club]
    if nation != "All":
        df = df[df["nationality"] == nation]
    if position != "All":
        df = df[df["player_positions"].astype(str).str.contains(position, na=False)]
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(-1)
    df = df[(df["age"] >= age_min) & (df["age"] <= age_max)]

    # Ensure sort_by exists
    if sort_by not in df.columns:
        st.warning(f"Sort-by column '{sort_by}' not present in dataset. Showing unsorted results.")
    else:
        df = df.sort_values(by=sort_by, ascending=ascending)

    st.success(f"Found {len(df)} player(s) matching filters")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    with st.expander("Download filtered list as CSV"):
        st.download_button("Download CSV", df.to_csv(index=False), file_name="filtered_players.csv")

