import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/players_20.csv")

# Select required columns
columns_needed = [
    "short_name", "long_name", "age", "height_cm", "weight_kg",
    "nationality", "club", "overall", "potential",
    "value_eur", "wage_eur", "player_positions",
    "dribbling", "passing", "shooting"
]

df = df[columns_needed]

# Create final_score for outfield players only
df_outfield = df[~df["player_positions"].str.contains("GK")].copy()
df_outfield["final_score"] = (
    df_outfield["overall"] * 0.5 +
    df_outfield["potential"] * 0.5 +
    df_outfield["dribbling"] * 0.1 +
    df_outfield["passing"] * 0.1 +
    df_outfield["shooting"] * 0.1
)

# Goalkeepers (they won’t have final_score)
df_gk = df[df["player_positions"].str.contains("GK")]

# Save files
joblib.dump(df_outfield, "model/players.pkl")
joblib.dump(df_gk, "model/gk.pkl")

print("Training complete. Files saved: players.pkl, gk.pkl")
