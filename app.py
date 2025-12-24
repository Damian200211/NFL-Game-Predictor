
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_data
from src.features import calculate_weekly_stats, calculate_rolling_stats, prepare_matchups
from src.model import train_model, predict_games

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NFL Game Predictor",
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg", width=100)
    st.title("Settings")
    season = st.selectbox("Select Season", [2025, 2024, 2023])
    current_week = st.slider("Current Week (Simulation)", 1, 18, 17)
    
    st.info("This model learns incrementally. As the season progresses, it gets smarter.")

# --- MAIN APP ---
st.title(f" NFL Predictor (Season {season})")
st.markdown("""
**Objective**: Predict game winners using advanced efficiency stats (EPA/CPOE).
**Model**: XGBoost trained on a Walk-Forward Validation basis.
""")

# 1. LOAD DATA
with st.spinner("Downloading Data..."):
    df_pbp, df_schedule = load_data(season)
    
# 2. PROCESS STATS
weekly_stats = calculate_weekly_stats(df_pbp)
rolling_stats = calculate_rolling_stats(weekly_stats)
full_dataset = prepare_matchups(df_schedule, rolling_stats)

# Split into Past (Training) and Future (Prediction) based on slider
# Note: 'result' might be NaN for future games, but we use the slider to simulate "known" world.
# We treat everything BEFORE current_week as Training.
# We treat current_week as the Target for prediction.

past_games = full_dataset[full_dataset['week'] < current_week].dropna(subset=['result', 'roll_off_epa'])
games_to_predict = df_schedule[df_schedule['week'] == current_week]

# 3. TRAIN
model = train_model(past_games)

# 4. PREDICT
# We need the "Latest Stats" as of the previous week
latest_stats = rolling_stats[rolling_stats['week'] < current_week].groupby('team').tail(1)
predictions = predict_games(model, games_to_predict, latest_stats)

# --- TABS ---
tab1, tab2, tab3 = st.tabs([" Predictions", " Team Power Rankings", " Model Insights"])

with tab1:
    st.subheader(f"Week {current_week} Predictions")
    
    if predictions.empty:
        st.warning("No games found or not enough data to predict.")
    else:
        # Display as a clean table
        # Format columns
        display_df = predictions.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
        
        # Color coding for high confidence
        def highlight_conf(val):
            conf = float(val.strip('%'))
            color = 'green' if conf > 80 else 'black'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            display_df[['Home', 'Away', 'Winner', 'Confidence']],
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.subheader("Team Efficiency (Latest)")
    # Scatter Plot: Offense vs Defense
    # Invert Def EPA (Lower is Better) for visualization purposes? 
    # Usually: Top Right = Good Offense, Good Defense (if we flip axis)
    
    # Latest averages
    chart_data = latest_stats[['team', 'roll_off_epa', 'roll_def_epa']].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=chart_data, x='roll_off_epa', y='roll_def_epa', ax=ax, s=100)
    
    # Add labels
    for i in range(chart_data.shape[0]):
        ax.text(
            chart_data.roll_off_epa.iloc[i]+0.002, 
            chart_data.roll_def_epa.iloc[i], 
            chart_data.team.iloc[i], 
            fontsize=9
        )
        
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title("Offensive EPA vs Defensive EPA (Expected Points Added)")
    ax.set_xlabel("Offense EPA/Play (Higher is Better)")
    ax.set_ylabel("Defense EPA/Play Allowed (Lower is Better)")
    ax.invert_yaxis() # Top is good defense
    
    st.pyplot(fig)

with tab3:
    st.subheader("How it works")
    st.code("""
    Features Used:
    1. Rolling Offensive EPA (Last 4 Games)
    2. Rolling Defensive EPA
    3. Home Field Advantage
    4. Opponent Strength
    """, language="python")
    
    st.metric("Training Set Size", f"{len(past_games)} Games")
