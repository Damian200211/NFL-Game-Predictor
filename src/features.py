
import pandas as pd
import numpy as np

def calculate_weekly_stats(df_pbp):
    """Calculates weekly offensive and defensive EPA/CPOE."""
    mask = (df_pbp['pass'] == 1) | (df_pbp['rush'] == 1)
    df_plays = df_pbp[mask].copy()

    # Offensive Stats
    off = df_plays.groupby(['season', 'week', 'posteam'])[['epa', 'cpoe']].mean().reset_index()
    off.rename(columns={'posteam': 'team', 'epa': 'off_epa', 'cpoe': 'off_cpoe'}, inplace=True)

    # Defensive Stats
    defs = df_plays.groupby(['season', 'week', 'defteam'])[['epa']].mean().reset_index()
    defs.rename(columns={'defteam': 'team', 'epa': 'def_epa'}, inplace=True)

    # Merge
    return pd.merge(off, defs, on=['season', 'week', 'team'], how='outer')

def calculate_rolling_stats(weekly_stats, window=4):
    """Calculates rolling averages for entering match stats."""
    weekly_stats = weekly_stats.sort_values(['team', 'season', 'week'])
    
    cols = ['off_epa', 'off_cpoe', 'def_epa']
    roll_cols = ['roll_off_epa', 'roll_off_cpoe', 'roll_def_epa']
    
    def get_roll(group):
        # Shift 1 to ensure pre-game stats
        return group[cols].shift(1).rolling(window=window, min_periods=1).mean()
        
    weekly_stats[roll_cols] = weekly_stats.groupby('team', group_keys=False).apply(get_roll)
    return weekly_stats

def prepare_matchups(df_schedule, weekly_stats):
    """Merges stats onto the schedule to create a modeling dataset."""
    
    # Filter only relevant columns to avoid warnings
    cols_to_use = ['season', 'week', 'home_team', 'away_team', 'result']
    # Ensure result exists (might be missing in future games)
    if 'result' not in df_schedule.columns:
        df_schedule['result'] = np.nan
        
    games = df_schedule[cols_to_use].copy()
    
    # Home Perspective
    home = games.copy().rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    home['is_home'] = 1
    home['win'] = (home['result'] > 0).astype(int)
    
    # Away Perspective
    away = games.copy().rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    away['is_home'] = 0
    away['win'] = (away['result'] < 0).astype(int)
    
    matchups = pd.concat([home, away], ignore_index=True)
    
    # Merge Stats
    roll_cols = ['roll_off_epa', 'roll_off_cpoe', 'roll_def_epa']
    
    # Own Stats
    df_final = pd.merge(matchups, weekly_stats[['season', 'week', 'team'] + roll_cols],
                        on=['season', 'week', 'team'], how='left')
                        
    # Opponent Stats
    opp_stats = weekly_stats[['season', 'week', 'team'] + roll_cols].rename(columns={'team': 'opponent'})
    for col in roll_cols:
        opp_stats.rename(columns={col: col + '_opp'}, inplace=True)
        
    df_final = pd.merge(df_final, opp_stats, on=['season', 'week', 'opponent'], how='left')
    
    return df_final
