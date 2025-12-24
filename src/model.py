
from xgboost import XGBClassifier
import pandas as pd

FEATURES = [
    'is_home', 
    'roll_off_epa', 'roll_off_cpoe', 'roll_def_epa',
    'roll_off_epa_opp', 'roll_off_cpoe_opp', 'roll_def_epa_opp'
]

def train_model(df_train):
    """Trains the XGBoost model."""
    df_clean = df_train.dropna(subset=FEATURES)
    X = df_clean[FEATURES]
    y = df_clean['win']
    
    model = XGBClassifier(n_estimators=60, max_depth=3, eval_metric='logloss')
    model.fit(X, y)
    return model

def predict_games(model, df_future_matchups, latest_stats):
    """Predicts future games using the trained model and latest stats."""
    
    preds = []
    
    # We expect df_future_matchups to be the schedule dataframe
    for _, row in df_future_matchups.iterrows():
        home = row['home_team']
        away = row['away_team']
        week = row['week']
        
        # Get Latest Stats for Home and Away
        h_stats = latest_stats[latest_stats['team'] == home]
        a_stats = latest_stats[latest_stats['team'] == away]
        
        if h_stats.empty or a_stats.empty:
            continue
            
        # Build feature row (Home perspective)
        # We manually construct it to ensure correct order
        feat_dict = {
            'is_home': [1],
            'roll_off_epa': [h_stats['roll_off_epa'].values[0]],
            'roll_off_cpoe': [h_stats['roll_off_cpoe'].values[0]],
            'roll_def_epa': [h_stats['roll_def_epa'].values[0]],
            'roll_off_epa_opp': [a_stats['roll_off_epa'].values[0]],
            'roll_off_cpoe_opp': [a_stats['roll_off_cpoe'].values[0]],
            'roll_def_epa_opp': [a_stats['roll_def_epa'].values[0]]
        }
        
        X_row = pd.DataFrame(feat_dict)
        prob = model.predict_proba(X_row[FEATURES])[:, 1][0]
        
        preds.append({
            'Week': week,
            'Home': home,
            'Away': away,
            'Home Win Prob': prob,
            'Winner': home if prob > 0.50 else away,
            'Confidence': prob if prob > 0.50 else 1 - prob
        })
        
    return pd.DataFrame(preds)
