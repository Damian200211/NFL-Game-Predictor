
import nflreadpy as nfl
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600*24) # Cache for 24 hours
def load_data(season):
    """Loads PBP and Schedule data for the given season."""
    print(f"Loading data for {season}...")
    
    # 1. PBP Data (Stats)
    df_pbp = nfl.load_pbp(seasons=[season]).to_pandas()
    
    # 2. Schedule Data (Matchups)
    df_schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    
    return df_pbp, df_schedule
