import pandas as pd
import numpy as np
import requests
import io
import os
import warnings
from datetime import datetime
import addData


# Create CSV DATA script
# This file creates data as a csv file for use in other parts of the project


# Get data for last 15 seasons
SEASONS = [f"{i:02d}{i+1:02d}" for i in range(10, 26)]
LEAGUES = {"EPL": "E0"}
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(script_dir, "..", "Data")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "football_training_data.csv")

# SILENCE WARNINGS (Clean up terminal output)
warnings.filterwarnings("ignore", category=UserWarning)

def download_data():
    """
    Download raw match stats from football-data.co.uk for multiple seasons
    """
    all_data = []
    print(f"--- DOWNLOADING DATA (2010 - 2025) ---")
    base_url = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for season in SEASONS:
        for league_name, league_code in LEAGUES.items():
            url = base_url.format(season=season, league=league_code)
            print(f"Downloading data for {league_name} season {season} from {url}")
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.content.decode('utf-8', errors='ignore')))
                    df['Season'] = season
                    df['League'] = league_name
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    all_data.append(df)
                else:
                    print(f"Failed to download data for {league_name} season {season}")
            except Exception as e:
                print(f"Error: {e}")
    
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        cols_to_check = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HST', 'AST']
        existing_cols = [c for c in cols_to_check if c in master_df.columns]
        master_df = master_df.dropna(subset=existing_cols)
        return master_df.sort_values('Date')
    else:
        return pd.DataFrame()
    
def calculate_form(df, form_window=5):
    """
    Creates the team's form (From last 5 matches)
    """
    if df.empty: return pd.DataFrame()
    print("\nCALCULATING TEAM FORM (Rolling Last 5 Games)")
    
    # Standardize columns to avoid KeyErrors if HST/AST are missing in older data
    if 'HST' not in df.columns: df['HST'] = 0
    if 'AST' not in df.columns: df['AST'] = 0
    
    # One row per TEAM per match
    home = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HST', 'AST']].copy()
    home = home.rename(columns={'HomeTeam': 'Team', 'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst', 'HST': 'ShotsFor', 'AST': 'ShotsAgainst'})
    home['IsHome'] = 1

    # Points: 3 for Win (H), 1 for Draw (D), 0 for Loss (A)
    home['Points'] = np.where(home['FTR'] == 'H', 3, np.where(home['FTR'] == 'D', 1, 0))

    away = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AST', 'HST']].copy()
    away = away.rename(columns={'AwayTeam': 'Team', 'FTAG': 'GoalsFor', 'FTHG': 'GoalsAgainst', 'AST': 'ShotsFor', 'HST': 'ShotsAgainst'})
    away['IsHome'] = 0
    # Points: 3 for Win (A), 1 for Draw (D), 0 for Loss (H)
    away['Points'] = np.where(away['FTR'] == 'A', 3, np.where(away['FTR'] == 'D', 1, 0))

    # Combine and Sort
    team_stats = pd.concat([home, away]).sort_values(['Team', 'Date'])

    # CALCULATE ROLLING STATS
    # We use .shift(1) so we only know form BEFORE the match starts
    grouped = team_stats.groupby('Team')

    # Rolling Metrics
    metrics = ['Points', 'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst']
    
    for m in metrics:
        team_stats[f'Form_{m}_{form_window}'] = grouped[m].transform(
            lambda x: x.shift(1).rolling(form_window, min_periods=1).mean()
        )

    return team_stats

def reshape_to_match_row(long_df, original_df):
    """
    Merge the calculated form stats back onto the original Home vs Away match rows.
    """
    print("\nMERGING DATA")
    # Get the form columns
    form_cols = [c for c in long_df.columns if c.startswith('Form_')]
    
    home_stats = long_df[long_df['IsHome'] == 1][['Date', 'Team'] + form_cols]
    away_stats = long_df[long_df['IsHome'] == 0][['Date', 'Team'] + form_cols]
    
    # Prefix columns with Home_ and Away_
    home_stats.columns = ['Date', 'HomeTeam'] + [f"Home_{c}" for c in form_cols]
    away_stats.columns = ['Date', 'AwayTeam'] + [f"Away_{c}" for c in form_cols]

    # Merge into original
    final_df = pd.merge(original_df, home_stats, on=['Date', 'HomeTeam'], how='left')
    final_df = pd.merge(final_df, away_stats, on=['Date', 'AwayTeam'], how='left')
    
    # Drop first few weeks where form is NaN
    points_col = [c for c in final_df.columns if 'Home_Form_Points' in c]
    
    if points_col:
        final_df = final_df.dropna(subset=points_col)
    
    return final_df

def add_time_weighting(df):
    """
    QUANT FEATURE: Time Decay
    Assigns a weight to each match based on how long ago it happened.
    Recent matches = 1.0
    Old matches = Closer to 0.0
    """
    print("\nAPPLYING TIME DECAY WEIGHTS")
    current_date = pd.to_datetime(datetime.now())
    
    # Calculate days ago
    df['DaysAgo'] = (current_date - df['Date']).dt.days
    
    # Exponential Decay Formula: Weight = e^(-lambda * days)
    # Lambda of 0.0005 means a game 4 years ago has ~50% weight of today
    decay_rate = 0.0005
    df['Time_Weight'] = np.exp(-decay_rate * df['DaysAgo'])
    
    # Normalize weights so they are typically between 0 and 1
    df['Time_Weight'] = df['Time_Weight'].round(4)
    
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Import Data
    raw_df = download_data()
    
    # Calculate Form
    long_stats = calculate_form(raw_df)
    
    # Reshape
    final_data = reshape_to_match_row(long_stats, raw_df)
    final_data = add_time_weighting(final_data)
    final_data = addData.calculate_advanced_dominance(final_data)
    
    # Only keep columns that actually exist (Old seasons might miss some odds)
    final_cols = [c for c in final_data if c in final_data.columns]
    
    final_data[final_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Saved {len(final_data)} matches to {OUTPUT_FILE}")