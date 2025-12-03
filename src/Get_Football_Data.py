import pandas as pd
import numpy as np
import requests
import io
import os

# Create CSV DATA script
# This file creates data as a csv file for use in other parts of the project


# Get data for last 15 seasons
SEASONS = [f"{i:02d}{i+1:02d}" for i in range(10, 25)]
LEAGUES = {"EPL": "E0", "LaLiga": "SP1", "Bundesliga": "D1"}
OUTPUT_FOLDER = "../Data"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/football_training_data.csv"

def download_data():
    """
    Download raw match stats from football-data.co.uk for multiple seasons
    """
    all_data = []
    print(f"--- DOWNLOADING DATA (2010 - 2025) ---")
    base_url = "http://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    
    for season in SEASONS:
        for league_name, league_code in LEAGUES.items():
            url = base_url.format(season=season, league=league_code)
            print(f"Downloading data for {league_name} season {season} from {url}")
            try:
                response = requests.get(url)
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
    print("\nCALCULATING TEAM FORM (Rolling Last 5 Games)")
    
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
    final_df = final_df.dropna(subset=['Home_Form_Pts', 'Away_Form_Pts'])
    
    return final_df
    


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Import Data
    raw_df = download_data()
    
    # Calculate Form
    long_stats = calculate_form(raw_df)
    
    # Reshape
    final_data = reshape_to_match_row(long_stats, raw_df)
    
    # Only keep columns that actually exist (Old seasons might miss some odds)
    final_cols = [c for c in final_data if c in final_data.columns]
    
    final_data[final_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Saved {len(final_data)} matches to {OUTPUT_FILE}")