def calculate_advanced_dominance(df):
    """
    CALCULATES 'TRUE DOMINANCE' (0 to 1 Scale)
    
    Since we don't have Possession %, we use a weighted proxy of:
    1. Total Shots Ratio (TSR) - Volume of attack
    2. Shots on Target Ratio (SoTR) - Quality of attack
    3. Corner Ratio - Territorial pressure
    """
    print("\nCALCULATING ADVANCED DOMINANCE METRICS...")
    
    # Avoid crashing if a game has 0 shots (boring, but happens)
    def safe_ratio(a, b):
        return a / (a + b).replace(0, 1)
    
    # Metrics
    
    df['TSR'] = safe_ratio(df['HS'], df['AS'])
    df['SoTR'] = safe_ratio(df['HST'], df['AST'])
    df['Corner_Ratio'] = safe_ratio(df['HC'], df['AC'])
    
    w_shots = 0.4
    w_sot = 0.4
    w_corners = 0.2
    
    df['Dominance'] = (df['TSR'] * w_shots) + \
                      (df['SoTR'] * w_sot) + \
                      (df['Corner_Ratio'] * w_corners)
    
    # Red Card penalyty
    # If a team gets a red card, their 'Dominance' naturally drops.
   
    df['Dominance'] -= (df['HR'] * 0.1) 
    df['Dominance'] += (df['AR'] * 0.1)
    
    # Clip values to ensure they stay between 0 and 1
    df['Dominance'] = df['Dominance'].clip(0, 1)
    
    return df