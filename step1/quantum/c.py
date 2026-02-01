# from: epsilon,confidence_level,shots,grover_calls,VaR_theoretical,VaR_predicted,mu,sigma,alpha_iqae
# to: method,dist,var_alpha,queries,epsilon,alpha_fail,p_hat,true_p,error

import pandas as pd
import os

method = "quantum"
dist = "normal"
var_alpha = 0.05

# p_hat = VaR_predicted
# true_p = VaR_theoretical
# error = |p_hat - true_p|
# queries = grover_calls

FROM_CSV = "./results/var_sweep_04.csv"
TO_CSV = "./data.csv"

def convert_csv():
    # Read the source CSV
    df = pd.read_csv(FROM_CSV)
    
    # Create the new dataframe with the required columns
    converted_df = pd.DataFrame({
        'method': method,
        'dist': dist,
        'var_alpha': var_alpha,
        'queries': df['grover_calls'],
        'epsilon': df['epsilon'],
        'alpha_fail': df['alpha_iqae'],  # Using alpha_iqae as alpha_fail
        'p_hat': df['VaR_predicted'],
        'true_p': df['VaR_theoretical'],
        'error': abs(df['VaR_predicted'] - df['VaR_theoretical'])
    })
    
    # Save to the target CSV
    converted_df.to_csv(TO_CSV, index=False)
    print(f"Converted {len(converted_df)} rows from {FROM_CSV} to {TO_CSV}")

if __name__ == "__main__":
    convert_csv()