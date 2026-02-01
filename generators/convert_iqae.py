import pandas as pd
import csv


mu = 0.15                      # Mean daily return (15%)
sigma = 0.20                   # Daily volatility (20%)
confidence_level = 0.95        # VaR confidence level

# Multi-day and distribution settings
T = 5                          # Number of days for multi-day VaR
dist = "skewnorm"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.6                      # AR(1) correlation coefficient


CSV_HEADERS_OUTPUT = [
    "Epsilon",
    "N",
    "VaR_prediction",
    "VaR_theoretical",
    "mu", "sigma", "confidence_level", "T", "dist", "df", "skew_alpha", "rho"
]

CSV_HEADERS_INPUT = [
	"epsilon", # -> "Epsilon"
	"alpha", # -> nothing, not used?
	"a_true", # -> nothing, not used?
	"a_hat_mean", # -> VaR_prediction
	"a_hat_std", # -> nothing, not used?
	"abs_error_mean", # -> nothing, not used?
	"abs_error_std", # -> nothing, not used?
	"ci_low", # -> not used
	"ci_high", # -> not used
	"shots_total_mean", # 
	"grover_calls_mean", # -> N
	"ks_used", # -> not used
	"runs" # -> not used
]

OUTPUT = '../graphs/data/iqae_converted.csv'
INPUT = './raw/iqae.csv'

df_input = pd.read_csv(INPUT)
output_rows = []

for index, row in df_input.iterrows():
	new_row = {
		"Epsilon": row["epsilon"],
		"N": row["grover_calls_mean"],
		"VaR_prediction": row["a_hat_mean"],
		"VaR_theoretical": None,  # Placeholder for theoretical VaR calculation
		"mu": mu,
		"sigma": sigma,
		"confidence_level": confidence_level,
		"T": T,
		"dist": dist,
		"df": df,
		"skew_alpha": skew_alpha,
		"rho": rho
	}
	output_rows.append(new_row)

with open(OUTPUT, mode='w', newline='') as file:
	writer = csv.DictWriter(file, fieldnames=CSV_HEADERS_OUTPUT)
	writer.writeheader()
	for row in output_rows:
		writer.writerow(row)