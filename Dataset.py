import pandas as pd
import json

csv_path = r"C:\Users\hp\Downloads\sentences.csv"
json_path = r"C:\Users\hp\Downloads\lan_to_language.json"

# Load large CSV (
df = pd.read_csv(csv_path)

# Load language mapping


with open(json_path, "r", encoding="utf-8") as f:
    lang_map = json.load(f)

df["language"] = df["lan_code"].map(lang_map)

# IMPORTANT: Random sample, not head()
df_sample = df.sample(n=500000, random_state=42)

df_sample.to_csv(
    r"C:\Users\hp\Downloads\sample_sentences.csv",
    index=False
)

print("Random sample saved:", df_sample.shape)



# Code Explanation -

# Loads a large CSV file containing sentences
# Loads a JSON file that maps language codes to full language names
# Replaces language codes in the dataset with readable language names
# Randomly selects 500,000 rows from the dataset
# Uses a fixed random seed to ensure reproducible sampling
# Saves the sampled data into a new CSV file
# Prints a confirmation showing the size of the saved dataset

# This code reads a big CSV file, replaces language codes with real names, randomly picks 500,000 rows, and saves them as a new CSV file.