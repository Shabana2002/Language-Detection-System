import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\sample_sentences.csv")
print("Original shape:", df.shape)

# Function to sample or take all if below 4000
def sample_language(group, n=4000):
    if len(group) > n:
        return group.sample(n=n, random_state=42)
    else:
        return group

# Apply sampling without future warning
df_balanced = (
    df.groupby("language", group_keys=False, as_index=False)
      .apply(sample_language, n=4000)
)

# Verify balance
print(df_balanced["language"].value_counts())
print("Balanced shape:", df_balanced.shape)

# Save balanced dataset
df_balanced.to_csv(
    r"C:\Users\hp\Downloads\language_detection_balanced_all.csv",
    index=False
)


print("Balanced dataset saved successfully!")
print(df["language"].unique())




# Code Explanation -

# Imports pandas to work with CSV data
# Loads the dataset from sample_sentences.csv
# Prints the original number of rows and columns
# Defines a function that:
# (1). Randomly selects up to 4000 rows per language
# (2). Keeps all rows if a language has fewer than 4000 samples
# Groups the data by language
# Applies the sampling function to each language group
# Creates a balanced dataset where no language dominates
# Prints the number of samples per language to verify balance
# Prints the final shape of the balanced dataset
# Saves the balanced dataset to a new CSV file
# Confirms successful saving
# Displays:
# (1). Final count of samples per language
# (2). All unique languages present in the original dataset


# This code makes sure every language has at most 4000 sentences so no language dominates the dataset.
