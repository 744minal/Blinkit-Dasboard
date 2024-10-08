# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('blinkit_data.csv')

# 1. Handling Missing Values
# Fill missing values for numerical columns with the median and for categorical with the mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=[object]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 2. Removing Duplicates
df.drop_duplicates(inplace=True)

# 3. Outlier Detection and Treatment
# Define a function to cap outliers
def cap_outliers(df, col, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = df[col].quantile(lower_quantile)
    upper_bound = df[col].quantile(upper_quantile)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

# Applying outlier capping for relevant numerical columns like 'Total Sales', 'Ratings'
for col in ['Total Sales', 'Average Sales', 'Ratings']:
    df = cap_outliers(df, col)

# 4. Correct Data Formatting
# Ensure numerical columns are in the correct format
df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')
df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')

# Ensure date columns are correctly formatted
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 5. Data Consistency
# Standardize categorical values (e.g., fixing typos)
df['Fat Content'] = df['Fat Content'].str.lower().replace({
    'lowfat': 'low fat', 
    'reg': 'regular', 
    'LF': 'low fat'
})

# Check for unique values in a categorical column and manually fix any inconsistencies
df['Outlet Size'].replace({'Small': 'small', 'Medium': 'medium', 'High': 'large'}, inplace=True)

# 6. Ensuring Proper Data Range
# Ensure numerical columns are within reasonable ranges
df = df[df['Total Sales'] >= 0]  # Total sales should not be negative
df = df[(df['Ratings'] >= 1) & (df['Ratings'] <= 5)]  # Ratings should be between 1 and 5

# 7. Handling Missing or Incomplete Date Entries
df = df.dropna(subset=['Date'])  # Drop rows where date is missing

# Save the cleaned dataset
df.to_csv('cleaned_blinkit_data.csv', index=False)

print("Data cleaning completed and saved as 'cleaned_blinkit_data.csv'")
