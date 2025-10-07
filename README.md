## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING:
 ```

# Feature Encoding and Feature Transformation Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

# STEP 1: Read the given Data
print("STEP 1: Reading the data...")
try:
    # Try to load the main dataset
    df = pd.read_csv('Data_to_Transform.csv')
    print("Data_to_Transform.csv loaded successfully")
except FileNotFoundError:
    try:
        # Try alternative file name
        df = pd.read_csv('Encoding Data.csv')
        print("Encoding Data.csv loaded successfully")
    except FileNotFoundError:
        try:
            # Try another alternative
            df = pd.read_csv('data.csv')
            print("data.csv loaded successfully")
        except FileNotFoundError:
            print("Error: No data file found. Please check file names.")
            # Create sample data for demonstration if no file is found
            df = pd.DataFrame({
                'Age': [25, 30, 35, np.nan, 45],
                'Income': [50000, 60000, np.nan, 75000, 90000],
                'Gender': ['Male', 'Female', 'Male', 'Female', np.nan],
                'Education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor'],
                'Satisfaction': ['High', 'Medium', 'Low', 'High', 'Medium']
            })
            print("Created sample data for demonstration")

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# STEP 2: Clean the Data Set using Data Cleaning Process
print("\nSTEP 2: Cleaning the data...")

# 2.1 Handle missing values
print("Handling missing values...")
# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Create a copy of the dataframe for cleaning
df_clean = df.copy()

# Fill numeric missing values with mean
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        mean_val = df_clean[col].mean()
        df_clean[col].fillna(mean_val, inplace=True)
        print(f"  - Filled missing values in '{col}' with mean: {mean_val:.2f}")

# Fill categorical missing values with mode
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"  - Filled missing values in '{col}' with mode: '{mode_val}'")

# 2.2 Check for duplicates
duplicate_count = df_clean.duplicated().sum()
if duplicate_count > 0:
    print(f"Removing {duplicate_count} duplicate rows...")
    df_clean = df_clean.drop_duplicates()
else:
    print("No duplicates found.")

# 2.3 Check for outliers in numeric columns
print("Checking for outliers in numeric columns...")
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  - '{col}' has {outliers} outliers")
        # For this exercise, we'll cap outliers rather than removing them
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        print(f"    Outliers in '{col}' have been capped at [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\nCleaned data summary:")
print(f"Shape after cleaning: {df_clean.shape}")
print("Missing values after cleaning:")
print(df_clean.isnull().sum())

# STEP 3: Apply Feature Encoding for the features in the data set
print("\nSTEP 3: Applying Feature Encoding...")

# Create a copy for encoding
df_encoded = df_clean.copy()

# 3.1 Label Encoding
print("Applying Label Encoding...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[f'{col}_label'] = le.fit_transform(df_clean[col])
    # Store mapping for interpretation
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    label_encoders[col] = mapping
    print(f"  - Label encoded '{col}' with mapping: {mapping}")

# 3.2 One-Hot Encoding
print("\nApplying One-Hot Encoding...")
df_onehot = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"  - Shape after one-hot encoding: {df_onehot.shape}")
print(f"  - New columns added: {df_onehot.shape[1] - df_clean.shape[1]}")

# 3.3 Combine both encoding methods
# Keep original columns and add label encoded columns
df_encoded_final = df_clean.copy()
for col in categorical_cols:
    df_encoded_final[f'{col}_label'] = df_encoded[f'{col}_label']

print("\nFinal encoded data (first 5 rows):")
print(df_encoded_final.head())

# STEP 4: Apply Feature Transformation for the features in the data set
print("\nSTEP 4: Applying Feature Transformation...")

# Create a copy for transformation
df_transformed = df_encoded_final.copy()

# 4.1 Standardization (Z-score normalization)
print("Applying Standardization...")
scaler = StandardScaler()
for col in numeric_cols:
    df_transformed[f'{col}_scaled'] = scaler.fit_transform(df_encoded_final[[col]])
    mean = df_encoded_final[col].mean()
    std = df_encoded_final[col].std()
    print(f"  - Standardized '{col}' (mean={mean:.2f}, std={std:.2f})")

# 4.2 Min-Max Scaling
print("\nApplying Min-Max Scaling...")
min_max_scaler = MinMaxScaler()
for col in numeric_cols:
    df_transformed[f'{col}_minmax'] = min_max_scaler.fit_transform(df_encoded_final[[col]])
    min_val = df_encoded_final[col].min()
    max_val = df_encoded_final[col].max()
    print(f"  - Min-Max scaled '{col}' (min={min_val:.2f}, max={max_val:.2f})")

# 4.3 Log Transformation (for positive skewed data)
print("\nApplying Log Transformation...")
for col in numeric_cols:
    # Check if all values are positive
    if (df_encoded_final[col] > 0).all():
        df_transformed[f'{col}_log'] = np.log(df_encoded_final[col])
        print(f"  - Log transformed '{col}'")
    else:
        print(f"  - Skipped log transformation for '{col}' (contains zero or negative values)")

# Make sure to import numpy if not already imported
# import numpy as np

```
# RESULT/OUTPUT:
<img width="1242" height="699" alt="Screenshot 2025-10-07 213126" src="https://github.com/user-attachments/assets/9a9e7fd9-c966-4df1-b7cc-708a8613bcbd" />
<img width="1257" height="620" alt="Screenshot 2025-10-07 213140" src="https://github.com/user-attachments/assets/c8a43d75-02c4-46ae-b7af-73e412c9852a" />
<img width="1233" height="717" alt="Screenshot 2025-10-07 213152" src="https://github.com/user-attachments/assets/e852e27d-ec25-4445-9933-663d7ff6188e" />




       
