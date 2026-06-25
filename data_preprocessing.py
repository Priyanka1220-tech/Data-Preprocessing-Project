import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check data types
print("\nData Types:")
print(df.dtypes)

print("Missing values before:")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing values after:")
print(df.isnull().sum())

print("\nUpdated Dataset:")
print(df)



# Handle missing values first
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

# Fill missing numerical values with mean
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill missing categorical values with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# One-Hot Encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Display the encoded dataset
print("Dataset after encoding:")
print(df.head())

print("\nColumns:")
print(df.columns)
from sklearn.preprocessing import StandardScaler

# Select numerical columns (except the target column)
num_cols = ['Age', 'Salary', 'Experience']

# Create scaler object
scaler = StandardScaler()

# Scale the numerical columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# Display the scaled dataset
print("Dataset after scaling:")
print(df.head())
# Create a new feature
df['Experience_Age_Ratio'] = df['Experience'] / df['Age']

# Display the updated dataset
print("Dataset after Feature Engineering:")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale only the training data to prevent data leakage
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display shapes
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
# Convert scaled training data back to DataFrame
X_train_df = pd.DataFrame(X_train, columns=X.columns)

# Add the target column back
final_df = pd.concat(
    [X_train_df, y_train.reset_index(drop=True)],
    axis=1
)

# Save as CSV
final_df.to_csv("cleaned_dataset.csv", index=False)

print("Cleaned dataset saved successfully as 'cleaned_dataset.csv'")
