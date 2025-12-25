import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import joblib

# Read data
df = pd.read_csv("diabetes_dataset.csv")

# Convert non-int input to ints 
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['smoking_history'] = df['smoking_history'].map({'No Info': 0, 'never': 1, 'current': 2, 'not current': 3, 'former': 3})

# Remove NaN values
df = df.dropna()

# Assign input columns
X = df.drop("diabetes", axis=1)

# Assign target (output) column
y = df["diabetes"]

# Use 80% of data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalizes data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model on data
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

# Save model 
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("done")
