import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/tickets.csv")

# Features
X = data["ticket"]

# Targets
y_category = data["category"]
y_priority = data["priority"]

# Convert text → numbers
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X_vectorized, y_category, test_size=0.2, random_state=42
)

X_train2, X_test2, y_pri_train, y_pri_test = train_test_split(
    X_vectorized, y_priority, test_size=0.2, random_state=42
)

# Train category model
category_model = LogisticRegression()
category_model.fit(X_train, y_cat_train)
# Evaluate Category Model

cat_predictions = category_model.predict(X_test)

print("Category Model Accuracy:", accuracy_score(y_cat_test, cat_predictions))

print("\nCategory Classification Report:\n")
print(classification_report(y_cat_test, cat_predictions))

# Train priority model
priority_model = LogisticRegression()
priority_model.fit(X_train2, y_pri_train)
# Evaluate Priority Model

pri_predictions = priority_model.predict(X_test2)

print("\nPriority Model Accuracy:", accuracy_score(y_pri_test, pri_predictions))

print("\nPriority Classification Report:\n")
print(classification_report(y_pri_test, pri_predictions))

# Save models
joblib.dump(category_model, "category_model.pkl")
joblib.dump(priority_model, "priority_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Models trained and saved successfully!")