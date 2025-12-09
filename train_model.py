import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load cached geolocation data (our training data source)
with open('geo_cache.json', 'r') as f:
    cache_data = json.load(f)

# Prepare dataset
data = []
for ip, info in cache_data.items():
    # Create feature vector
    features = {
        'latitude': info.get('latitude', 0),
        'longitude': info.get('longitude', 0),
        'abuseConfidence': info.get('abuseConfidence', 0),
        'country': info.get('country', 'Unknown'),
        'cached': 1 if info.get('cached', False) else 0
    }
    
    # Label: High risk if abuseConfidence >= 90
    label = 1 if features['abuseConfidence'] >= 90 else 0
    
    data.append({**features, 'is_ddos': label})

df = pd.DataFrame(data)

# Encode country names to numbers
le = LabelEncoder()
df['country_encoded'] = le.fit_transform(df['country'])

# Features and target
X = df[['latitude', 'longitude', 'abuseConfidence', 'country_encoded', 'cached']]
y = df['is_ddos']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'DDoS']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and encoder
joblib.dump(model, 'ddos_model.pkl')
joblib.dump(le, 'country_encoder.pkl')

print("\n✅ Model saved as 'ddos_model.pkl'")
print("✅ Encoder saved as 'country_encoder.pkl'")
