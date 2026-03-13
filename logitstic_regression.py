import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv(r"D:\Machine Learning\Road_Accident_Data.csv")

print(df.head())
print(df.columns)

# Handle Missing
df = df.dropna()

# Encoding
le_weather = LabelEncoder()
le_road = LabelEncoder()
le_acc = LabelEncoder()

df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Road_Condition'] = le_road.fit_transform(df['Road_Condition'])
df['Accident'] = le_acc.fit_transform(df['Accident'])

# Features
X = df[['Speed', 'Alcohol', 'Weather', 'Road_Condition']]
y = df['Accident']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Manual Prediction
w = le_weather.transform(['Rainy'])[0]
r = le_road.transform(['Wet'])[0]

sample = [[75, 1, w, r]]
print("Prediction:", model.predict(sample)[0])
