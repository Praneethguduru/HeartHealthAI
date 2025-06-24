import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('framingham.csv')

# Drop the 'education' column
data = data.drop(columns=['education'])

# Handle missing values by dropping rows with any missing values
data = data.dropna()

# Define features and target variable
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_data=(X_test_scaled, y_test))

# Save the trained model
model.save('heart_attack_prediction_model.h5')

# Evaluate the model using the `evaluate` method
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy (evaluate method): {accuracy * 100:.2f}%")

# Evaluate the model using `accuracy_score` manually
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
accuracy_manual = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (manual method): {accuracy_manual * 100:.2f}%")

# Model summary
model.summary()
