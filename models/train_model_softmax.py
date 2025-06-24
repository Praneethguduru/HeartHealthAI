import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('framingham.csv')

# Drop unnecessary columns
data = data.drop(columns=['education'])

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define features and target variable
X = data_imputed.drop(columns=['TenYearCHD'])
y = data_imputed['TenYearCHD']

# One-hot encode the target variable
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the neural network model with softmax activation in the output layer
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')  # 2 output nodes for softmax
])

# Compile the model with categorical_crossentropy loss for softmax
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('heart_attack_prediction_model_with_softmax.h5')
