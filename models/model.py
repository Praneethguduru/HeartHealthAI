import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.utils import to_categorical
import pickle

# Load and clean the dataset
data = pd.read_csv('framingham.csv')
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

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define an improved neural network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Adding dropout to prevent overfitting
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')  # Softmax for binary classification with 2 output classes
])

# Compile the model with class weights to handle imbalance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model using accuracy and AUC-ROC
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred_labels)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model AUC-ROC: {auc_roc:.2f}")

# Save the trained model
model.save('heart_attack_prediction_model_improved.h5')
