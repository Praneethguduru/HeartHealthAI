import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.utils import to_categorical
import pickle

# Load the dataset
data = pd.read_csv('framingham.csv')

# Drop the 'education' column
data = data.drop(columns=['education'])

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define features and target variable
X = data_imputed.drop(columns=['TenYearCHD'])
y = data_imputed['TenYearCHD']

# One-hot encode the target variable for softmax output
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

# Define the neural network model with deep learning techniques
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),  # Apply Batch Normalization
    Dropout(0.5),  # Apply Dropout
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output neurons for binary classification with softmax
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Compute class weights to handle class imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train the model with class weights, early stopping, and learning rate reduction
model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy and AUC-ROC score
accuracy = accuracy_score(y_test_labels, y_pred_labels)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model AUC-ROC: {auc_roc:.2f}")

# Save the trained model
model.save('heart_attack_prediction_model_advanced.h5')
