import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
import pickle

# Load and preprocess the dataset
data = pd.read_csv('framingham.csv')
data = data.drop(columns=['education'])
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_imputed.drop(columns=['TenYearCHD'])
y = data_imputed['TenYearCHD']
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the neural network model with additional features
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train the model
model.fit(
    X_train_scaled, y_train,
    epochs=200, batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred_labels)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Neural Network Model Accuracy: {accuracy * 100:.2f}%")
print(f"Neural Network Model AUC-ROC: {auc_roc:.2f}")

# XGBoost Model for Comparison
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weights[1], use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, np.argmax(y_train, axis=1))

# XGBoost Evaluation
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(np.argmax(y_test, axis=1), xgb_pred)
xgb_auc_roc = roc_auc_score(np.argmax(y_test, axis=1), xgb_model.predict_proba(X_test)[:, 1])

print(f"XGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"XGBoost Model AUC-ROC: {xgb_auc_roc:.2f}")
