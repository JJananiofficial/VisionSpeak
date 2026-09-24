import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Dataset
data = pd.read_csv("gesture_dataset.csv")

X = data.iloc[:, 1:].values # 84 Coordinates
y = data.iloc[:, 0].values  # Labels

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Neural Network Architecture (Updated input_shape to 84)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(84,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
print("Training Model...")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the Model
model.save("vision_speak_model.h5")
print("Model saved as vision_speak_model.h5")