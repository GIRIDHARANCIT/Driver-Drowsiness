# training/train_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# Paths & settings
data_dir = r'C:\Users\girid\OneDrive\Desktop\Driver-Drowsiness-main/data_collection/dataset'
categories = ['alert', 'drowsy']
img_size = 64

# Load images
data = []
for category in categories:
    folder = os.path.join(data_dir, category)
    label = categories.index(category)
    if not os.path.exists(folder):
        print(f"⚠️ Warning: {folder} doesn't exist!")
        continue
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        data.append((resized, label))

print(f"✅ Loaded {len(data)} images.")

# Shuffle and split
np.random.shuffle(data)
X = np.array([d[0] for d in data]).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array([d[1] for d in data])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧪 Training set: {len(X_train)}, Validation set: {len(X_val)}")

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Plot & save history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save model
os.makedirs('../python_app/model', exist_ok=True)
keras_path = '../python_app/model/drowsiness_model.h5'
model.save(keras_path)
print(f"✅ Saved Keras model to {keras_path}")

# Convert to TFLite
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
os.makedirs('../flutter_app/assets', exist_ok=True)
tflite_path = '../flutter_app/assets/model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model to {tflite_path}")

print("🎉 Training complete! You can now run your detection script or mobile app.")
