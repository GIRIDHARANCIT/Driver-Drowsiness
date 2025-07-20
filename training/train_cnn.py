 # Driver Drowsiness Detection - CNN Training Script

# 📦 Install required packages (run once, or use in notebook)
# !pip install tensorflow scikit-learn matplotlib opencv-python

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Step 1: Define dataset paths & parameters
data_dir = '../data_collection/dataset'  # adjust if needed
categories = ['alert', 'drowsy']
img_size = 64

# -----------------------------------------------------------
# Step 2: Load and preprocess images
data = []

for category in categories:
    folder = os.path.join(data_dir, category)
    label = categories.index(category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, label])

print(f"✅ Loaded {len(data)} images")

# -----------------------------------------------------------
# Step 3: Shuffle and split data
np.random.shuffle(data)

X, y = [], []
for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# -----------------------------------------------------------
# Step 4: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------------------------------------
# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# -----------------------------------------------------------
# Step 6: Plot training history and save plot
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
plt.savefig('training_history.png')  # save the plot as image
plt.show()

# -----------------------------------------------------------
# Step 7: Save trained model
save_path = '../python_app/model/drowsiness_model.h5'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"✅ Saved model to {save_path}")

# -----------------------------------------------------------
# Step 8: Convert to TFLite and save for Flutter app
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = '../flutter_app/assets/model.tflite'
os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Saved TFLite model to {tflite_path}")

# -----------------------------------------------------------
print("\n🎉 All done! Use the saved model in your real-time detection script or Flutter app.")
