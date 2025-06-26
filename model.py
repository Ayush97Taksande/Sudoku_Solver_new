import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset path
path = 'Digits_new'
myList = os.listdir(path)

# Load images from the dataset
images = []
classNo = []
numOfClasses = len(myList)

print("Loading images...")
for x in range(numOfClasses):
    myPiclist = os.listdir(os.path.join(path, str(x)))
    for y in myPiclist:
        curImg = cv2.imread(os.path.join(path, str(x), y))
        if curImg is not None:
            curImg = cv2.resize(curImg, (32, 32))
            images.append(curImg)
            classNo.append(x)
        else:
            print(f"Error reading image: {os.path.join(path, str(x), y)}")
    print(f"Loaded class {x} with {len(myPiclist)} images")

images = np.array(images)
classNo = np.array(classNo)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=0.2, random_state=42, stratify=classNo)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

print("\nData shapes:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)

# Preprocessing function
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Apply preprocessing
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Reshape data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weights = dict(enumerate(class_weights))
print("\nClass weights:", class_weights)

# Image augmentation
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10,
    horizontal_flip=False,
    vertical_flip=False
)

# Only augment training data
train_datagen.fit(X_train)

# One-hot encoding
Y_train = to_categorical(Y_train, numOfClasses)
Y_test = to_categorical(Y_test, numOfClasses)
Y_validation = to_categorical(Y_validation, numOfClasses)

# Improved model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(numOfClasses, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

model = create_model()
print("\nModel summary:")
model.summary()

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
print("\nTraining model...")
history = model.fit(
    train_datagen.flow(X_train, Y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=30,
    validation_data=(X_validation, Y_validation),
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save model
model.save("digit_classifier.h5")
print("Model saved as digit_classifier.h5")

# Prediction function
def predict_digit(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image")
        return None
        
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Example usage
# test_image_path = "path_to_your_test_image.jpg"
# digit, confidence = predict_digit(test_image_path, model)
# print(f"Predicted Digit: {digit} with confidence: {confidence:.2f}")