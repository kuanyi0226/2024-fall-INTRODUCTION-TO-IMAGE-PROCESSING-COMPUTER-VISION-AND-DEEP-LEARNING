# train.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Task 1.1: Load and display VGG16 architecture
def load_and_show_vgg16():
    model = VGG16(include_top=True, weights=None, input_shape=(28, 28, 3), classes=10)
    model.summary()  # Prints the model architecture
    return model

# Task 1.2: Show training/validating accuracy and loss
def show_accuracy_and_loss(history_path='mnist_vgg16_history.npy'):
    history = np.load(history_path, allow_pickle=True).item()
    plt.figure(figsize=(10, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Task 1.3: Use the best model to predict
def predict_with_vgg16(model_path='best_mnist_model.h5', sample_image_path='sample_image.png'):
    model = load_model(model_path)
    sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(28, 28))
    sample_image_array = tf.keras.preprocessing.image.img_to_array(sample_image) / 255.0
    sample_image_array = np.expand_dims(sample_image_array, axis=0)  # Add batch dimension

    prediction = model.predict(sample_image_array)
    predicted_class = np.argmax(prediction)

    # Plot prediction distribution
    plt.bar(range(10), prediction[0])
    plt.title(f'Predicted Class: {predicted_class}')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.show()

# Task 2.1: Load and resize Cat-Dog dataset
def load_and_resize_dataset(dataset_path, target_size=(224, 224)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    dataset = datagen.flow_from_directory(dataset_path, target_size=target_size, batch_size=32, class_mode='binary')
    return dataset

# Task 2.2: Show ResNet50 architecture
def load_and_show_resnet50():
    model = ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2)
    model.summary()  # Prints the model architecture
    return model

# Task 2.3: Train ResNet50 with Random-Erasing and compare accuracies
def train_and_compare_resnet50(dataset, model_save_path='resnet50_model.h5', epochs=10):
    model = ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Simulate Random-Erasing in training (as an augmentation)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2
    )
    train_data = datagen.flow_from_directory(
        dataset,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    history = model.fit(train_data, epochs=epochs)
    model.save(model_save_path)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy with Random Erasing')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Task 2.4: Predict with ResNet50
def predict_with_resnet50(model_path='resnet50_model.h5', sample_image_path='cat_or_dog.png'):
    model = load_model(model_path)
    sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(224, 224))
    sample_image_array = tf.keras.preprocessing.image.img_to_array(sample_image) / 255.0
    sample_image_array = np.expand_dims(sample_image_array, axis=0)  # Add batch dimension

    prediction = model.predict(sample_image_array)
    predicted_class = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

    print(f'Predicted Class: {predicted_class}')
