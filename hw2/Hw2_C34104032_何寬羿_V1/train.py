import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Q1: Load and Show VGG16 Model Architecture
def load_and_show_vgg16():
    model = VGG16(weights=None, include_top=True, input_shape=(32, 32, 1), classes=10)
    model.summary()
    return model

# Q1: Show Training/Validation Accuracy and Loss
def show_accuracy_and_loss(history_path='mnist_vgg16_history.npy'):
    if not os.path.exists(history_path):
        print("History file not found!")
        return
    
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

# Q1: Predict with VGG16 Model
def predict_with_vgg16(model_path='best_mnist_model.h5', image_path='sample_image.png'):
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("Model or image file not found!")
        return None

    model = load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32), color_mode='grayscale')
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # Plot prediction distribution
    plt.bar(range(10), prediction[0])
    plt.title(f'Predicted Class: {predicted_class}')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.show()

    return predicted_class

# Q2: Load and Resize Cat-Dog Dataset
def load_and_resize_cat_dog_dataset(dataset_path, target_size=(224, 224), batch_size=32):
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found!")
        return None, None

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_data = datagen.flow_from_directory(
        os.path.join(dataset_path, "training_dataset"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
    )
    validation_data = datagen.flow_from_directory(
        os.path.join(dataset_path, "validation_dataset"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
    )
    return train_data, validation_data

# Q2: Train or Load ResNet50 Model
def train_or_load_resnet50(model_path, dataset_path, epochs=10):
    if os.path.exists(model_path):
        print(f"Model {model_path} already exists. Loading the model...")
        return load_model(model_path)

    print("Training ResNet50 model...")
    train_data, validation_data = load_and_resize_cat_dog_dataset(dataset_path)
    if train_data is None or validation_data is None:
        return None

    # 修改 ResNet50 的輸出層
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        verbose=1,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return model


# Q2: Show ResNet50 Model Architecture
def show_resnet50_architecture():
    model = ResNet50(weights=None, include_top=True, input_shape=(224, 224, 3), classes=2)
    model.summary()
    return model

# Q2: Predict with ResNet50 Model
def predict_with_resnet50(model_path, image_path):
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("Model or image file not found!")
        return None

    model = load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = "Dog" if prediction[0][0] > 0.5 else "Cat"
    return predicted_class

if __name__ == "__main__":
    dataset_path = "../Q2_Dataset/dataset/"
    model_path = "model/Q2_resnet50_model.h5"
    image_path = "sample_image.jpg"  # Replace with the image you want to test

    # Train or load the model
    model = train_or_load_resnet50(model_path, dataset_path, epochs=10)

    # Predict with the model
    if model:
        prediction = predict_with_resnet50(model_path, image_path)
        print(f"Predicted Class: {prediction}")
