
import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

 
def segment_eye(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    threshold_img = cv2.adaptiveThreshold(
        blurred_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    segmented_img = cv2.dilate(threshold_img, kernel, iterations=2)
    return segmented_img

 
def blur_background(img, segmented_img):
    blurred_background = cv2.GaussianBlur(img, (21, 21), 0)
    result = np.where(segmented_img[..., None] == 255, img, blurred_background)
    return result

 
def highlight_cloudiness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    highlighted = cv2.addWeighted(img, 1, cv2.cvtColor(white_areas, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return highlighted

 
def remove_reflections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, reflections = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    reflection_removed = cv2.inpaint(img, reflections, 3, cv2.INPAINT_TELEA)
    return reflection_removed

 
dataset_path = "/Users/anoodabdulrhman/Desktop/Dataset/Train"
preprocessed_dataset_path = "/Users/anoodabdulrhman/Desktop/Dataset/Preprocessed_Train"
model_save_path = "/Users/anoodabdulrhman/Desktop/optimized_cnn_model.keras"
test_path = "/Users/anoodabdulrhman/Desktop/Dataset/Test"

 
def preprocess_and_augment_images(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                input_path = os.path.join(root, file)
                img = cv2.imread(input_path)
                if img is None:
                    continue
                
         
                segmented_img = segment_eye(img)
                img = blur_background(img, segmented_img)
                img = highlight_cloudiness(img)
                img = remove_reflections(img)
                
      
                relative_path = os.path.relpath(root, dataset_path)
                save_dir = os.path.join(output_path, relative_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output_file_path = os.path.join(save_dir, file)
                cv2.imwrite(output_file_path, img)

    print("Preprocessing complete and images saved to:", output_path)

 
def preprocess_data_with_preprocessed_images(preprocessed_path):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        preprocessed_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        subset="training"
    )
    validation_generator = datagen.flow_from_directory(
        preprocessed_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        subset="validation"
    )
    return train_generator, validation_generator

 
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

 
def train_model(model, train_generator, validation_generator, save_path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=25,
        callbacks=[early_stopping, checkpoint]
    )
    print("Training complete. Model saved to", save_path)
    return history

 
def test_model(model_path, test_path):
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode="binary",
        shuffle=False
    )
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

 
def predict_image(model_path, image_path):
    model = load_model(model_path)
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unable to load.")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        segmented_img = segment_eye(img)
        img = blur_background(img, segmented_img)
        img = highlight_cloudiness(img)
        img = remove_reflections(img)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = model.predict(img)
        if prediction > 0.5:
            print("Prediction: No Cataract.")
        else:
            print("Prediction: Cataract detected.")
    except Exception as e:
        print(f"Error loading or processing the image: {e}")

 
if __name__ == "__main__": 
    preprocess_and_augment_images(dataset_path, preprocessed_dataset_path)

 
    train_generator, validation_generator = preprocess_data_with_preprocessed_images(preprocessed_dataset_path)
    print("Class indices:", train_generator.class_indices)

   
    model = build_cnn_model()
    train_model(model, train_generator, validation_generator, model_save_path)
 
    test_model(model_save_path, test_path)

 
    test_image_path = "/Users/anoodabdulrhman/Desktop/Dataset/Test/Cataract/cat_0_205.jpg"
    predict_image(model_save_path, test_image_path)



