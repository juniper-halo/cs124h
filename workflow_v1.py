import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def load_data_from_hdf5(file_path, target_size=(350, 350)):
    with h5py.File(file_path, 'r') as hf:
        images = np.array(hf['hashimghd'])     
        masks = np.array(hf['masksdf'])        
    
    # Resizing images and masks
    images_resized = [tf.image.resize(image, target_size) for image in images]
    masks_resized = [tf.image.resize(mask, target_size) for mask in masks]

    # Normalizing images and masks
    images_resized = np.array(images_resized) / 255.0
    masks_resized = np.array(masks_resized) / 255.0
    

    masks_resized = (masks_resized > 0.5).astype(np.float32)

    return images_resized, masks_resized


def save_prediction(model, image, save_path, threshold=0.5):
    
    input_image = tf.image.resize(image, (350, 350))
    input_image = tf.expand_dims(input_image, axis=3)  
    input_image = input_image / 255.0 

    
    prediction = model.predict(input_image)
    prediction = tf.squeeze(prediction) 

    
    binary_mask = (prediction >= threshold).numpy().astype(np.uint8)


    plt.imsave(save_path, binary_mask, cmap='gray')


train_images, train_masks = load_data_from_hdf5("C:\Users\madha\Downloads\MYD02QKM_6.1-20241106_214645")
val_images, val_masks = load_data_from_hdf5("C:\Users\madha\Downloads\MYD02QKM_6_split.1-20241106_214645")


model = unet_model(input_size=(350, 350, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(8).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(8).prefetch(tf.data.AUTOTUNE)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=25)

save_prediction(model, "C:\Users\madha\Downloads\MYD02QKM_6.1-20241106_214645\MYD02QKM.A2024030.0830.061.2024069200912.hdf", 'C:\Users\madha\wildfire_prediction.png')