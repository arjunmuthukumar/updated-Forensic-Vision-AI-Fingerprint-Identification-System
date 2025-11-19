# model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_embedding_model(input_shape=(224,224,1), embedding_dim=128):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x)  # normalized embeddings

    model = models.Model(inputs, x, name='fingerprint_embedding')
    return model

def build_classifier(embedding_model, num_classes):
    # Freeze embedding if fine-tuning is not desired
    inputs = embedding_model.input
    embeddings = embedding_model.output
    x = layers.Dense(256, activation='relu')(embeddings)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='fp_classifier')
