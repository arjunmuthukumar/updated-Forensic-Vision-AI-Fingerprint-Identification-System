# train.py
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from preprocessing import preprocess_for_model
from model import build_embedding_model, build_classifier

def load_dataset(root_dir):
    # expects folder structure root_dir/<person_id>/*.bmp
    X, y, labels = [], [], []
    persons = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    label_map = {p:i for i,p in enumerate(persons)}
    for p in persons:
        files = glob.glob(os.path.join(root_dir, p, '*'))
        for f in files:
            try:
                X.append(preprocess_for_model(f, augment=False))
                y.append(label_map[p])
            except Exception as e:
                print("skip", f, e)
    X = np.stack(X, axis=0)
    y = np.array(y)
    return X, y, label_map

def main():
    root = 'data/fingerprints'  # adapt
    X, y, label_map = load_dataset(root)
    num_classes = len(label_map)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    embedding_dim = 128
    emb_model = build_embedding_model(input_shape=X.shape[1:], embedding_dim=embedding_dim)
    classifier = build_classifier(emb_model, num_classes)

    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('models/classifier_best.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    classifier.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=30,
        callbacks=[checkpoint, reduce_lr]
    )

    # save embedding model separately
    emb_model.save('models/embedding_model')
    classifier.save('models/classifier_full')

if __name__ == '__main__':
    main()

