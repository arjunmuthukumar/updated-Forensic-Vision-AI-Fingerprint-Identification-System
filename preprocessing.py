# preprocessing.py
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import random

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def normalize(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.ptp(img) + 1e-8)
    return (img * 255).astype(np.uint8)

def enhance_ridges(img):
    # Simple CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def binarize(img, th=0):
    if th == 0:
        th, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, out = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    return out

def center_crop(img, size=(224,224)):
    h,w = img.shape[:2]
    th, tw = size
    startx = max(0, w//2 - tw//2)
    starty = max(0, h//2 - th//2)
    return img[starty:starty+th, startx:startx+tw]

def resize(img, size=(224,224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def random_augment_pil(pil_img):
    # simple PIL augmentations: rotate, contrast, noise
    if random.random() < 0.5:
        pil_img = pil_img.rotate(random.uniform(-15,15))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8,1.3))
    if random.random() < 0.3:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.0)))
    return pil_img

def preprocess_for_model(path, size=(224,224), augment=False):
    img = read_gray(path)
    img = enhance_ridges(img)
    img = normalize(img)
    img = resize(img, size)
    pil = Image.fromarray(img)
    if augment:
        pil = random_augment_pil(pil)
    arr = np.array(pil).astype(np.float32) / 255.0
    # Add channel
    arr = np.expand_dims(arr, axis=-1)
    return arr
