# crypto_utils.py
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64

def generate_key():
    # 256-bit key
    return AESGCM.generate_key(bit_length=256)

def encrypt(key: bytes, plaintext: bytes, associated_data: bytes = None) -> bytes:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    # store nonce + ciphertext, base64 for JSON-friendliness
    return base64.b64encode(nonce + ct)

def decrypt(key: bytes, b64_cipher: bytes, associated_data: bytes = None) -> bytes:
    raw = base64.b64decode(b64_cipher)
    nonce = raw[:12]
    ct = raw[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, associated_data)
