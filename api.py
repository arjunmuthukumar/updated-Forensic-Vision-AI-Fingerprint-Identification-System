# api.py
import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from inference import FingerprintMatcher
from crypto_utils import encrypt, decrypt, generate_key
import base64
import json

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key')  # set in env
jwt = JWTManager(app)

# Simple in-memory store for demonstration; in prod use DB / KMS for keys
TEMPLATE_STORE = {}  # id -> {cipher, meta}
MASTER_KEY = base64.b64decode(os.environ.get('MASTER_KEY_B64')) if os.environ.get('MASTER_KEY_B64') else generate_key()

matcher = FingerprintMatcher()  # loads embedding model and faiss index

@app.route('/auth', methods=['POST'])
def auth():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    # Replace with real auth check (LDAP/IdP)
    if username == 'admin' and password == 'password':
        token = create_access_token(identity=username)
        return jsonify(access_token=token)
    return jsonify(msg='Bad credentials'), 401

@app.route('/register-template', methods=['POST'])
@jwt_required()
def register_template():
    """
    Accepts: multipart form-data with 'file' (fingerprint image) and 'id' or 'label'
    Stores encrypted embedding vector
    """
    file = request.files.get('file')
    label = request.form.get('label', 'unknown')
    if not file:
        return jsonify(msg='no file'), 400

    # Save temporarily
    tmp_path = f"/tmp/{file.filename}"
    file.save(tmp_path)

    emb = matcher.emb.predict(matcher._prep(tmp_path)) if hasattr(matcher, '_prep') else matcher.embed(tmp_path)
    # emb is numpy array
    import numpy as np
    raw = np.array(emb).astype('float32').tobytes()
    cipher = encrypt(MASTER_KEY, raw, associated_data=label.encode('utf-8'))
    TEMPLATE_STORE[label] = {'cipher': cipher.decode(), 'meta': {'label': label}}
    return jsonify(msg='stored', label=label)

@app.route('/query', methods=['POST'])
@jwt_required()
def query():
    # Accept image upload and return top matches
    file = request.files.get('file')
    if not file:
        return jsonify(msg='no file'), 400
    tmp_path = f"/tmp/{file.filename}"
    file.save(tmp_path)

    results = matcher.query(tmp_path, top_k=5)
    return jsonify(results=results)

@app.route('/health')
def health():
    return jsonify(status='ok')

if __name__ == '__main__':
    # use gunicorn in production
    app.run(host='0.0.0.0', port=5000)
