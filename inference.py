# inference.py
import numpy as np
import faiss
import json
from preprocessing import preprocess_for_model
from tensorflow.keras.models import load_model

class FingerprintMatcher:
    def __init__(self, emb_model_path='models/embedding_model', index_path='models/faiss.index'):
        self.emb = load_model(emb_model_path)
        self.index = faiss.read_index(index_path)
        with open(index_path + '.meta.json', 'r') as fh:
            self.meta = json.load(fh)

    def embed(self, image_path):
        x = preprocess_for_model(image_path)
        x = np.expand_dims(x, 0)
        v = self.emb.predict(x)
        return v.astype('float32')[0]

    def query(self, image_path, top_k=5):
        v = self.embed(image_path).reshape(1,-1).astype('float32')
        D, I = self.index.search(v, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append({'dist': float(dist), 'meta': self.meta[idx]})
        return results

if __name__ == '__main__':
    m = FingerprintMatcher()
    res = m.query('samples/query1.bmp', top_k=3)
    print(res)
