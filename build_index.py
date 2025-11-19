# build_index.py
import os
import glob
import numpy as np
import faiss
from preprocessing import preprocess_for_model
from tensorflow.keras.models import load_model

def build(root_dir, emb_model_path='models/embedding_model', index_path='models/faiss.index'):
    emb = load_model(emb_model_path)
    features = []
    meta = []
    persons = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    for p in persons:
        files = glob.glob(os.path.join(root_dir, p, '*'))
        for f in files:
            try:
                x = preprocess_for_model(f)
                x = np.expand_dims(x, 0)
                v = emb.predict(x)
                features.append(v[0].astype('float32'))
                meta.append({'path': f, 'label': p})
            except Exception as e:
                print("skip", f, e)

    feats = np.stack(features, axis=0)
    d = feats.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(feats)
    faiss.write_index(index, index_path)
    # save meta
    import json
    with open(index_path + '.meta.json', 'w') as fh:
        json.dump(meta, fh)
    print("Index saved:", index_path)

if __name__ == '__main__':
    build('data/fingerprints')
