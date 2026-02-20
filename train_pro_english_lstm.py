import os
import re
import json
import random
from pathlib import Path


FILE_PATH = r"S:\MCA PRACTICAL\3rd sem\Minor Project\TruthLens\Data\Preprocessed\Preprocessed_english_data.csv"
TEXT_COL = "clean_joined"     
LABEL_COL = "isfake"          
OUT_DIR = r"models\pro\english_pro_lstm"
RANDOM_SEED = 42


MAX_VOCAB = 30000        # reduce for CPU/memory
MAX_LEN = 200            # sequence length for padding/truncating
EMBEDDING_DIM = 100      # 50/100 recommended; use 100 for glove.6B.100d
FAST_MODE = True         # subsample for quick runs 
SUBSAMPLE_TRAIN = 4000   # number of train samples when FAST_MODE True
BATCH_SIZE = 32
EPOCHS = 8

# GloVe.trainable embeddings used.
GLOVE_PATH = Path("Data/embeddings/glove.6B.100d.txt")  

# Imports (local, standard ML libs)
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ------------------ UTIL - robust CSV loader ------------------
def load_csv(path, text_col, label_col):
    
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    usecols = [text_col, label_col]
    try:
        df = pd.read_csv(p, usecols=usecols, encoding="utf-8", on_bad_lines="skip", low_memory=True)
        return df
    except Exception:
        import csv, sys
        try:
            csv.field_size_limit(sys.maxsize)
        except Exception:
            pass
        chunks = []
        for chunk in pd.read_csv(
            p, usecols=usecols, engine="python", encoding="utf-8",
            on_bad_lines="skip", chunksize=50000, sep=",",
            quotechar=None, quoting=csv.QUOTE_NONE, escapechar="\\"
        ):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


# ------------------ UTIL - label normalizer ------------------
def normalize_labels(ser):
    s = ser.copy()
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int)
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "fake": 1, "false": 1, "fake news": 1, "1": 1,
        "real": 0, "true": 0, "genuine": 0, "0": 0
    }
    mapped = s.map(mapping)
    if mapped.isnull().any():
        # fallback: if any of the "real" keywords in text -> 0 else 1
        mapped = mapped.fillna(s.apply(lambda x: 0 if any(w in x for w in ["real", "true", "genuine"]) else 1))
    return mapped.astype(int)


# ------------------ UTIL - text cleaner ------------------
def basic_clean(texts):
    url_pat = re.compile(r"http\S+|www\.\S+")
    handle_pat = re.compile(r"[@#]\w+")
    cleaned = []
    for t in texts:
        s = str(t).lower()
        s = url_pat.sub(" ", s)
        s = handle_pat.sub(" ", s)
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        cleaned.append(s)
    return cleaned


# ------------------ EMBEDDING MATRIX (GloVe) ------------------
def build_embedding_matrix(word_index, glove_path: Path, emb_dim):
    if not glove_path.exists():
        print("[info] GloVe not found at:", glove_path)
        return None
    print("[info] Loading GloVe from:", glove_path, " (this may take a while)...")
    embedding_index = {}
    with open(glove_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < emb_dim + 1:
                continue
            word = parts[0]
            coefs = np.asarray(parts[1:emb_dim+1], dtype="float32")
            embedding_index[word] = coefs
    print(f"[info] Loaded {len(embedding_index)} vectors from GloVe.")
    num_words = min(MAX_VOCAB, len(word_index) + 1)
    embedding_matrix = np.random.normal(0, 0.01, size=(num_words, emb_dim)).astype("float32")
    matched = 0
    for w, i in word_index.items():
        if i >= num_words:
            continue
        vec = embedding_index.get(w)
        if vec is not None:
            embedding_matrix[i] = vec
            matched += 1
    print(f"[info] Matched {matched}/{num_words} words to GloVe.")
    return embedding_matrix


# ------------------ MODEL BUILD ------------------
def build_model(vocab_size, emb_dim, input_length, embedding_matrix=None):
    inp = keras.Input(shape=(input_length,), dtype="int32", name="input")
    if embedding_matrix is not None:
        emb = layers.Embedding(
            input_dim=vocab_size, output_dim=emb_dim, input_length=input_length,
            weights=[embedding_matrix], trainable=False, name="embedding"
        )(inp)
    else:
        emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=input_length, name="embedding")(inp)

    # Small bidirectional LSTM stack for CPU
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(emb)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ------------------ MAIN ------------------
def main():
    print("Step A: load CSV...")
    df = load_csv(FILE_PATH, TEXT_COL, LABEL_COL)
    print("Raw rows loaded:", len(df))

    # Basic cleaning / filter duplicates
    print("Step B: basic cleaning/dedup...")
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() >= 20].reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    print("Removed duplicates:", before - len(df), "(remaining:", len(df), ")")

    # normalize labels
    print("Step C: normalize labels...")
    y = normalize_labels(df[LABEL_COL])
    X = df[TEXT_COL].astype(str)
    print("Label distribution:", y.value_counts().to_dict())

    # splits
    print("Step D: train/val/test splits (stratified)...")
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    tr_idx, val_idx = next(sss.split(X_train_all, y_train_all))
    X_tr, y_tr = X_train_all.iloc[tr_idx].reset_index(drop=True), y_train_all.iloc[tr_idx].reset_index(drop=True)
    X_val, y_val = X_train_all.iloc[val_idx].reset_index(drop=True), y_train_all.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # FAST_MODE subsample for speed on CPU
    if FAST_MODE:
        print("[FAST_MODE] Subsampling train set to", SUBSAMPLE_TRAIN)
        pos_idx = y_tr[y_tr == 1].index.tolist()
        neg_idx = y_tr[y_tr == 0].index.tolist()
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            print("[warning] class imbalance or no examples in one class; skipping subsample")
        else:
            pos_n = max(1, int(SUBSAMPLE_TRAIN * (y_tr.sum() / len(y_tr))))
            neg_n = max(1, SUBSAMPLE_TRAIN - pos_n)
            sampled = random.sample(pos_idx, min(len(pos_idx), pos_n)) + random.sample(neg_idx, min(len(neg_idx), neg_n))
            random.shuffle(sampled)
            X_tr = X_tr.iloc[sampled].reset_index(drop=True)
            y_tr = y_tr.iloc[sampled].reset_index(drop=True)
            print("After subsample, train size:", len(X_tr))

    # Clean texts
    print("Step E: cleaning texts...")
    X_tr_clean = basic_clean(X_tr.tolist())
    X_val_clean = basic_clean(X_val.tolist())
    X_test_clean = basic_clean(X_test.tolist())

    # Tokenizer + sequences
    print("Step F: tokenizer & sequences...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr_clean)
    X_tr_seq = tokenizer.texts_to_sequences(X_tr_clean)
    X_val_seq = tokenizer.texts_to_sequences(X_val_clean)
    X_test_seq = tokenizer.texts_to_sequences(X_test_clean)

    X_tr_pad = pad_sequences(X_tr_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # build embedding matrix if glove present
    glove_full = Path(GLOVE_PATH)
    if not glove_full.exists():
        # try relative to script dir
        glove_full = Path(__file__).parents[0] / GLOVE_PATH
    embedding_matrix = None
    if glove_full.exists():
        embedding_matrix = build_embedding_matrix(tokenizer.word_index, glove_full, EMBEDDING_DIM)
    else:
        print("[info] No GloVe found; using trainable embeddings.")

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print("Vocab size:", vocab_size)

    # build model
    print("Step G: build model...")
    model = build_model(vocab_size=vocab_size, emb_dim=EMBEDDING_DIM, input_length=MAX_LEN, embedding_matrix=embedding_matrix)
    model.summary()

    # callbacks
    ckpt_path = Path(OUT_DIR) / "best_model.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, save_weights_only=False),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    # fit
    print("Step H: training (this runs on CPU if no GPU available)...")
    history = model.fit(
        X_tr_pad, y_tr.values,
        validation_data=(X_val_pad, y_val.values),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2
    )

    # load best ckpt if saved
    if ckpt_path.exists():
        print("[info] loading best model from:", ckpt_path)
        model = keras.models.load_model(str(ckpt_path))

    # choose threshold on validation
    print("Step I: choose threshold using validation set...")
    val_probs = model.predict(X_val_pad, batch_size=BATCH_SIZE).ravel()
    best_th, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    print("Chosen threshold (val):", best_th, "val F1:", best_f1)

    # evaluate on test
    test_probs = model.predict(X_test_pad, batch_size=BATCH_SIZE).ravel()
    test_preds = (test_probs >= best_th).astype(int)
    print("Test accuracy:", accuracy_score(y_test, test_preds))
    print("Test F1:", f1_score(y_test, test_preds))
    try:
        print("Test ROC-AUC:", roc_auc_score(y_test, test_probs))
    except Exception:
        pass
    print("Classification report:\n", classification_report(y_test, test_preds, digits=4))

    # ------------------ SAVE ARTIFACTS ------------------
    print("Saving model and artifacts to", OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # save tokenizer as JSON
    try:
        tok_json = tokenizer.to_json()
        with open(Path(OUT_DIR) / "tokenizer.json", "w", encoding="utf-8") as f:
            f.write(tok_json)
        print("Saved tokenizer ->", Path(OUT_DIR) / "tokenizer.json")
    except Exception as e:
        print("[warning] failed to save tokenizer.json:", e)

    # save keras model as .keras
    keras_model_path = Path(OUT_DIR) / "english_pro_lstm.keras"
    try:
        model.save(str(keras_model_path))   # recommended .keras format
        print("Saved Keras model ->", keras_model_path)
    except Exception as e:
        print("[error] failed to save model as .keras:", e)
        # fallback: try SavedModel format directory
        try:
            savedmodel_dir = Path(OUT_DIR) / "saved_model_dir"
            model.save(str(savedmodel_dir), save_format="tf")
            print("Saved fallback SavedModel ->", savedmodel_dir)
        except Exception as e2:
            print("[error] fallback save also failed:", e2)

    # save meta + threshold
    meta = {
        "vocab_size": int(vocab_size),
        "embedding_dim": int(EMBEDDING_DIM),
        "max_len": int(MAX_LEN),
        "fast_mode": bool(FAST_MODE),
        "n_train_used": int(len(X_tr)),
        "threshold": float(best_th)
    }
    with open(Path(OUT_DIR) / "info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    try:
        joblib.dump({"threshold": float(best_th)}, Path(OUT_DIR) / "meta.joblib")
    except Exception as e:
        print("[warning] failed to joblib.dump meta:", e)

    print("Done. Artifacts saved to:", OUT_DIR)
    print("Load model: tf.keras.models.load_model('path/to/english_pro_lstm.keras')")
    print("Load tokenizer: from tensorflow.keras.preprocessing.text import tokenizer_from_json; tok = tokenizer_from_json(json.load(open('tokenizer.json')))")
    # ------------------ END ------------------


if __name__ == "__main__":
    main()
