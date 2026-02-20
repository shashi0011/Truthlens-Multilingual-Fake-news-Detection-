import os, re, json, random, joblib
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# KEY PARAMS (remember)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

FILE_PATH = r"S:\MCA PRACTICAL\3rd sem\Minor Project\TruthLens\Data\Preprocessed\Preprocessed_hindi_data_fixed.csv"
TEXT_COL = "clean_joined"
LABEL_COL = "label"
OUT_DIR = r"S:\MCA PRACTICAL\3rd sem\Minor Project\TruthLens\models\pro\hindi_pro_lstm_from_tfidf_ref"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

MAX_VOCAB = 30000
MAX_LEN = 200
EMBEDDING_DIM = 300
FASTTEXT_PATH = Path("embeddings/cc.hi.300.vec")  # update if you have fastText
FAST_MODE = True
SUBSAMPLE_TRAIN = 4000
BATCH_SIZE = 32
EPOCHS = 8
MIN_TEXT_LEN = 3
LOWERCASE = False  # Devanagari texts -> keep False

try:
    from gensim.models.keyedvectors import KeyedVectors
    GENSIM = True
except Exception:
    GENSIM = False


def load_csv(path, text_col, label_col):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")
    try:
        return pd.read_csv(p, usecols=[text_col, label_col], encoding="utf-8", on_bad_lines="skip")
    except Exception:
        chunks = pd.read_csv(p, usecols=[text_col, label_col], engine="python",
                             chunksize=50000, encoding="utf-8", on_bad_lines="skip")
        return pd.concat(chunks, ignore_index=True)


def clean_text(texts):
    url_pat = re.compile(r"http\S+|www\.\S+")
    handle_pat = re.compile(r"[@#]\w+")
    out = []
    for t in texts:
        s = str(t)
        s = s.lower().strip() if LOWERCASE else s.strip()
        s = url_pat.sub(" ", s); s = handle_pat.sub(" ", s)
        s = re.sub(r"[^\u0900-\u097F\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        out.append(s)
    return out


def build_fasttext_matrix(word_index, ft_path, emb_dim):
    if not (GENSIM and ft_path.exists()):
        print("[info] fastText not loaded; using random embeddings")
        return None
    ft = KeyedVectors.load_word2vec_format(str(ft_path), binary=False)
    num_words = min(MAX_VOCAB, len(word_index) + 1)
    mat = np.random.normal(0, 0.01, (num_words, emb_dim)).astype("float32")
    matched = 0
    for w, i in word_index.items():
        if i >= num_words: continue
        if w in ft:
            mat[i] = ft[w]; matched += 1
    print(f"[info] Matched {matched}/{num_words} fastText vectors")
    return mat


def build_model(vocab_size, emb_dim, seq_len, emb_matrix=None):
    inp = keras.Input(shape=(seq_len,), dtype="int32")
    if emb_matrix is not None:
        emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[emb_matrix],
                               trainable=False)(inp)
    else:
        emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(emb)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = load_csv(FILE_PATH, TEXT_COL, LABEL_COL)
    print("Dataset loaded. Rows:", len(df))  # show dataset once after import

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(lambda x: x.strip())
    df = df[df[TEXT_COL].str.len() >= MIN_TEXT_LEN].reset_index(drop=True)
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    # Ensure labels are ints 0/1
    if not pd.api.types.is_integer_dtype(df[LABEL_COL]):
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    y = df[LABEL_COL].astype(int)
    X = df[TEXT_COL].astype(str)

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(sss.split(X_train_all, y_train_all))
    X_tr, y_tr = X_train_all.iloc[tr_idx].reset_index(drop=True), y_train_all.iloc[tr_idx].reset_index(drop=True)
    X_val, y_val = X_train_all.iloc[val_idx].reset_index(drop=True), y_train_all.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    if FAST_MODE:
        pos_idx = y_tr[y_tr == 1].index.tolist()
        neg_idx = y_tr[y_tr == 0].index.tolist()
        if len(pos_idx) != 0 and len(neg_idx) != 0:
            pos_n = max(1, int(SUBSAMPLE_TRAIN * (y_tr.sum() / len(y_tr))))
            neg_n = max(1, SUBSAMPLE_TRAIN - pos_n)
            sampled = random.sample(pos_idx, min(len(pos_idx), pos_n)) + random.sample(neg_idx, min(len(neg_idx), neg_n))
            random.shuffle(sampled)
            X_tr = X_tr.iloc[sampled].reset_index(drop=True)
            y_tr = y_tr.iloc[sampled].reset_index(drop=True)
            print("After subsample, train size:", len(X_tr))
        else:
            print("[warning] class imbalance; skipping subsample")

    X_tr_clean = clean_text(X_tr.tolist())
    X_val_clean = clean_text(X_val.tolist())
    X_test_clean = clean_text(X_test.tolist())

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr_clean)
    X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_tr_clean), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_clean), maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_clean), maxlen=MAX_LEN, padding="post", truncating="post")

    ft_path = FASTTEXT_PATH
    if not ft_path.exists():
        try:
            ft_path = Path(__file__).parent / FASTTEXT_PATH
        except NameError:
            ft_path = Path.cwd() / FASTTEXT_PATH

    embedding_matrix = build_fasttext_matrix(tokenizer.word_index, ft_path, EMBEDDING_DIM)
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print("Vocab size:", vocab_size)

    model = build_model(vocab_size=vocab_size, emb_dim=EMBEDDING_DIM, seq_len=MAX_LEN, emb_matrix=embedding_matrix)
    ckpt_path = Path(OUT_DIR) / "best_model.keras"
    keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    # class weights
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = dict(zip(classes, cw))

    history = model.fit(
        X_tr_seq, y_tr.values,
        validation_data=(X_val_seq, y_val.values),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, class_weight=class_weight, verbose=2
    )

    if ckpt_path.exists():
        model = keras.models.load_model(str(ckpt_path))

    # choose threshold by F1 on validation
    val_probs = model.predict(X_val_seq, batch_size=BATCH_SIZE).ravel()
    best_th, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    print("Chosen threshold (val):", best_th, "val F1:", best_f1)

    test_probs = model.predict(X_test_seq, batch_size=BATCH_SIZE).ravel()
    test_preds = (test_probs >= best_th).astype(int)
    print("Test accuracy:", accuracy_score(y_test, test_preds))
    print("Test F1:", f1_score(y_test, test_preds))
    try:
        print("Test ROC-AUC:", roc_auc_score(y_test, test_probs))
    except Exception:
        pass
    print("Classification report:\n", classification_report(y_test, test_preds, digits=4))

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = Path(OUT_DIR) / "hindi_lstm.keras"
    tokenizer_path = Path(OUT_DIR) / "tokenizer.json"
    meta_path = Path(OUT_DIR) / "meta.joblib"
    model.save(str(model_path))
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    joblib.dump({"threshold": float(best_th)}, meta_path)

    print("Saved model ->", str(model_path.resolve()))
    print("Saved tokenizer ->", str(tokenizer_path.resolve()))
    print("Saved meta ->", str(meta_path.resolve()))

if __name__ == "__main__":
    main()
