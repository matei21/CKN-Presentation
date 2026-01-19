import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd
import gc
import zipfile
from tqdm import tqdm
from tensorflow.keras import layers, models, constraints, callbacks
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

def extract_data_once():
    archive_name = "GraphProt_CLIP_sequences.tar.bz2"
    data_dir = "RBP24_Data"
    if not os.path.exists(data_dir):
        if not os.path.exists(archive_name):
            return
        try:
            with tarfile.open(archive_name, "r:bz2") as tar:
                tar.extractall(path=data_dir)
        except: pass

def load_local_rbp24(protein_name, max_len=375):
    extract_data_once()
    base_path = os.path.join("RBP24_Data", "GraphProt_CLIP_sequences")
    map_dict = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3, 'N':0}
    seqs, labels = [], []
    if not os.path.exists(base_path): base_path = "RBP24_Data"
    found = False
    if os.path.exists(base_path):
        for f in os.listdir(base_path):
            if f.startswith(protein_name):
                found = True
                label = 1 if "pos" in f else 0
                with open(os.path.join(base_path, f), 'r') as fp:
                    for line in fp:
                        if line.startswith('>') or not line.strip(): continue
                        s = line.strip().upper()
                        s = s[:max_len].ljust(max_len, 'N')
                        seqs.append([map_dict.get(c, 0) for c in s])
                        labels.append(label)
    if not found:
        return None, None
    return tf.one_hot(seqs, depth=4).numpy().astype('float32'), np.array(labels)

try:
    get_custom_objects().clear()
except: pass

@tf.keras.utils.register_keras_serializable()
class BioLinearCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        return tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID') + self.bias
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

@tf.keras.utils.register_keras_serializable()
class BioPolynomialCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.degree = degree
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1, 1, 1), initializer='ones', trainable=True)
        self.c = self.add_weight(name="c", shape=(1, 1, 1), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        norm_x = tf.sqrt(tf.nn.conv1d(tf.square(inputs), tf.ones_like(self.kernel), stride=1, padding='VALID') + 1e-7)
        normalized_dot = dot / norm_x
        return tf.pow(self.gamma * normalized_dot + self.c, self.degree) + self.bias
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "degree": self.degree})
        return config

@tf.keras.utils.register_keras_serializable()
class BioSphericalCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name="s", shape=(1, 1, 1), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        norm_x = tf.sqrt(tf.nn.conv1d(tf.square(inputs), tf.ones_like(self.kernel), stride=1, padding='VALID') + 1e-7)
        return self.scale * (dot / norm_x) + self.bias
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

@tf.keras.utils.register_keras_serializable()
class BioGaussianCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name="s", shape=(1, 1, 1), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1, 1, 1), initializer=tf.constant_initializer(1.0),
                                     constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        x_sq = tf.nn.conv1d(tf.square(inputs), tf.ones_like(self.kernel), stride=1, padding='VALID')
        norm_x = tf.sqrt(x_sq + 1e-7)
        cosine_sim = dot / norm_x
        dist_sq = 2.0 - 2.0 * cosine_sim
        return self.scale * tf.exp(-self.gamma * tf.nn.relu(dist_sq)) + self.bias
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

def run_protein_study(protein_name):
    X_all, y_all = load_local_rbp24(protein_name)
    if X_all is None: return
    MODELS = ["CNN", "Linear", "Polynomial", "Spherical", "Gaussian"]
    RUNS = 3
    EPOCHS = 15
    FILTERS = 32
    BATCH = 32
    KERNEL_SIZE = 10
    results = {m: {'acc': [], 'auc': [], 'f1': [], 'train_time': [], 'infer_time': [], 'params': 0} for m in MODELS}
    best_models = {m: {'model': None, 'auc': 0.0} for m in MODELS}
    pbar = tqdm(total=RUNS * len(MODELS), desc=f"Training {protein_name}", unit="model")
    for i in range(RUNS):
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=i, stratify=y_all)
        for m in MODELS:
            tf.keras.backend.clear_session()
            inp = layers.Input(shape=(375, 4))
            if m=="CNN": x = layers.Conv1D(FILTERS, KERNEL_SIZE, activation='softplus', padding='valid')(inp)
            elif m=="Linear": x = BioLinearCKN1D(FILTERS, KERNEL_SIZE)(inp)
            elif m=="Polynomial": x = BioPolynomialCKN1D(FILTERS, KERNEL_SIZE)(inp)
            elif m=="Spherical": x = BioSphericalCKN1D(FILTERS, KERNEL_SIZE)(inp)
            elif m=="Gaussian": x = BioGaussianCKN1D(FILTERS, KERNEL_SIZE)(inp)
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Dropout(0.2)(x)
            out = layers.Dense(1, activation='sigmoid')(x)
            model = models.Model(inp, out)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            if i == 0: results[m]['params'] = model.count_params()
            t0 = time.time()
            model.fit(X_tr, y_tr, epochs=EPOCHS, verbose=0, batch_size=BATCH, validation_split=0.1)
            results[m]['train_time'].append(time.time() - t0)
            t1 = time.time()
            preds = model.predict(X_te, verbose=0)
            results[m]['infer_time'].append((time.time() - t1) * 1000)
            if len(np.unique(y_te)) > 1:
                auc = roc_auc_score(y_te, preds)
            else: auc = 0.5
            results[m]['auc'].append(auc)
            results[m]['acc'].append(accuracy_score(y_te, (preds>0.5).astype(int)))
            results[m]['f1'].append(f1_score(y_te, (preds>0.5).astype(int)))
            if auc > best_models[m]['auc']:
                best_models[m]['auc'] = auc
                best_models[m]['model'] = model
            del model, preds
            gc.collect()
            pbar.update(1)
    pbar.close()
    df_rows = []
    for m in MODELS:
        accs, aucs, f1s = np.array(results[m]['acc']), np.array(results[m]['auc']), np.array(results[m]['f1'])
        m_acc, s_acc = np.mean(accs), np.std(accs)
        ci = 1.96 * (s_acc / np.sqrt(RUNS))
        print(f"{m:<12} | {m_acc:.3f} | {np.mean(aucs):.4f} | {np.median(aucs):.4f} | {np.std(aucs):.4f} | {np.mean(f1s):.3f} | {m_acc-ci:.3f}-{m_acc+ci:.3f} | {results[m]['params']:<10} | {np.mean(results[m]['train_time']):.2f} | {np.mean(results[m]['infer_time']):.2f}")
        for a in aucs: df_rows.append({'Model': m, 'AUC': a})
    df_plot = pd.DataFrame(df_rows)
    plt.figure(figsize=(10, 6))
    data_to_plot = [df_plot[df_plot['Model'] == m]['AUC'].values for m in MODELS]
    plt.boxplot(data_to_plot, labels=MODELS, patch_artist=True)
    plt.title(f"{protein_name}: AUC Distribution")
    plt.ylabel("AUC Score")
    plt.grid(True, alpha=0.3)
    safe_name = protein_name.replace("/", "_")
    plt.savefig(f"Results_{safe_name}.png")
    plt.close()
    for m in MODELS:
        if best_models[m]['model'] is not None:
            save_name = f"{protein_name}_{m}.keras"
            try:
                best_models[m]['model'].save(save_name)
            except: pass
    del X_all, y_all, results, df_plot, best_models
    gc.collect()

PROTEIN_LIST = ["PTBv1"]
for prot in PROTEIN_LIST:
    try:
        run_protein_study(prot)
    except: continue
