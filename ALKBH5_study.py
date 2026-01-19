import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.keras import layers, models, constraints
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def load_local_rbp24(protein_name="ALKBH5_Baltz2012", max_len=375):
    archive_name = "GraphProt_CLIP_sequences.tar.bz2"
    data_dir = "RBP24_Data"
    if not os.path.exists(data_dir):
        if not os.path.exists(archive_name):
            raise FileNotFoundError(f"{archive_name} missing!")
        with tarfile.open(archive_name, "r:bz2") as tar:
            tar.extractall(path=data_dir)
    base_path = os.path.join(data_dir, "GraphProt_CLIP_sequences")
    map_dict = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3, 'N':0}
    seqs, labels = [], []
    found = False
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
    if not found: raise ValueError(f"Could not find {protein_name} files!")
    return tf.one_hot(seqs, depth=4).numpy(), np.array(labels)

class BioLinearCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size = filters, kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        return tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID') + self.bias

class BioPolynomialCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size, self.degree = filters, kernel_size, degree
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1, 1, 1), initializer='ones', trainable=True)
        self.c = self.add_weight(name="c", shape=(1, 1, 1), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        return tf.pow(self.gamma * dot + self.c, self.degree) + self.bias

class BioSphericalCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size = filters, kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name="s", shape=(1, 1, 1), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        norm_x = tf.sqrt(tf.nn.conv1d(tf.square(inputs), tf.ones_like(self.kernel), stride=1, padding='VALID') + 1e-7)
        return self.scale * (dot / norm_x) + self.bias

class BioGaussianCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size = filters, kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name="s", shape=(1, 1, 1), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1, 1, 1), initializer=tf.constant_initializer(0.1),
                                    constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        x_sq = tf.nn.conv1d(tf.square(inputs), tf.ones_like(self.kernel), stride=1, padding='VALID')
        dist_sq = tf.nn.relu(x_sq + 1.0 - 2 * dot)
        return self.scale * tf.exp(-self.gamma * dist_sq) + self.bias

X_all, y_all = load_local_rbp24()
MODELS = ["CNN", "Linear", "Polynomial", "Spherical", "Gaussian"]
RUNS = 30
EPOCHS = 50
FILTERS = 32
BATCH = 32
KERNEL_SIZE = 10
results = {'acc':[], 'auc':[], 'train_time':[], 'infer_time':[], 'params':[], 'model':[]}

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
        params = model.count_params()
        t0 = time.time()
        model.fit(X_tr, y_tr, epochs=EPOCHS, verbose=0, batch_size=BATCH)
        tt = (time.time() - t0)
        t0 = time.time()
        preds = model.predict(X_te, verbose=0)
        it = (time.time() - t0) * 1000
        auc = roc_auc_score(y_te, preds) if len(np.unique(y_te))>1 else 0.5
        acc = accuracy_score(y_te, (preds>0.5).astype(int))
        results['model'].append(m)
        results['acc'].append(acc)
        results['auc'].append(auc)
        results['train_time'].append(tt)
        results['infer_time'].append(it)
        results['params'].append(params)
    if (i+1)%5==0: print(f"Run {i+1}/{RUNS} Done")

df = pd.DataFrame(results)
for m in MODELS:
    d = df[df.model==m]
    print(f"{m:<12} | {d.acc.mean():.1%} | {d.auc.mean():.4f} | {d.auc.median():.4f} | {d.auc.std():.4f} | {d.params.iloc[0]:<8} | {d.train_time.mean():.2f}s | {d.infer_time.mean():.0f}ms")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.3)
axes[0,0].boxplot([df[df.model==m].auc for m in MODELS], labels=MODELS, patch_artist=True)
axes[0,0].set_title(f'AUC Distribution ({RUNS} Runs)')
axes[0,0].set_ylabel('AUC Score')
axes[0,0].grid(True, alpha=0.3)
colors = plt.cm.viridis(np.linspace(0, 1, len(MODELS)))
for idx, m in enumerate(MODELS):
    d = df[df.model==m]
    axes[0,1].scatter(d.train_time, d.auc, label=m, color=colors[idx], alpha=0.7)
axes[0,1].set_title('Training Time vs AUC')
axes[0,1].set_xlabel('Total Training Time (s)')
axes[0,1].set_ylabel('AUC')
axes[0,1].legend()
params_list = [df[df.model==m].params.iloc[0] for m in MODELS]
aucs_list = [df[df.model==m].auc.mean() for m in MODELS]
bars = axes[1,0].bar(MODELS, aucs_list, color='skyblue')
axes[1,0].set_ylim(min(aucs_list)-0.02, max(aucs_list)+0.01)
axes[1,0].set_title('Model Size Efficiency')
for bar, p in zip(bars, params_list):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{p} params', ha='center', va='bottom', fontsize=8)
infer_list = [df[df.model==m].infer_time.mean() for m in MODELS]
axes[1,1].bar(MODELS, infer_list, color='salmon')
axes[1,1].set_title('Inference Latency (Lower is Better)')
axes[1,1].set_ylabel('Time (ms)')
plt.suptitle(f"RBP-24 (ALKBH5) Benchmark: {RUNS} Runs x {EPOCHS} Epochs", fontsize=16)
plt.show()
