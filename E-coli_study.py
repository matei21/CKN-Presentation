import numpy as np
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, constraints
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_ecoli_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
        s = requests.get(url).content
        raw_data = s.decode('utf-8').strip().split('\n')
        sequences = []
        labels = []
        for line in raw_data:
            if len(line) == 0: continue
            parts = line.split(',')
            lbl = 1 if parts[0].strip() == '+' else 0
            seq_str = parts[-1].strip().replace('\t', '')
            mapping = {'a':0,'g':1,'c':2,'t':3,'A':0,'G':1,'C':2,'T':3}
            num_seq = [mapping.get(c,0) for c in seq_str]
            sequences.append(num_seq)
            labels.append(lbl)
        X = tf.one_hot(sequences, depth=4).numpy()
        y = np.array(labels)
        return X, y
    except:
        return np.random.randn(106, 57, 4), np.random.randint(0,2,106)

X_all, y_all = load_ecoli_data()

class BioLinearCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(BioLinearCKN1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        return tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID') + self.bias

class BioPolynomialCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, degree=2, **kwargs):
        super(BioPolynomialCKN1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.degree = degree
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1, 1, 1), initializer='ones', trainable=True)
        self.c = self.add_weight(name='c', shape=(1, 1, 1), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        return tf.pow(self.gamma * dot + self.c, self.degree) + self.bias

class BioSphericalCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, initial_scale=10.0, **kwargs):
        super(BioSphericalCKN1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.initial_scale = initial_scale
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1, 1, 1),
                                    initializer=tf.constant_initializer(self.initial_scale), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        sq_inputs = tf.square(inputs)
        ones_k = tf.ones_like(self.kernel)
        patch_energy = tf.nn.conv1d(sq_inputs, ones_k, stride=1, padding='VALID')
        patch_norms = tf.sqrt(patch_energy + 1e-5)
        return self.scale * (dot / (patch_norms + 1e-5)) + self.bias

class GatedSphericalCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, initial_scale=10.0, **kwargs):
        super(GatedSphericalCKN1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.initial_scale = initial_scale
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0,1]), trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1,1,1), initializer=tf.constant_initializer(self.initial_scale), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        expected_energy = np.sqrt(self.kernel_size)
        self.tau = self.add_weight(name='tau', shape=(1,1,1), initializer=tf.constant_initializer(expected_energy*0.75), trainable=True)
        self.alpha = self.add_weight(name='alpha', shape=(1,1,1), initializer=tf.constant_initializer(3.0), constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        sq_inputs = tf.square(inputs)
        ones_k = tf.ones_like(self.kernel)
        patch_energy = tf.nn.conv1d(sq_inputs, ones_k, stride=1, padding='VALID')
        patch_norms = tf.sqrt(patch_energy + 1e-5)
        cosine_sim = dot / (patch_norms + 1e-5)
        gate = tf.nn.sigmoid(self.alpha * (patch_norms - self.tau))
        return self.scale * (cosine_sim * gate) + self.bias

class BioGaussianCKN1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(BioGaussianCKN1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size, input_shape[-1], self.filters),
                                     initializer='glorot_uniform', constraint=constraints.UnitNorm(axis=[0, 1]), trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1,1,1), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,1,1), initializer=tf.constant_initializer(0.5), constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID')
        sq_inputs = tf.square(inputs)
        ones_k = tf.ones_like(self.kernel)
        x_sq = tf.nn.conv1d(sq_inputs, ones_k, stride=1, padding='VALID')
        dist_sq = tf.nn.relu(x_sq + 1.0 - 2 * dot)
        return self.scale * tf.exp(-self.gamma * dist_sq) + self.bias

def build_model(model_type, n_filters=16):
    inputs = layers.Input(shape=(57,4))
    if model_type=="CNN":
        x = layers.Conv1D(n_filters,7,activation='softplus',padding='valid')(inputs)
    elif model_type=="Linear":
        x = BioLinearCKN1D(n_filters,7)(inputs)
    elif model_type=="Polynomial":
        x = BioPolynomialCKN1D(n_filters,7,degree=2)(inputs)
    elif model_type=="Pure Cosine":
        x = BioSphericalCKN1D(n_filters,7)(inputs)
    elif model_type=="Gated Cosine":
        x = GatedSphericalCKN1D(n_filters,7)(inputs)
    elif model_type=="Gaussian":
        x = BioGaussianCKN1D(n_filters,7)(inputs)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(1,activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

MODELS = ["CNN","Linear","Polynomial","Pure Cosine","Gated Cosine","Gaussian"]
RUNS = 100
EPOCHS = 60
FILTERS = 16
results = {m:{'acc':[],'auc':[]} for m in MODELS}

for i in range(RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,test_size=0.25,random_state=i,stratify=y_all)
    for m_name in MODELS:
        model = build_model(m_name,FILTERS)
        model.fit(X_train,y_train,epochs=EPOCHS,verbose=0,batch_size=8)
        loss, acc = model.evaluate(X_test,y_test,verbose=0)
        preds = model.predict(X_test,verbose=0)
        try: auc = roc_auc_score(y_test,preds)
        except: auc=0.5
        results[m_name]['acc'].append(acc)
        results[m_name]['auc'].append(auc)
    if (i+1)%10==0: print(f"Run {i+1} Done")

print("\n")
print(f"{'Model':<15} | {'Mean Acc':<10} | {'Mean AUC':<10} | {'Median AUC':<10} | {'AUC Std':<10} | {'95% CI Lower':<12} | {'95% CI Upper':<12}")
for m in MODELS:
    accs = results[m]['acc']
    aucs = results[m]['auc']
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    ci_lower = mean_auc - 1.96*std_auc/np.sqrt(RUNS)
    ci_upper = mean_auc + 1.96*std_auc/np.sqrt(RUNS)
    print(f"{m:<15} | {np.mean(accs):.3f} | {mean_auc:.3f} | {np.median(aucs):.3f} | {std_auc:.3f} | {ci_lower:.3f} | {ci_upper:.3f}")

plt.figure(figsize=(12,6))
plt.boxplot([results[m]['auc'] for m in MODELS],labels=MODELS,patch_artist=True)
plt.title("Bio-Realism Check: 100 Runs (AUC Scores)")
plt.ylabel("AUC")
plt.grid(True,alpha=0.3)
plt.show()
