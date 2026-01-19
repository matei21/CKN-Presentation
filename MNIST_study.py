import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, constraints
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

class LinearCKN(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(LinearCKN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME') + self.bias

class PolynomialCKN(layers.Layer):
    def __init__(self, filters, kernel_size, degree=2, **kwargs):
        super(PolynomialCKN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.degree = degree
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='ones', trainable=True)
        self.c = self.add_weight(name='c', shape=(1,), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME')
        return tf.pow(self.gamma * dot + self.c, self.degree) + self.bias

class SphericalCKN(layers.Layer):
    def __init__(self, filters, kernel_size, initial_scale=10.0, **kwargs):
        super(SphericalCKN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.initial_scale = initial_scale
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1, 1, 1, 1),
                                     initializer=tf.constant_initializer(self.initial_scale), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
    def call(self, inputs):
        dot_product = tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME')
        squared_inputs = tf.square(inputs)
        ones_kernel = tf.ones_like(self.kernel)
        patch_energy = tf.nn.conv2d(squared_inputs, ones_kernel, strides=1, padding='SAME')
        patch_norms = tf.sqrt(patch_energy + 1e-5)
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.kernel), axis=[0, 1, 2]) + 1e-5)
        cosine_sim = dot_product / (patch_norms * kernel_norm)
        return self.scale * cosine_sim + self.bias

class GaussianCKN(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(GaussianCKN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.filters),
                                     initializer='glorot_uniform', trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1, 1, 1, 1),
                                     initializer=tf.constant_initializer(1.0), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer=tf.constant_initializer(0.01),
                                    constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME')
        x_sq = tf.nn.conv2d(tf.square(inputs), tf.ones_like(self.kernel), strides=1, padding='SAME')
        w_sq = tf.reduce_sum(tf.square(self.kernel), axis=[0,1,2])
        dist_sq = tf.nn.relu(x_sq + w_sq - 2 * dot)
        return self.scale * tf.exp(-self.gamma * dist_sq) + self.bias

def build_cnn(filters1=32, filters2=64):
    inputs = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(filters1, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(filters2, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(0.001, global_clipnorm=1.0),
                 loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model(kernel_type):
    if kernel_type == "CNN":
        return build_cnn()
    inputs = layers.Input(shape=(28,28,1))
    if kernel_type == "Linear":
        x = LinearCKN(32, (3,3))(inputs)
    elif kernel_type == "Polynomial":
        x = PolynomialCKN(32, (3,3))(inputs)
    elif kernel_type == "Spherical":
        x = SphericalCKN(32, (3,3))(inputs)
    elif kernel_type == "Gaussian":
        x = GaussianCKN(32, (3,3))(inputs)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    if kernel_type == "Linear":
        x = LinearCKN(64, (3,3))(x)
    elif kernel_type == "Polynomial":
        x = PolynomialCKN(64, (3,3))(x)
    elif kernel_type == "Spherical":
        x = SphericalCKN(64, (3,3))(x)
    elif kernel_type == "Gaussian":
        x = GaussianCKN(64, (3,3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.22)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(0.001, global_clipnorm=1.0),
                 loss='categorical_crossentropy', metrics=['accuracy'])
    return model

MODELS = ["CNN", "Linear", "Polynomial", "Spherical", "Gaussian"]
RUNS = 100
EPOCHS = 10
BATCH_SIZE = 256
results = {m: {'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []} for m in MODELS}

for run in range(RUNS):
    print(f"Run {run+1}/{RUNS}")
    idx = np.random.permutation(len(x_train))
    x_train_r = x_train[idx]
    y_train_r = y_train_ohe[idx]
    for m_name in MODELS:
        model = build_model(m_name)
        model.fit(x_train_r, y_train_r, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 verbose=0, validation_split=0.1)
        preds_ohe = model.predict(x_test, verbose=0)
        preds = np.argmax(preds_ohe, axis=1)
        y_true = y_test
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')
        precision = precision_score(y_true, preds, average='weighted')
        recall = recall_score(y_true, preds, average='weighted')
        try:
            auc = roc_auc_score(y_test_ohe, preds_ohe, multi_class='ovr')
        except:
            auc = np.nan
        results[m_name]['acc'].append(acc)
        results[m_name]['f1'].append(f1)
        results[m_name]['precision'].append(precision)
        results[m_name]['recall'].append(recall)
        results[m_name]['auc'].append(auc)
        print(f"   {m_name:<10}: Acc={acc*100:.2f}%, F1={f1:.3f}, AUC={auc:.3f}")

print("\n")
for m in MODELS:
    accs = np.array(results[m]['acc'])
    f1s = np.array(results[m]['f1'])
    precs = np.array(results[m]['precision'])
    recs = np.array(results[m]['recall'])
    aucs = np.array(results[m]['auc'])
    ci_low = np.percentile(accs, 2.5)
    ci_high = np.percentile(accs, 97.5)
    print(f"{m:<12} | {np.mean(accs):.3f} | {np.std(accs):.3f} | {np.mean(f1s):.3f} | {np.mean(precs):.3f} | {np.mean(recs):.3f} | {np.nanmean(aucs):.3f} | {ci_low:.3f}-{ci_high:.3f}")

plt.figure(figsize=(12,6))
plt.boxplot([results[m]['acc'] for m in MODELS], labels=MODELS, patch_artist=True)
plt.title(f"MNIST Test Accuracy Distribution ({RUNS} runs)")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()
