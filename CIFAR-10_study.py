import numpy as np
import tensorflow as tf
import time
import os
import zipfile
from tensorflow.keras import layers, models, constraints
from tensorflow.keras.utils import to_categorical
from google.colab import files

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except: pass
check_gpu()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class LinearCKN(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(3, 3, input_shape[-1], self.filters),
                                     initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer="zeros", trainable=True)
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=1, padding="SAME") + self.bias

class PolynomialCKN(layers.Layer):
    def __init__(self, filters, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.degree = degree
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(3, 3, input_shape[-1], self.filters),
                                     initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer="zeros", trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1,), initializer="ones", trainable=True)
        self.c = self.add_weight(name="c", shape=(1,), initializer="zeros", trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv2d(inputs, self.kernel, strides=1, padding="SAME")
        return tf.pow(self.gamma * dot + self.c, self.degree) + self.bias

class SphericalCKN(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(3, 3, input_shape[-1], self.filters),
                                     initializer="glorot_uniform",
                                     constraint=constraints.UnitNorm(axis=[0, 1, 2]),
                                     trainable=True)
        self.scale = self.add_weight(name="s", shape=(1,), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer="zeros", trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv2d(inputs, self.kernel, strides=1, padding="SAME")
        norm_x = tf.sqrt(tf.nn.conv2d(tf.square(inputs), tf.ones_like(self.kernel), strides=1, padding="SAME") + 1e-5)
        norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.kernel), axis=[0,1,2]) + 1e-5)
        return self.scale * (dot / (norm_x * norm_w)) + self.bias

class GaussianCKN(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
    def build(self, input_shape):
        self.kernel = self.add_weight(name="k", shape=(3, 3, input_shape[-1], self.filters),
                                     initializer="glorot_uniform",
                                     constraint=constraints.UnitNorm(axis=[0, 1, 2]),
                                     trainable=True)
        self.scale = self.add_weight(name="s", shape=(1,), initializer=tf.constant_initializer(10.0), trainable=True)
        self.bias = self.add_weight(name="b", shape=(self.filters,), initializer="zeros", trainable=True)
        self.gamma = self.add_weight(name="g", shape=(1,), initializer=tf.constant_initializer(0.01),
                                    constraint=constraints.NonNeg(), trainable=True)
    def call(self, inputs):
        dot = tf.nn.conv2d(inputs, self.kernel, strides=1, padding="SAME")
        x_sq = tf.nn.conv2d(tf.square(inputs), tf.ones_like(self.kernel), strides=1, padding="SAME")
        w_sq = tf.reduce_sum(tf.square(self.kernel), axis=[0,1,2])
        dist = tf.nn.relu(x_sq + w_sq - 2*dot)
        return self.scale * tf.exp(-self.gamma * dist) + self.bias

def build_model(layer_type):
    inputs = layers.Input(shape=(32, 32, 3))
    def get_layer(filters):
        if layer_type == "CNN": return layers.Conv2D(filters, (3,3), padding='same')
        if layer_type == "Linear": return LinearCKN(filters)
        if layer_type == "Polynomial": return PolynomialCKN(filters)
        if layer_type == "Spherical": return SphericalCKN(filters)
        if layer_type == "Gaussian": return GaussianCKN(filters)
    x = get_layer(32)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softplus')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = get_layer(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softplus')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = get_layer(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softplus')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='softplus')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

MODELS = ["CNN", "Linear", "Polynomial", "Spherical", "Gaussian"]
RUNS = 3
EPOCHS = 25
results = {m: {'acc': [], 'train_time': [], 'infer_time': [], 'params': 0} for m in MODELS}
best_acc_tracker = {m: 0.0 for m in MODELS}

for m in MODELS:
    dummy = build_model(m)
    results[m]['params'] = dummy.count_params()
    for run in range(RUNS):
        tf.keras.backend.clear_session()
        model = build_model(m)
        start_train = time.time()
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=0)
        total_train_time = time.time() - start_train
        avg_epoch_time = total_train_time / EPOCHS
        start_infer = time.time()
        model.predict(x_test, batch_size=128, verbose=0)
        total_infer_time = time.time() - start_infer
        acc = model.evaluate(x_test, y_test, verbose=0)[1]
        results[m]['acc'].append(acc)
        results[m]['train_time'].append(avg_epoch_time)
        results[m]['infer_time'].append(total_infer_time)
        if acc > best_acc_tracker[m]:
            best_acc_tracker[m] = acc
            fname = f"CIFAR_{m}_best.keras"
            model.save(fname)

for m in MODELS:
    accs = np.array(results[m]['acc']) * 100
    train_times = np.array(results[m]['train_time'])
    infer_times = np.array(results[m]['infer_time']) * 1000
    params = results[m]['params']
    mean_acc = np.mean(accs)
    mean_train = np.mean(train_times)
    mean_infer = np.mean(infer_times)
    train_eff = 1e9 / (params * (mean_train * 1000))
    infer_eff = 1e9 / (params * mean_infer)
    print(f"{m:<12} | {mean_acc:.2f}% | {mean_train:.2f}s | {mean_infer:.0f}ms | {train_eff:.2f} | {infer_eff:.2f}")
