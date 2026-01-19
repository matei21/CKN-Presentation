import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, constraints

SEQ_LEN = 101  # MUST match training


# =========================
# 1. BIO CKN LAYERS
# =========================

@tf.keras.utils.register_keras_serializable()
class BioLinearCKN1D(layers.Layer):
    def __init__(self, filters=32, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.k = self.add_weight(
            name="k",
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer="glorot_uniform",
        )
        self.b = self.add_weight(name="b", shape=(self.filters,), initializer="zeros")

    def call(self, inputs):
        return tf.nn.conv1d(inputs, self.k, stride=1, padding="VALID") + self.b


@tf.keras.utils.register_keras_serializable()
class BioPolynomialCKN1D(layers.Layer):
    def __init__(self, filters=32, kernel_size=7, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.degree = degree

    def build(self, input_shape):
        self.k = self.add_weight(name="k",
                                 shape=(self.kernel_size, input_shape[-1], self.filters))
        self.b = self.add_weight(name="b", shape=(self.filters,))
        self.g = self.add_weight(name="g", shape=(1, 1, 1), initializer="ones")
        self.c = self.add_weight(name="c", shape=(1, 1, 1), initializer="zeros")

    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.k, stride=1, padding="VALID")
        return tf.pow(self.g * dot + self.c, self.degree) + self.b


@tf.keras.utils.register_keras_serializable()
class BioNormalizedPolynomialCKN1D(layers.Layer):
    def __init__(self, filters=32, kernel_size=7, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.degree = degree

    def build(self, input_shape):
        self.k = self.add_weight(name="k",
                                 shape=(self.kernel_size, input_shape[-1], self.filters))
        self.b = self.add_weight(name="b", shape=(self.filters,))
        self.g = self.add_weight(name="g", shape=(1, 1, 1), initializer="ones")
        self.c = self.add_weight(name="c", shape=(1, 1, 1), initializer="zeros")

    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.k, stride=1, padding="VALID")
        norm_x = tf.sqrt(
            tf.nn.conv1d(tf.square(inputs),
                         tf.ones_like(self.k),
                         stride=1,
                         padding="VALID") + 1e-7
        )
        return tf.pow(self.g * (dot / norm_x) + self.c, self.degree) + self.b


@tf.keras.utils.register_keras_serializable()
class BioSphericalCKN1D(layers.Layer):
    def __init__(self, filters=32, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.k = self.add_weight(
            name="k",
            shape=(self.kernel_size, input_shape[-1], self.filters),
            constraint=constraints.UnitNorm(axis=[0, 1])
        )
        self.s = self.add_weight(name="s", shape=(1, 1, 1),
                                 initializer=tf.constant_initializer(10.0))
        self.b = self.add_weight(name="b", shape=(self.filters,))

    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.k, stride=1, padding="VALID")
        norm_x = tf.sqrt(
            tf.nn.conv1d(tf.square(inputs),
                         tf.ones_like(self.k),
                         stride=1,
                         padding="VALID") + 1e-7
        )
        return self.s * (dot / norm_x) + self.b


@tf.keras.utils.register_keras_serializable()
class BioGaussianCKN1D(layers.Layer):
    def __init__(self, filters=32, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.k = self.add_weight(
            name="k",
            shape=(self.kernel_size, input_shape[-1], self.filters),
            constraint=constraints.UnitNorm(axis=[0, 1])
        )
        self.s = self.add_weight(name="s", shape=(1, 1, 1),
                                 initializer=tf.constant_initializer(10.0))
        self.g = self.add_weight(name="g", shape=(1, 1, 1),
                                 initializer=tf.constant_initializer(1.0))
        self.b = self.add_weight(name="b", shape=(self.filters,))

    def call(self, inputs):
        dot = tf.nn.conv1d(inputs, self.k, stride=1, padding="VALID")
        norm_x = tf.sqrt(
            tf.nn.conv1d(tf.square(inputs),
                         tf.ones_like(self.k),
                         stride=1,
                         padding="VALID") + 1e-7
        )
        dist_sq = 2.0 - 2.0 * (dot / norm_x)
        return self.s * tf.exp(-self.g * tf.nn.relu(dist_sq)) + self.b


# =========================
# 2. INPUT PROCESSING
# =========================

def process_input(sequence: str, dataset: str):
    vocab = "ACGT" if dataset == "Ecoli" else "ACGU"
    seq = sequence.upper().strip()

    if dataset != "Ecoli":
        seq = seq.replace("T", "U")

    char_map = {c: i for i, c in enumerate(vocab)}
    tokens = [char_map.get(c, 0) for c in seq[:SEQ_LEN]]
    tokens += [0] * (SEQ_LEN - len(tokens))

    return tf.one_hot([tokens], depth=len(vocab))


# =========================
# 3. PREDICTION + VIZ
# =========================

def extract_viz_data(model, input_text, dataset, model_type):
    X = process_input(input_text, dataset)

    preds = model.predict(X, verbose=0)[0]
    prob = float(preds[0])

    # Locate first bio layer
    target = next(
        (l for l in model.layers if isinstance(l, layers.Layer) and hasattr(l, "k")),
        None
    )

    heatmap = []
    arcs = []
    vectors = []

    if target is not None:
        sub = tf.keras.Model(model.input, target.output)
        act = sub.predict(X, verbose=0)[0]
        heatmap = np.max(act, axis=-1).tolist()

        weights = target.get_weights()
        if "Polynomial" in model_type and len(weights) > 0:
            arr = np.array(heatmap)
            th = np.percentile(arr, 85)
            idxs = np.where(arr > th)[0]
            for i in range(len(idxs) - 1):
                if abs(idxs[i] - idxs[i + 1]) < 20:
                    arcs.append({
                        "start": int(idxs[i]),
                        "end": int(idxs[i + 1]),
                        "strength": float(arr[idxs[i]])
                    })

        if "Spherical" in model_type and len(weights) > 0:
            W = weights[0]
            for f in range(min(8, W.shape[-1])):
                vectors.append(W[W.shape[0] // 2, :, f][:3].tolist())

    return {
        "dataset": dataset,
        "model_type": model_type,
        "prediction": prob,
        "label": "Positive" if prob > 0.5 else "Negative",
        "heatmap": heatmap,
        "polynomial_arcs": arcs,
        "sphere_vectors": vectors,
        "gaussian_dist": float(1.0 - prob) if "Gaussian" in model_type else None
    }


def create_dummy_model():
    inp = layers.Input(shape=(SEQ_LEN, 4))
    x = BioLinearCKN1D(8, 5)(inp)
    x = layers.GlobalMaxPooling1D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)
