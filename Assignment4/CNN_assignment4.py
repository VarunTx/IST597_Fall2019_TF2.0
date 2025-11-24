# -*- coding: utf-8 -*-

Lab 3 

Author:- vt485


import tensorflow as tf
import numpy as np
import os, json, csv
import matplotlib.pyplot as plt




tf.random.set_seed(1234)
np.random.seed(1234)

batch_size = 64
hidden_size = 100
num_classes = 10
num_epochs = 5
learning_rate = 0.01

results_dir = "normalization_results"
os.makedirs(results_dir, exist_ok=True)

# ------------------------------
# DATA
# ------------------------------

(x_train, y_train_raw), (x_test, y_test_raw) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]
y_train = tf.one_hot(y_train_raw, depth=num_classes)
y_test  = tf.one_hot(y_test_raw,  depth=num_classes)

# ------------------------------
# NORMALIZATIONS
# ------------------------------

def batch_norm(x, g, b, eps=1e-5):
    m = tf.reduce_mean(x, axis=0)
    v = tf.reduce_mean(tf.square(x - m), axis=0)
    xhat = (x - m) / tf.sqrt(v + eps)
    return g * xhat + b

def layer_norm(x, g, b, eps=1e-5):
    m = tf.reduce_mean(x, axis=1, keepdims=True)
    v = tf.reduce_mean(tf.square(x - m), axis=1, keepdims=True)
    xhat = (x - m) / tf.sqrt(v + eps)
    g = tf.reshape(g, [1, -1])
    b = tf.reshape(b, [1, -1])
    return g * xhat + b

def weight_norm_dense(v, g, eps=1e-5):
    vnorm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=0) + eps)
    return v * (g / vnorm)

# ------------------------------
# CNN
# ------------------------------

class CNN(tf.Module):
    def __init__(self, hidden_size, output_size, mode=None):
        super().__init__()
        self.mode = mode

        self.W1 = tf.Variable(tf.random.normal([5,5,1,32], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([32]))

        flat = 14*14*32

        self.W2 = tf.Variable(tf.random.normal([flat, hidden_size], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([hidden_size]))
        self.V2 = tf.Variable(tf.random.normal([flat, hidden_size], stddev=0.1))
        self.g2 = tf.Variable(tf.ones([hidden_size]))

        self.W3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([output_size]))
        self.V3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.g3 = tf.Variable(tf.ones([output_size]))

        self.gamma_h = tf.Variable(tf.ones([hidden_size]))
        self.beta_h  = tf.Variable(tf.zeros([hidden_size]))

    def conv_pool(self, X):
        c = tf.nn.conv2d(X, self.W1, strides=[1,1,1,1], padding="SAME")
        c = tf.nn.bias_add(c, self.b1)
        c = tf.nn.relu(c)
        p = tf.nn.max_pool2d(c, 2, 2, padding="SAME")
        return p

    def __call__(self, X, training=True):
        h = self.conv_pool(X)
        h = tf.reshape(h, [tf.shape(h)[0], -1])

        if self.mode == "weight":
            W2 = weight_norm_dense(self.V2, self.g2)
        else:
            W2 = self.W2

        z = tf.matmul(h, W2) + self.b2

        if self.mode == "batch":
            z = batch_norm(z, self.gamma_h, self.beta_h)
        elif self.mode == "layer":
            z = layer_norm(z, self.gamma_h, self.beta_h)

        h2 = tf.nn.relu(z)

        if self.mode == "weight":
            W3 = weight_norm_dense(self.V3, self.g3)
        else:
            W3 = self.W3

        return tf.matmul(h2, W3) + self.b3

# ------------------------------
# LOSS / ACCURACY
# ------------------------------

def loss_fn(model, x, y):
    logits = model(x)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

def accuracy_fn(model, x, y, bs=256):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(bs)
    correct = 0
    total = 0
    for xb, yb in ds:
        logits = model(xb, training=False)
        preds = tf.argmax(logits, axis=1)
        labels = tf.argmax(yb, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()
        total += xb.shape[0]
    return correct / total

# ------------------------------
# TRAIN
# ------------------------------

def train_model(mode):
    print("\n=== MODE:", mode.upper(), "===")
    norm = None if mode=="none" else mode
    model = CNN(hidden_size, num_classes, mode=norm)
    opt = tf.keras.optimizers.SGD(learning_rate)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    metrics = []

    for epoch in range(1, num_epochs+1):
        tot, num = 0,0
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                loss = loss_fn(model, xb, yb)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            bs = xb.shape[0]
            tot += loss.numpy()*bs
            num += bs

        L = tot/num
        tr = accuracy_fn(model, x_train, y_train)*100
        te = accuracy_fn(model, x_test,  y_test)*100

        print(f"Epoch {epoch} | Loss={L:.4f} | Train={tr:.2f}% | Test={te:.2f}%")
        metrics.append({"epoch":epoch, "loss":float(L), "train":float(tr), "test":float(te)})

    return model, metrics

# ------------------------------
# RUN
# ------------------------------

modes = ["none", "batch", "layer", "weight"]
allm = {}

for mode in modes:
    _, m = train_model(mode)
    allm[mode] = m
    with open(f"{results_dir}/results_{mode}.json","w") as f:
        json.dump(m, f, indent=2)

with open(f"{results_dir}/summary.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode","loss","train_acc","test_acc"])
    for mode in modes:
        last = allm[mode][-1]
        w.writerow([mode,last["loss"],last["train"],last["test"]])

# ------------------------------
# PLOTS
# ------------------------------

plt.figure()
for mode in modes:
    plt.plot([x["epoch"] for x in allm[mode]],
             [x["loss"]  for x in allm[mode]], label=mode)
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.savefig(f"{results_dir}/loss_curves.png"); plt.close()

plt.figure()
for mode in modes:
    plt.plot([x["epoch"] for x in allm[mode]],
             [x["test"]  for x in allm[mode]], label=mode)
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)")
plt.savefig(f"{results_dir}/accuracy_curves.png"); plt.close()

print("\nSaved all outputs in:", results_dir)

