# forgetting_mlp.py
# Author: vt485
# Analyzing Catastrophic Forgetting in MLPs (Permuted MNIST)

import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Configuration ----------------
GLOBAL_SEED = 12345
NUM_TASKS = 10
EPOCHS_FIRST_TASK = 50
EPOCHS_LATER_TASK = 20
BATCH_SIZE = 32
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
HIDDEN_UNITS = 256


MODEL_DEPTHS = [2]                
USE_DROPOUT_OPTIONS = [False]     
OPTIMIZERS = ["sgd", "adam"]     
LOSS_MODES = ["nll", "l1", "l2", "l1_l2"]
DROPOUT_RATE = 0.5

RESULTS_DIR = "results_forgetting"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Seed Setup ----------------
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_global_seed(GLOBAL_SEED)

# ---------------- Load MNIST ----------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape([-1, INPUT_SIZE])
x_test = x_test.reshape([-1, INPUT_SIZE])

VAL_SIZE = 10000
train_x, val_x = x_train[:-VAL_SIZE], x_train[-VAL_SIZE:]
train_y, val_y = y_train[:-VAL_SIZE], y_train[-VAL_SIZE:]

# ---------------- Permuted MNIST ----------------
def generate_permutations(num_tasks, seed):
    rng = np.random.RandomState(seed)
    return [rng.permutation(INPUT_SIZE) for _ in range(num_tasks)]

def permute_data(images, perm):
    return images[:, perm]

TASK_PERMS = generate_permutations(NUM_TASKS, GLOBAL_SEED)

def get_task_datasets(task_id):
    p = TASK_PERMS[task_id]
    tr = tf.data.Dataset.from_tensor_slices((permute_data(train_x, p), train_y)).shuffle(10000, seed=GLOBAL_SEED).batch(BATCH_SIZE)
    va = tf.data.Dataset.from_tensor_slices((permute_data(val_x, p), val_y)).batch(BATCH_SIZE)
    te = tf.data.Dataset.from_tensor_slices((permute_data(x_test, p), y_test)).batch(BATCH_SIZE)
    return tr, va, te

ALL_TASKS = [get_task_datasets(i) for i in range(NUM_TASKS)]

# ---------------- Model ----------------
def build_mlp(depth=2, dropout=False, loss_mode="nll"):
    if loss_mode == "l1":
        reg = tf.keras.regularizers.L1(1e-4)
    elif loss_mode == "l2":
        reg = tf.keras.regularizers.L2(1e-4)
    elif loss_mode == "l1_l2":
        reg = tf.keras.regularizers.L1L2(1e-4, 1e-4)
    else:
        reg = None

    layers = [tf.keras.Input(shape=(INPUT_SIZE,))]
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu', kernel_regularizer=reg))
        if dropout:
            layers.append(tf.keras.layers.Dropout(DROPOUT_RATE))
    layers.append(tf.keras.layers.Dense(NUM_CLASSES))
    return tf.keras.Sequential(layers)

def get_optimizer(name):
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=0.001)
    return tf.keras.optimizers.RMSprop(learning_rate=0.001)

# ---------------- Train / Eval ----------------
def train_step(model, opt, x, y):  # fixed: no @tf.function
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        loss_reg = tf.reduce_sum(model.losses) if model.losses else 0.0
        loss = loss_ce + loss_reg
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def eval_step(model, x, y):
    logits = model(x, training=False)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(y, tf.int32)), tf.float32))
    return correct, tf.cast(tf.shape(y)[0], tf.float32)

def evaluate(model, ds):
    total_c, total_n = 0.0, 0.0
    for xb, yb in ds:
        c, n = eval_step(model, xb, yb)
        total_c += c.numpy()
        total_n += n.numpy()
    return total_c / total_n

def train_on_task(model, opt, train_ds, val_ds, epochs, task, hist):
    for e in range(epochs):
        for xb, yb in train_ds:
            train_step(model, opt, xb, yb)
        val_acc = evaluate(model, val_ds)
        hist[task].append(val_acc)
        print(f"Task {task+1} | Epoch {e+1}/{epochs} | ValAcc={val_acc:.4f}")

# ---------------- Continual Training ----------------
def run_experiment(depth, dropout, opt_name, loss_mode):
    model = build_mlp(depth, dropout, loss_mode)
    opt = get_optimizer(opt_name)
    R = np.zeros((NUM_TASKS, NUM_TASKS), dtype=np.float32)
    hist = {t: [] for t in range(NUM_TASKS)}

    for t in range(NUM_TASKS):
        tr, va, te = ALL_TASKS[t]
        epochs = EPOCHS_FIRST_TASK if t == 0 else EPOCHS_LATER_TASK
        print(f"Training Task {t+1}/{NUM_TASKS} | depth={depth}, drop={dropout}, loss={loss_mode}, opt={opt_name}")
        train_on_task(model, opt, tr, va, epochs, t, hist)

        for j in range(t+1):
            _, _, te_j = ALL_TASKS[j]
            acc = evaluate(model, te_j)
            R[t, j] = acc
            print(f"After Task {t+1}, Test on Task {j+1}: {acc:.4f}")

    T = NUM_TASKS
    ACC = np.mean(R[T-1, :T])
    BWT = np.mean([R[T-1, i] - R[i, i] for i in range(T-1)])

    tag = f"D{depth}_drop{int(dropout)}_{loss_mode}_{opt_name}"
    np.save(os.path.join(RESULTS_DIR, f"R_{tag}.npy"), R)
    plt.figure()
    for t in range(NUM_TASKS):
        if hist[t]:
            plt.plot(hist[t], label=f"T{t+1}")
    plt.xlabel("Epochs")
    plt.ylabel("Val Acc")
    plt.title(tag)
    plt.legend(fontsize='x-small')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"val_{tag}.png"))
    plt.close()

    with open(os.path.join(RESULTS_DIR, f"metrics_{tag}.txt"), "w") as f:
        f.write(f"ACC={ACC:.4f}\nBWT={BWT:.4f}\n")

    print(f"Final ACC={ACC:.4f}, BWT={BWT:.4f}")

# ---------------- Main ----------------
if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    for d in MODEL_DEPTHS:
        for dr in USE_DROPOUT_OPTIONS:
            for o in OPTIMIZERS:
                for l in LOSS_MODES:
                    run_experiment(d, dr, o, l)
