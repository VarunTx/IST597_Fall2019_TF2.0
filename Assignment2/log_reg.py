# author: vt485
# TF2 eager, NO Keras layers. Logistic regression on Fashion-MNIST.

import argparse, random, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# minimal deps: tensorflow-datasets; sklearn is optional for extras
try:
    import tensorflow_datasets as tfds
except Exception as e:
    raise SystemExit("Please install tensorflow-datasets: pip install tensorflow-datasets") from e

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    HAVE_SK = True
except Exception:
    TSNE = KMeans = accuracy_score = LinearSVC = RandomForestClassifier = None
    HAVE_SK = False

# -------- utils --------
def seed_from_name(name: str) -> int:
    s = "".join(str(ord(c)) for c in (name or "Student"))
    return int(s) % 2_147_483_647

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def ensure_dirs():
    Path("plots").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)

def select_device(choice: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if choice == "gpu" and gpus: return "/GPU:0"
    if choice == "cpu" or (choice == "gpu" and not gpus): return "/CPU:0"
    return "/GPU:0" if gpus else "/CPU:0"

def save_rows(path, rows, header):
    with open(path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)

# -------- tiny manual optimizers --------
class Optimizer:  # minimal interface
    def apply(self, vars_, grads): raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr): self.lr = lr
    def apply(self, vars_, grads):
        for v, g in zip(vars_, grads): v.assign_sub(self.lr * g)

class Momentum(Optimizer):
    def __init__(self, lr, m=0.9): self.lr=lr; self.m=m; self.v={}
    def apply(self, vars_, grads):
        for v,g in zip(vars_,grads):
            k=id(v); vel=self.v.get(k, tf.zeros_like(v))
            vel = self.m*vel + g
            v.assign_sub(self.lr * vel); self.v[k]=vel

class RMSProp(Optimizer):
    def __init__(self, lr, rho=0.9, eps=1e-8): self.lr=lr; self.rho=rho; self.eps=eps; self.s={}
    def apply(self, vars_, grads):
        for v,g in zip(vars_,grads):
            k=id(v); s=self.s.get(k, tf.zeros_like(v))
            s = self.rho*s + (1-self.rho)*tf.square(g)
            v.assign_sub(self.lr * g / (tf.sqrt(s)+self.eps)); self.s[k]=s

class Adam(Optimizer):
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.m={}; self.v={}; self.t=0
    def apply(self, vars_, grads):
        self.t += 1
        for v,g in zip(vars_,grads):
            k=id(v)
            m=self.m.get(k, tf.zeros_like(v))
            v2=self.v.get(k, tf.zeros_like(v))
            m = self.b1*m + (1-self.b1)*g
            v2= self.b2*v2+ (1-self.b2)*tf.square(g)
            mhat = m / (1 - self.b1**self.t)
            vhat = v2/ (1 - self.b2**self.t)
            v.assign_sub(self.lr * mhat / (tf.sqrt(vhat)+self.eps))
            self.m[k]=m; self.v[k]=v2

def make_opt(name, lr, momentum):
    name = name.lower()
    if name=="sgd": return SGD(lr)
    if name=="momentum": return Momentum(lr, momentum)
    if name=="rmsprop": return RMSProp(lr)
    if name=="adam": return Adam(lr)
    raise ValueError("unknown optimizer")

# -------- data --------
def load_fmnist(val_count=5_000):
    (ds_train_full, ds_test), info = tfds.load(
        "fashion_mnist", split=["train", "test"], as_supervised=True, with_info=True
    )
    total = info.splits["train"].num_examples
    val_count = min(val_count, total-1)
    train_count = total - val_count

    def _prep(x,y):
        x = tf.cast(x, tf.float32)/255.0     # (28,28,1)
        x = tf.reshape(x, [28*28])           # flatten for LR
        y = tf.cast(y, tf.int32)
        return x,y

    tr = ds_train_full.take(train_count).map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    va = ds_train_full.skip(train_count).take(val_count).map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    te = ds_test.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return tr, va, te, train_count, val_count

def batches(ds, bs, shuffle=False, seed=0):
    if shuffle: ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

# -------- model --------
class LogReg:
    def __init__(self, l2=1e-4, seed=0):
        rng = np.random.RandomState(seed)
        self.W = tf.Variable(rng.normal(0,0.01,size=(784,10)).astype(np.float32))
        self.b = tf.Variable(np.zeros((10,), dtype=np.float32))
        self.l2 = l2

    def logits(self, x):  # x: (B,784)
        return tf.matmul(x, self.W) + self.b

    def loss(self, x, y):
        lg = self.logits(x)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=lg)
        reg = self.l2 * tf.nn.l2_loss(self.W)
        return tf.reduce_mean(ce) + reg

    def accuracy(self, x, y):
        preds = tf.argmax(self.logits(x), axis=1, output_type=tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

# -------- train/eval --------
def run_epoch(model, ds, opt):
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for xb, yb in ds:
        if opt is None:
            loss = model.loss(xb,yb); acc = model.accuracy(xb,yb)
        else:
            with tf.GradientTape() as tape:
                loss = model.loss(xb,yb)
            grads = tape.gradient(loss, [model.W, model.b])
            opt.apply([model.W, model.b], grads)
            acc = model.accuracy(xb,yb)
        bs = int(xb.shape[0])
        tot_loss += float(loss)*bs
        tot_acc  += float(acc)*bs
        n += bs
    return tot_loss/n, tot_acc/n

# -------- plots --------
def plot_curve(values, title, ylabel, path_png):
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(1,len(values)+1), values)
    plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel); plt.grid(True, alpha=0.25)
    plt.savefig(path_png, bbox_inches="tight"); plt.close()

def plot_sample_images(ds_test_flat, W, B, path_img, count=9, img_h=28, img_w=28):
    xs_flat, ys = [], []
    for x,y in ds_test_flat.take(count):
        x = x.numpy()           # (784,) flattened by loader
        ys.append(int(y.numpy()))
        if x.ndim == 1:
            img = x.reshape(img_h, img_w)
        elif x.ndim == 3 and x.shape[-1] == 1:
            img = x[...,0]
        else:
            img = x
        xs_flat.append(img.reshape(-1))
    X = np.stack(xs_flat).astype(np.float32)        # (count, 784)
    logits = X @ W.numpy() + B.numpy()
    preds = np.argmax(logits, axis=1)

    fig, axes = plt.subplots(3,3, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(img_h, img_w), cmap="binary")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"True: {ys[i]}, Pred: {int(preds[i])}")
    plt.tight_layout(); plt.savefig(path_img, bbox_inches="tight"); plt.close()

def plot_weight_images(W, path_png):
    w = W.numpy()
    vmin, vmax = np.percentile(w,1), np.percentile(w,99)
    fig, axes = plt.subplots(2,5, figsize=(10,4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(w[:,i].reshape(28,28), cmap="seismic", vmin=vmin, vmax=vmax)
        ax.set_title(f"class {i}"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path_png, bbox_inches="tight"); plt.close()

def plot_confusion(cm, path_png):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion matrix")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.colorbar()
    plt.tight_layout(); plt.savefig(path_png, bbox_inches="tight"); plt.close()

def tsne_on_weights(W, path_png):
    if TSNE is None:
        print("[Skip] scikit-learn not installed; cannot run t-SNE.")
        return
    pts = TSNE(n_components=2, init="random", learning_rate="auto",
               perplexity=3, random_state=0).fit_transform(W.numpy().T)
    plt.figure(figsize=(6,5))
    plt.scatter(pts[:,0], pts[:,1])
    for i in range(10): plt.text(pts[i,0]+0.5, pts[i,1]+0.5, str(i))
    plt.title("t-SNE of class weight vectors"); plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight"); plt.close()

# -------- sklearn baselines (optional) --------
def sklearn_baselines(ds_train_flat, ds_val_flat, limit=5000, path_csv="results/p2_sklearn.csv"):
    if not HAVE_SK:
        print("[Skip] scikit-learn not installed; cannot run SVM/RF baselines.")
        return
    def to_numpy(ds, cap):
        Xs, Ys, c = [], [], 0
        for x,y in ds:
            Xs.append(x.numpy().reshape(-1)); Ys.append(int(y.numpy())); c += 1
            if c>=cap: break
        return np.stack(Xs), np.array(Ys)
    Xtr, Ytr = to_numpy(ds_train_flat, limit)
    Xva, Yva = to_numpy(ds_val_flat, min(limit, len(Xtr)))

    rows=[]
    svm = LinearSVC(max_iter=2000)
    svm.fit(Xtr, Ytr)
    acc1 = accuracy_score(Yva, svm.predict(Xva)); rows.append(["LinearSVM", acc1])

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(Xtr, Ytr)
    acc2 = accuracy_score(Yva, rf.predict(Xva)); rows.append(["RandomForest", acc2])

    save_rows(path_csv, rows, ["model","val_accuracy"])
    print(f"[Saved] {path_csv}")

# -------- main --------
def main(args):
    ensure_dirs()
    seed = seed_from_name(args.seed_name); set_global_seed(seed)
    device = select_device(args.device)
    print(f"[Device] requested={args.device} -> using {device}")

    with tf.device(device):
        ds_tr_flat, ds_va_flat, ds_te_flat, n_tr, n_va = load_fmnist(args.val_split)
        ds_tr = batches(ds_tr_flat, args.batch_size, shuffle=True, seed=seed)
        ds_va = batches(ds_va_flat, args.batch_size)
        ds_te = batches(ds_te_flat, args.batch_size)
        print(f"Sizes → train: ({n_tr}, 784)  val: ({n_va}, 784)  test: (10000, 784)")

        model = LogReg(l2=args.l2, seed=seed)
        opt = make_opt(args.opt, args.lr, args.momentum)

        tr_losses, va_losses, tr_accs, va_accs, epoch_times = [], [], [], [], []
        best_va = float("inf"); patience_left = args.early_stop_patience

        for epoch in range(1, args.epochs+1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = run_epoch(model, ds_tr, opt)
            va_loss, va_acc = run_epoch(model, ds_va, None)
            dt = time.perf_counter() - t0

            tr_losses.append(tr_loss); va_losses.append(va_loss)
            tr_accs.append(tr_acc);   va_accs.append(va_acc)
            epoch_times.append(dt)

            print(f"Epoch {epoch:2d}/{args.epochs} - time {dt:.2f}s | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            if args.early_stop_patience>0:
                if va_loss < best_va - 1e-6:
                    best_va = va_loss; patience_left = args.early_stop_patience
                else:
                    patience_left -= 1
                    if patience_left<=0:
                        print("Early stopping."); break

        # test
        _, te_acc = run_epoch(model, ds_te, None)
        print(f"Final Test Accuracy: {te_acc:.4f}")

        # save curves & timing
        rows = [[i+1, tr_losses[i], va_losses[i], tr_accs[i], va_accs[i], epoch_times[i]]
                for i in range(len(epoch_times))]
        save_rows("results/p2_epoch_times.csv", rows,
                  ["epoch","train_loss","val_loss","train_acc","val_acc","elapsed_sec"])
        plot_curve(tr_accs,  "Train accuracy",        "accuracy", "plots/p2_train_acc.png")
        plot_curve(va_accs,  "Validation accuracy",   "accuracy", "plots/p2_val_acc.png")
        plot_curve(tr_losses,"Train loss",            "loss",     "plots/p2_train_loss.png")
        plot_curve(va_losses,"Validation loss",       "loss",     "plots/p2_val_loss.png")
        plot_curve(epoch_times,"Per-epoch time",      "sec",      "plots/p2_timing.png")

        # visuals
        plot_sample_images(ds_te_flat, model.W, model.b, "plots/p2_sample_images.png")
        plot_weight_images(model.W, "plots/p2_weight_images.png")

        # confusion matrix (no sklearn needed)
        cm = np.zeros((10,10), dtype=int)
        for xb, yb in ds_te:
            pred = tf.argmax(model.logits(xb), axis=1, output_type=tf.int32).numpy()
            y    = yb.numpy()
            for t,p in zip(y,pred): cm[t,p] += 1
        np.savetxt("results/p2_confusion_matrix.txt", cm, fmt="%d")
        plot_confusion(cm, "plots/p2_confusion_matrix.png")
        print("Confusion matrix saved to results/p2_confusion_matrix.txt")

        # optional extras
        if args.tsne: tsne_on_weights(model.W, "plots/p2_tsne_weights.png")
        if args.run_svm or args.run_rf: sklearn_baselines(ds_tr_flat, ds_va_flat, limit=args.sk_limit)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TF2 eager logistic regression (no Keras)")
    ap.add_argument("--seed_name", type=str, default="Student")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", type=str, choices=["sgd","momentum","rmsprop","adam"], default="adam")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--val_split", type=int, default=5000)
    ap.add_argument("--early_stop_patience", type=int, default=0)
    ap.add_argument("--device", type=str, choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--run_svm", action="store_true")
    ap.add_argument("--run_rf", action="store_true")
    ap.add_argument("--sk_limit", type=int, default=5000)
    args = ap.parse_args()
    main(args)
