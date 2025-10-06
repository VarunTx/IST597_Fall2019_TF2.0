# author: vt485
# TensorFlow 2.x eager, NO Keras. Manual GD with tf.GradientTape.

import argparse, random, time, csv
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------ utils ------------------------
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

# ------------------------ losses ------------------------
def mse_loss(y, yhat): return tf.reduce_mean(tf.square(y - yhat))
def l1_loss(y, yhat):  return tf.reduce_mean(tf.abs(y - yhat))

def huber_loss(y, yhat, delta=1.0):
    r = tf.abs(y - yhat)
    quad = tf.minimum(r, delta)
    lin  = r - quad
    return tf.reduce_mean(0.5 * tf.square(quad) + delta * lin)

def hybrid_loss(y, yhat, lam=0.5):
    return lam * l1_loss(y, yhat) + (1.0 - lam) * mse_loss(y, yhat)

# ------------------------ data ------------------------
def gen_data(n, x_dist="normal", noise_type="gaussian", noise_std=0.6,
             noise_x=False, noise_y=True, seed=0):
    rnd = np.random.RandomState(seed)
    if x_dist == "normal":
        x = rnd.normal(0.0, 2.0, size=(n, 1)).astype(np.float32)   # spread nicely
    else:
        x = rnd.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float32)

    if noise_x and noise_type != "none":
        x += _sample_noise(rnd, noise_type, noise_std, x.shape)

    y = 3.0 * x + 2.0
    if noise_y and noise_type != "none":
        y += _sample_noise(rnd, noise_type, noise_std, y.shape)

    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

def _sample_noise(rnd, kind, std, shape):
    if kind == "gaussian": return rnd.normal(0, std, size=shape).astype(np.float32)
    if kind == "uniform":  return rnd.uniform(-std, std, size=shape).astype(np.float32)
    if kind == "laplace":  return rnd.laplace(0, std, size=shape).astype(np.float32)
    return np.zeros(shape, dtype=np.float32)

# ------------------------ training ------------------------
def train(args):
    ensure_dirs()
    seed = seed_from_name(args.seed_name); set_global_seed(seed)
    device = select_device(args.device)
    print(f"[Device] requested={args.device} -> using {device}")

    with tf.device(device):
        x, y = gen_data(args.num_examples, args.x_dist, args.noise_type, args.noise_std,
                        args.noise_x, args.noise_y, seed=seed)

        # Scalar parameters (broadcast over N×1 inputs)
        W = tf.Variable(args.init_w, dtype=tf.float32)
        B = tf.Variable(args.init_b, dtype=tf.float32)

        def predict(x_): return x_ * W + B

        def loss_fn(y_true, y_pred):
            if args.loss == "mse":    return mse_loss(y_true, y_pred)
            if args.loss == "l1":     return l1_loss(y_true, y_pred)
            if args.loss == "huber":  return huber_loss(y_true, y_pred, args.delta)
            if args.loss == "hybrid": return hybrid_loss(y_true, y_pred, args.lambda_l1)
            raise ValueError("unknown loss")

        lr = args.lr
        best = float("inf")
        patience_left = args.patience
        rng = np.random.RandomState(seed + 7)

        losses, times = [], []

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            with tf.GradientTape() as tape:
                yhat = predict(x)
                loss = loss_fn(y, yhat)
            dW, dB = tape.gradient(loss, [W, B])
            W.assign_sub(lr * dW); B.assign_sub(lr * dB)

            # Optional LR jitter
            if args.lr_jitter_every > 0 and epoch % args.lr_jitter_every == 0:
                lr = max(1e-6, lr + float(rng.normal(0, args.lr_jitter_std)))

            # Optional noise to parameters
            if args.noise_wb and args.noise_type != "none":
                nW = _sample_noise(rng, args.noise_type, args.noise_std, (1,)).astype(np.float32)[0]
                nB = _sample_noise(rng, args.noise_type, args.noise_std, (1,)).astype(np.float32)[0]
                W.assign_add(nW); B.assign_add(nB)

            cur = float(loss.numpy()); losses.append(cur)
            dt = time.perf_counter() - t0; times.append(dt)

            # Patience scheduler (halve LR when not improving)
            if cur < best - 1e-9:
                best = cur; patience_left = args.patience
            else:
                patience_left -= 1
                if args.patience > 0 and patience_left <= 0:
                    lr = max(lr * 0.5, 1e-6); patience_left = args.patience

            if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
                print(f"Step {epoch:4d}: Loss={cur:.6f} | lr={lr:.6g} | "
                      f"W={W.numpy():.4f} | B={B.numpy():.4f} | {dt*1000:.1f}ms")

        # ---- save timings ----
        with open(Path("results") / "p1_epoch_times.csv", "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(["epoch","loss","elapsed_sec"])
            for i,(l,t) in enumerate(zip(losses, times), 1): cw.writerow([i,l,t])
        print("[Saved] results/p1_epoch_times.csv")

        # ---- pretty plots (not too dense) ----
        x_np = x.numpy().reshape(-1); y_np = y.numpy().reshape(-1)

        # subsample for scatter to avoid overplotting
        k = min(args.plot_points, x_np.shape[0])
        idx = np.random.choice(x_np.shape[0], k, replace=False)
        xs, ys = x_np[idx], y_np[idx]

        # sorted line for a clean overlay
        x_line = np.linspace(x_np.min()-0.2, x_np.max()+0.2, 300, dtype=np.float32)
        y_line = x_line * W.numpy() + B.numpy()

        plt.figure(figsize=(7,5.2))
        plt.scatter(xs, ys, s=18, alpha=0.45, label="data")
        plt.plot(x_line, y_line, linewidth=2.0, label="fit")
        plt.grid(True, alpha=0.25)
        plt.title(f"Linear fit ({args.loss}, x={args.x_dist})")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend()
        p1 = Path("plots") / f"p1_fit_{args.loss}_{args.x_dist}.png"
        plt.savefig(p1, bbox_inches="tight"); plt.close(); print(f"[Saved] {p1}")

        plt.figure(figsize=(7,4))
        plt.plot(np.arange(1,len(losses)+1), losses)
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs epoch")
        p2 = Path("plots") / "p1_loss_curve.png"
        plt.savefig(p2, bbox_inches="tight"); plt.close(); print(f"[Saved] {p2}")

        plt.figure(figsize=(7,4))
        plt.plot(np.arange(1,len(times)+1), times)
        plt.xlabel("epoch"); plt.ylabel("seconds"); plt.title("Per-epoch time")
        p3 = Path("plots") / "p1_timing.png"
        plt.savefig(p3, bbox_inches="tight"); plt.close(); print(f"[Saved] {p3}")

        print(f"Final params: W≈{W.numpy():.4f}, B≈{B.numpy():.4f}")

# ------------------------ cli ------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TF2 eager linear regression (no Keras)")
    ap.add_argument("--seed_name", type=str, default="Student")
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--loss", type=str, choices=["mse","l1","huber","hybrid"], default="hybrid")
    ap.add_argument("--delta", type=float, default=1.0, help="Huber delta")
    ap.add_argument("--lambda_l1", type=float, default=0.35, help="Hybrid: lambda for L1 term")
    ap.add_argument("--num_examples", type=int, default=800, help="fewer points => nicer scatter")
    ap.add_argument("--x_dist", type=str, choices=["normal","uniform"], default="normal")
    ap.add_argument("--noise_type", type=str, choices=["none","gaussian","laplace","uniform"], default="gaussian")
    ap.add_argument("--noise_std", type=float, default=0.8)
    ap.add_argument("--noise_x", action="store_true")
    ap.add_argument("--noise_y", action="store_true", default=True)
    ap.add_argument("--noise_wb", action="store_true")
    ap.add_argument("--lr_jitter_every", type=int, default=0)
    ap.add_argument("--lr_jitter_std", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--plot_points", type=int, default=600, help="max points to plot in scatter")
    ap.add_argument("--device", type=str, choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--init_w", type=float, default=0.0)
    ap.add_argument("--init_b", type=float, default=0.0)
    args = ap.parse_args()
    train(args)
