"""
A tiny multilayer perceptron trained from scratch with numpy.
No TensorFlow / PyTorch / Keras — forward pass, backprop and Adam are all
written out explicitly so you can see exactly what learning is doing.
"""

import functools

import numpy as np


def _quiet_fp(fn):
    """macOS Accelerate BLAS raises spurious FP flags inside matmul even when
    results are finite. Silence those flags locally (only around our math)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)
    return wrapper


class MLP:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.sizes = list(layer_sizes)
        self.W, self.b = [], []
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            scale = np.sqrt(2.0 / fan_in)  # He initialization for ReLU
            self.W.append(rng.standard_normal((fan_in, self.sizes[i + 1])) * scale)
            self.b.append(np.zeros(self.sizes[i + 1]))
            self.mW.append(np.zeros_like(self.W[-1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.mb.append(np.zeros_like(self.b[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))
        self.t = 0

    # ---- forward -------------------------------------------------------
    @_quiet_fp
    def forward(self, x):
        acts, pre = [x], []
        for i in range(len(self.W) - 1):
            z = acts[-1] @ self.W[i] + self.b[i]
            pre.append(z)
            acts.append(np.maximum(z, 0.0))  # ReLU
        z = acts[-1] @ self.W[-1] + self.b[-1]  # logits
        pre.append(z)
        e = np.exp(z - z.max(axis=1, keepdims=True))  # stable softmax
        acts.append(e / e.sum(axis=1, keepdims=True))
        return acts, pre

    # ---- loss + gradients ---------------------------------------------
    @_quiet_fp
    def loss_and_grads(self, x, y):
        acts, pre = self.forward(x)
        B = x.shape[0]
        probs = acts[-1]
        loss = float(-np.log(probs[np.arange(B), y] + 1e-9).mean())

        dz = probs.copy()
        dz[np.arange(B), y] -= 1.0  # d(softmax + cross-entropy)
        dz /= B
        grads_W, grads_b = [], []

        grads_W.append(acts[-2].T @ dz)
        grads_b.append(dz.sum(axis=0))
        da = dz @ self.W[-1].T

        for i in range(len(self.W) - 2, -1, -1):
            dz = da * (pre[i] > 0)  # ReLU derivative
            grads_W.append(acts[i].T @ dz)
            grads_b.append(dz.sum(axis=0))
            if i > 0:
                da = dz @ self.W[i].T

        grads_W.reverse()
        grads_b.reverse()
        return loss, grads_W, grads_b

    # ---- optimization --------------------------------------------------
    def step(self, x, y, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4):
        loss, gW, gb = self.loss_and_grads(x, y)
        self.t += 1
        b1c = 1.0 - beta1 ** self.t
        b2c = 1.0 - beta2 ** self.t
        for i in range(len(self.W)):
            gwi = gW[i] + wd * self.W[i]
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gwi
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gwi * gwi)
            self.W[i] -= lr * (self.mW[i] / b1c) / (np.sqrt(self.vW[i] / b2c) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] * gb[i])
            self.b[i] -= lr * (self.mb[i] / b1c) / (np.sqrt(self.vb[i] / b2c) + eps)
        return loss

    # ---- inference -----------------------------------------------------
    def predict_proba(self, x):
        return self.forward(x)[0][-1]

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def accuracy(self, x, y, batch=4096):
        correct = 0
        for s in range(0, len(y), batch):
            pred = self.predict(x[s:s + batch])
            correct += int((pred == y[s:s + batch]).sum())
        return correct / len(y)

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        data = {"sizes": np.array(self.sizes)}
        for i in range(len(self.W)):
            data[f"W{i}"] = self.W[i]
            data[f"b{i}"] = self.b[i]
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        net = cls([int(x) for x in d["sizes"]])
        for i in range(len(net.W)):
            net.W[i] = d[f"W{i}"]
            net.b[i] = d[f"b{i}"]
        return net
