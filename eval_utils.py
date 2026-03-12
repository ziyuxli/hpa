import os
import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score



def save_rows_to_csv(rows, csv_path, mode="a"):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    write_header = mode == "w" or not os.path.exists(csv_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [{"run_id": run_id, **row} for row in rows]

    fieldnames = list(rows[0].keys())
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# Linear probe
def train_linear_probe(X_train, y_train, X_val, y_val, epochs=200, lr=1e-2, wd=0.0, device="cuda"):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    X_train = F.normalize(X_train, dim=1)
    X_val = F.normalize(X_val, dim=1)

    num_classes = int(y_train.max().item() + 1)
    D = X_train.shape[1]
    clf = torch.nn.Linear(D, num_classes).to(device)

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)

    best = 0.0
    best_state = None

    for _ in range(epochs):
        clf.train()
        logits = clf(X_train)
        loss = F.cross_entropy(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            pred = clf(X_val).argmax(dim=1)
            acc = (pred == y_val).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu() for k, v in clf.state_dict().items()}

    return best, best_state

def layerwise_linear_probe(cls_by_layer, targets, train_frac=0.8, seed=0, device="cuda", meta=None, csv_path=None):
    N = targets.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_frac * N)
    idx_tr = perm[:n_train]
    idx_va = perm[n_train:]

    y_tr = targets[idx_tr]
    y_va = targets[idx_va]

    rows = []
    for l, X in enumerate(cls_by_layer):
        X_tr = X[idx_tr]
        X_va = X[idx_va]
        acc, _ = train_linear_probe(X_tr, y_tr, X_va, y_va, device=device)

        row = {
            "eval": "linear_probe",
            "seed": seed,
            "layer": l,
            "train_frac": train_frac,
            "val_acc": acc,
        }
        rows.append(row)
        print(f"[Linear probe] layer {l:02d}: val acc = {acc*100:.2f}%")

    if csv_path is not None:
        rows_to_write = []
        for row in rows:
            out_row = {f"meta_{k}": v for k, v in (meta or {}).items()}
            out_row.update(row)
            rows_to_write.append(out_row)
        save_rows_to_csv(rows_to_write, csv_path)
        print("Saved CSV:", csv_path)

    return rows


@torch.no_grad()
def fewshot_knn_episode(X, y, shots=1, k=1, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)

    classes = torch.unique(y)
    support_idx = []
    query_idx = []

    for c in classes.tolist():
        idx = torch.nonzero(y == c, as_tuple=False).squeeze(1)
        if idx.numel() <= shots:
            # not enough examples to form query set
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        support_idx.append(perm[:shots])
        query_idx.append(perm[shots:])

    if len(support_idx) == 0:
        return float("nan"), 0

    support_idx = torch.cat(support_idx)
    query_idx = torch.cat(query_idx)

    Xs = X[support_idx].to(device)
    ys = y[support_idx].to(device)
    Xq = X[query_idx].to(device)
    yq = y[query_idx].to(device)

    Xs = F.normalize(Xs, dim=1)
    Xq = F.normalize(Xq, dim=1)

    sims = Xq @ Xs.T  # [Nq, Ns]
    vals, nn = sims.topk(k=min(k, Xs.shape[0]), dim=1)  # [Nq, k]
    nn_y = ys[nn]  # [Nq, k]

    num_classes = int(y.max().item() + 1)
    votes = torch.zeros(Xq.shape[0], num_classes, device=device)
    votes.scatter_add_(1, nn_y, vals)  # similarity-weighted vote
    pred = votes.argmax(dim=1)

    acc = (pred == yq).float().mean().item()
    return acc, int(yq.numel())


@torch.no_grad()
def layerwise_fewshot_knn(cls_by_layer, targets, shots=1, k=1, episodes=50, seed=0, device="cpu",
                          meta=None, csv_path=None):
    targets = targets.cpu()
    rows = []

    for l, X in enumerate(cls_by_layer):
        X = X.cpu()
        accs = []
        total_q = 0

        for e in range(episodes):
            acc, n_q = fewshot_knn_episode(
                X, targets, shots=shots, k=k, seed=seed + 1000*l + e, device=device
            )
            if n_q > 0 and acc == acc:
                accs.append(acc)
                total_q += n_q

        if len(accs) == 0:
            mean_acc, std_acc = float("nan"), float("nan")
        else:
            mean_acc = float(torch.tensor(accs).mean().item())
            std_acc = float(torch.tensor(accs).std(unbiased=False).item())

        print(f"[Few-shot kNN] layer {l:02d}: {shots}-shot, k={k}, "
              f"acc={mean_acc*100:.2f}% ± {std_acc*100:.2f}%  (episodes={len(accs)})")

        row = {
            "eval": "fewshot_knn",
            "seed": seed,
            "layer": l,
            "shots": shots,
            "k": k,
            "episodes": episodes,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "total_queries": total_q,
        }
        rows.append(row)

    if csv_path is not None:
        rows_to_write = []
        for row in rows:
            out_row = {f"meta_{k}": v for k, v in (meta or {}).items()}
            out_row.update(row)
            rows_to_write.append(out_row)
        save_rows_to_csv(rows_to_write, csv_path)
        print("Saved CSV:", csv_path)

    return rows






















def load_csv(human49_csv):
    object_to_vec = {}
    dim_names = None

    with open(human49_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        dim_names = header[1:]

        for row in reader:
            if not row:
                continue
            obj = row[0].strip()
            vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
            object_to_vec[obj] = vec

    return object_to_vec, dim_names


def layerwise_human49_r2(
    cls_by_layer,
    targets,
    class_id_to_name,
    human49_csv,
    test_frac=0.3,
    seed=42,
    alpha=1.0,
    meta=None,
    csv_path=None,
):
    """
    Evaluate each layer embedding by predicting Human49 vector with object-level split.

    Args:
        cls_by_layer: list[Tensor], each [N, D]
        targets: Tensor [N]
        class_id_to_name: dict[int, str]
        human49_csv: path to human49.csv
        test_frac: float
        seed: int
        alpha: ridge regularization strength
        meta: dict
        csv_path: save per-layer results

    Returns:
        results: list[dict]
    """
    object_to_vec, dim_names = load_csv(human49_csv)

    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    class_names = [class_id_to_name[int(t)] for t in targets_np]

    Y = np.stack([object_to_vec[name] for name in class_names], axis=0)  # [N, 49]

    # split by object identity, not by image
    unique_objects = np.array(sorted(set(class_names)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_objects)

    n_test_obj = max(1, int(round(len(unique_objects) * test_frac)))
    test_objects = set(unique_objects[:n_test_obj].tolist())
    train_objects = set(unique_objects[n_test_obj:].tolist())

    train_mask = np.array([name in train_objects for name in class_names], dtype=bool)
    test_mask = np.array([name in test_objects for name in class_names], dtype=bool)

    results = []

    for layer_idx, feats in enumerate(cls_by_layer):
        X = feats.cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats)

        X_train = X[train_mask]
        Y_train = Y[train_mask]
        X_test = X[test_mask]
        Y_test = Y[test_mask]

        # Standardize X, then ridge regression for multi-output prediction
        reg = make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)

        # overall multioutput r2
        r2_uniform = r2_score(Y_test, Y_pred, multioutput="uniform_average")
        r2_variance_weighted = r2_score(Y_test, Y_pred, multioutput="variance_weighted")

        # per-dimension r2
        per_dim_r2 = r2_score(Y_test, Y_pred, multioutput="raw_values")
        per_dim_r2 = np.asarray(per_dim_r2, dtype=np.float64)

        row = {
            "eval": "human49_r2",
            "seed": int(seed),
            "layer": layer_idx,
            "r2_uniform_avg": float(r2_uniform),
            "r2_variance_weighted": float(r2_variance_weighted),
            "r2_dim_mean": float(np.mean(per_dim_r2)),
            "r2_dim_median": float(np.median(per_dim_r2)),
            "n_train_images": int(train_mask.sum()),
            "n_test_images": int(test_mask.sum()),
            "n_train_objects": int(len(train_objects)),
            "n_test_objects": int(len(test_objects)),
            "ridge_alpha": float(alpha),
        }

        results.append(row)

        print(
            f"[Human49 R2] layer={layer_idx:02d} | "
            f"R2(uniform)={r2_uniform:.4f} | "
            f"R2(var_weighted)={r2_variance_weighted:.4f}"
        )

    if csv_path is not None:
        rows_to_write = []
        for row in results:
            out_row = {f"meta_{k}": v for k, v in (meta or {}).items()}
            out_row.update(row)
            rows_to_write.append(out_row)
        save_rows_to_csv(rows_to_write, csv_path)
        print(f"Saved CSV: {csv_path}")

    return results










def layerwise_supercategory_prediction(
    cls_by_layer,
    targets,
    class_id_to_name,
    supercategory_csv,
    test_frac=0.3,
    seed=42,
    C=1.0,
    max_iter=1000,
    meta=None,
    csv_path=None,
):
    object_to_attr, attr_names = load_csv(supercategory_csv)

    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    class_names = [class_id_to_name[int(t)] for t in targets_np]

    Y = np.stack([object_to_attr[name] for name in class_names], axis=0)  # [N, K]
    Y = Y.astype(np.float32)

    unique_objects = np.array(sorted(set(class_names)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_objects)

    n_test_obj = max(1, int(round(len(unique_objects) * test_frac)))
    test_objects = set(unique_objects[:n_test_obj].tolist())
    train_objects = set(unique_objects[n_test_obj:].tolist())

    train_mask = np.array([name in train_objects for name in class_names], dtype=bool)
    test_mask = np.array([name in test_objects for name in class_names], dtype=bool)

    Y_train = Y[train_mask]
    Y_test = Y[test_mask]

    # keep only supercategorys that are not constant in training
    train_pos = Y_train.sum(axis=0)
    train_neg = Y_train.shape[0] - train_pos
    valid_train_cols = (train_pos > 0) & (train_neg > 0)

    if valid_train_cols.sum() == 0:
        raise ValueError("All supercategorys are constant in training split. Cannot train classifiers.")

    Y_train_valid = Y_train[:, valid_train_cols]
    Y_test_valid = Y_test[:, valid_train_cols]
    valid_attr_names = [attr_names[i] for i in range(len(attr_names)) if valid_train_cols[i]]

    results = []

    for layer_idx, feats in enumerate(cls_by_layer):
        X = feats.cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats)

        X_train = X[train_mask]
        X_test = X[test_mask]

        clf = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=max_iter,
                    random_state=seed,
                )
            )
        )
        clf.fit(X_train, Y_train_valid)

        # probability for positive class
        Y_score = clf.predict_proba(X_test)  # [N_test, K_valid]

        # AP can work even if some test labels are all-0 or all-1, but macro over such labels is not ideal.
        # ROC-AUC requires both classes in test, so we filter again for test-valid labels.
        test_pos = Y_test_valid.sum(axis=0)
        test_neg = Y_test_valid.shape[0] - test_pos
        valid_test_cols = (test_pos > 0) & (test_neg > 0)

        Y_pred = (Y_score > 0.5).astype(int)

        # accuracy
        acc = accuracy_score(
            Y_test_valid.flatten(),
            Y_pred.flatten()
        )

        # AUC
        if valid_test_cols.sum() > 0:
            auc = roc_auc_score(
                Y_test_valid[:, valid_test_cols],
                Y_score[:, valid_test_cols],
                average="macro"
            )
        else:
            auc = np.nan

        row = {
            "eval": "supercategory",
            "seed": int(seed),
            "layer": layer_idx,
            "acc": float(acc),
            "auc": float(auc),
            "n_train_images": int(train_mask.sum()),
            "n_test_images": int(test_mask.sum()),
            "n_train_objects": int(len(train_objects)),
            "n_test_objects": int(len(test_objects)),
            "n_total_supercategorys": int(Y.shape[1]),
            "n_train_valid_supercategorys": int(valid_train_cols.sum()),
            "logreg_C": float(C),
            "max_iter": int(max_iter),
        }
        results.append(row)

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"

        print(
            f"[Attribute Eval] layer={layer_idx:02d} | "
            f"ACC={acc:.4f} | "
            f"AUC={auc_str}"
        )

    if csv_path is not None:
        rows_to_write = []
        for row in results:
            out_row = {f"meta_{k}": v for k, v in (meta or {}).items()}
            out_row.update(row)
            rows_to_write.append(out_row)
        save_rows_to_csv(rows_to_write, csv_path)
        print(f"Saved CSV: {csv_path}")

    return results