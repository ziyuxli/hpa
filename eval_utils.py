import torch
import torch.nn.functional as F
import os
import csv

def save_rows_to_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    write_header = not os.path.exists(csv_path)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


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
    meta = meta or {}
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
            **meta,
            "eval": "linear_probe",
            "layer": l,
            "train_frac": train_frac,
            "seed": seed,
            "val_acc": acc,
        }
        rows.append(row)
        print(f"[Linear probe] layer {l:02d}: val acc = {acc*100:.2f}%")

    if csv_path is not None and len(rows) > 0:
        save_rows_to_csv(rows, csv_path)
        print("Saved CSV:", csv_path)

    return rows



# knn
@torch.no_grad()
def knn_predict(train_X, train_y, test_X, k=20, device="cuda"):
    train_X = F.normalize(train_X.to(device), dim=1)
    test_X = F.normalize(test_X.to(device), dim=1)
    train_y = train_y.to(device)

    sims = test_X @ train_X.T  # [Nte, Ntr]
    vals, idx = sims.topk(k, dim=1)
    nn_y = train_y[idx]  # [Nte, k]

    num_classes = int(train_y.max().item() + 1)
    # similarity-weighted vote
    votes = torch.zeros(test_X.shape[0], num_classes, device=device)
    votes.scatter_add_(1, nn_y, vals)
    return votes.argmax(dim=1).detach().cpu()

@torch.no_grad()
def layerwise_knn(cls_by_layer, targets, train_frac=0.8, k=20, seed=0, device="cuda",
                  meta=None, csv_path=None):
    meta = meta or {}
    N = targets.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_frac * N)
    idx_tr = perm[:n_train]
    idx_te = perm[n_train:]

    y_tr = targets[idx_tr]
    y_te = targets[idx_te]

    rows = []
    for l, X in enumerate(cls_by_layer):
        pred = knn_predict(X[idx_tr], y_tr, X[idx_te], k=k, device=device)
        acc = (pred == y_te).float().mean().item()

        row = {
            **meta,
            "eval": "knn",
            "layer": l,
            "train_frac": train_frac,
            "k": k,
            "seed": seed,
            "test_acc": acc,
        }
        rows.append(row)
        print(f"[kNN k={k}] layer {l:02d}: test acc = {acc*100:.2f}%")

    if csv_path is not None and len(rows) > 0:
        save_rows_to_csv(rows, csv_path)
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
    meta = meta or {}
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
            **meta,
            "eval": "fewshot_knn",
            "layer": l,
            "shots": shots,
            "k": k,
            "episodes": episodes,
            "seed": seed,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "total_queries": total_q,
        }
        rows.append(row)

    if csv_path is not None and len(rows) > 0:
        save_rows_to_csv(rows, csv_path)
        print("Saved CSV:", csv_path)

    return rows
