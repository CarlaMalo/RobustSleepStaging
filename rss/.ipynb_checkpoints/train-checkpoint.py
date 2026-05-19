from sklearn.ensemble import RandomForestClassifier
import time


def random_forest(X_train, y_train, n_estimators=100, random_state=42, n_jobs=-1, verbose=True):
    """Train a RandomForestClassifier ""
    Args:
        X_train, y_train: training arrays
        n_estimators, random_state, n_jobs: forwarded to sklearn
        verbose: if True, print start/finish messages and duration
    Returns:
        RandomForestClassifier
    """
    t0 = None
    if verbose:
        print(f"[train] starting RandomForest training: n_estimators={n_estimators}, n_samples={len(X_train)}, n_features={X_train.shape[1] if X_train.ndim>1 else '1'}")
        t0 = time.time()

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, verbose=1 if verbose else 0)
    clf.fit(X_train, y_train)

    if verbose and t0 is not None:
        dt = time.time() - t0
        print(f"[train] training complete — elapsed: {dt:.1f}s")

    return clf
