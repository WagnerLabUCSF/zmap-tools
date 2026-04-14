from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

DEFAULT_FAISS_NLIST = 4096
DEFAULT_FAISS_NPROBE = 16
DEFAULT_FAISS_MAX_TRAIN_POINTS = 200000
FAISS_TRAIN_POINTS_PER_CENTROID = 40
MAX_FAISS_INDEX_CACHE = 16

_FAISS_INDEX_CACHE: dict[tuple[str, str, int, int], dict[str, Any]] = {}
_FAISS_INDEX_CACHE_ORDER: list[tuple[str, str, int, int]] = []


def _as_float32_c(x: np.ndarray) -> np.ndarray:
    """Cast array to float32 and ensure C-contiguous memory layout (required by FAISS)."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize an array to unit vectors (used for cosine similarity via inner product)."""
    x = _as_float32_c(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _parse_cuda_device(device: str) -> int:
    """Parse a device string like 'cuda:1' into an integer GPU index. Returns 0 on failure."""
    d = str(device).strip().lower()
    if ":" in d:
        try:
            return max(0, int(d.split(":", 1)[1]))
        except Exception:
            return 0
    return 0


def _search_sklearn(
    ref: np.ndarray,
    query: np.ndarray,
    n_neighbors: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    kNN search using scikit-learn's NearestNeighbors.

    Used as the fallback when FAISS is unavailable or fails. Slower than FAISS
    on large references but supports any sklearn-compatible metric.

    Returns
    -------
    neighbor_indices : np.ndarray, shape (n_query, n_neighbors)
    distances : np.ndarray, shape (n_query, n_neighbors)
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(ref)
    distances, neighbor_indices = nn.kneighbors(query, return_distance=True)
    return neighbor_indices, distances


def _search_faiss(
    ref: np.ndarray,
    query: np.ndarray,
    n_neighbors: int,
    metric: str,
    device: str,
    nprobe: int | None = None,
    cache_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    kNN search using FAISS with optional GPU acceleration and index caching.

    Builds an IVF-Flat index by default for fast approximate search on large references.
    Falls back to an exact Flat index if IVF training fails. Cosine similarity is
    implemented as inner product after L2-normalization; euclidean uses L2 distance.

    Parameters
    ----------
    ref : np.ndarray, shape (n_ref, d)
        Reference embedding matrix.
    query : np.ndarray, shape (n_query, d)
        Query embedding matrix.
    n_neighbors : int
        Number of nearest neighbors to return.
    metric : {'euclidean', 'cosine'}
        Distance metric.
    device : str
        Target device — 'cpu', 'auto', 'cuda', or 'cuda:<id>'.
    nprobe : int, optional
        Number of IVF cells to probe during search. Higher values increase recall
        at the cost of speed. Defaults to DEFAULT_FAISS_NPROBE (16).
    cache_key : str, optional
        If provided, the built FAISS index is cached in-memory under this key and
        reused on subsequent calls with matching (key, metric, n_ref, d). Up to
        MAX_FAISS_INDEX_CACHE (16) indices are retained (LRU eviction).

    Returns
    -------
    indices : np.ndarray, shape (n_query, n_neighbors), dtype int64
        Indices into ref for each neighbor. Entries are -1 (set to NaN in distances)
        when FAISS cannot fill all k neighbors (rare with Flat indices).
    distances : np.ndarray, shape (n_query, n_neighbors), dtype float32
        Euclidean distances or cosine-distance proxies (1 - cosine_similarity).
    meta : dict
        Diagnostic info: backend_used, device_used, index_type, nlist, nprobe,
        cache_key_used, cache_hit.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FAISS import failed: {e}") from e

    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"FAISS backend supports only euclidean/cosine, got {metric}.")

    ref_use = _as_float32_c(ref)
    query_use = _as_float32_c(query)
    if metric == "cosine":
        ref_use = _l2_normalize(ref_use)
        query_use = _l2_normalize(query_use)
        metric_type = faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatIP(ref_use.shape[1])
    else:
        metric_type = faiss.METRIC_L2
        quantizer = faiss.IndexFlatL2(ref_use.shape[1])

    n_ref = int(ref_use.shape[0])
    nprobe_target = int(DEFAULT_FAISS_NPROBE) if nprobe is None else int(nprobe)
    if nprobe_target <= 0:
        raise ValueError(f"nprobe must be positive, got {nprobe_target}.")
    cache_hit = False
    cache_key_use = str(cache_key).strip() if cache_key is not None else ""
    cache_token: tuple[str, str, int, int] | None = None
    if cache_key_use:
        cache_token = (cache_key_use, metric, int(ref_use.shape[0]), int(ref_use.shape[1]))
        cached = _FAISS_INDEX_CACHE.get(cache_token, None)
        if cached is not None:
            cpu_index = cached["cpu_index"]
            index_type = str(cached["index_type"])
            nlist_eff = int(cached["nlist"])
            cache_hit = True
            try:
                _FAISS_INDEX_CACHE_ORDER.remove(cache_token)
            except ValueError:
                pass
            _FAISS_INDEX_CACHE_ORDER.append(cache_token)
        else:
            cached = None
    else:
        cached = None

    if not cache_hit:
        
        # Keep nlist adaptive by shard size to avoid over-partition warnings.
        max_nlist_by_train = max(1, n_ref // int(FAISS_TRAIN_POINTS_PER_CENTROID))
        nlist_eff = max(1, min(int(DEFAULT_FAISS_NLIST), n_ref, max_nlist_by_train))

        # Default to IVF (nlist=4096, nprobe=16) for faster search on large references.
        # If IVF training fails,  fall back to exact Flat index.
        index_type = "ivf_flat"
        try:
            cpu_index = faiss.IndexIVFFlat(
                quantizer,
                ref_use.shape[1],
                nlist_eff,
                metric_type,
            )
            train_n = min(
                n_ref,
                max(
                    nlist_eff * int(FAISS_TRAIN_POINTS_PER_CENTROID),
                    min(n_ref, int(DEFAULT_FAISS_MAX_TRAIN_POINTS)),
                ),
            )
            if train_n < n_ref:
                rng = np.random.default_rng(0)
                train_idx = rng.choice(n_ref, size=train_n, replace=False)
                train_x = ref_use[train_idx]
            else:
                train_x = ref_use
            cpu_index.train(_as_float32_c(train_x))
            cpu_index.nprobe = max(1, min(nprobe_target, nlist_eff))
        except Exception as e:
            warnings.warn(f"FAISS IVF build failed ({e}); falling back to Flat index.")
            index_type = "flat"
            nlist_eff = 1
            if metric == "cosine":
                cpu_index = faiss.IndexFlatIP(ref_use.shape[1])
            else:
                cpu_index = faiss.IndexFlatL2(ref_use.shape[1])

        cpu_index.add(ref_use)
        if cache_token is not None:
            _FAISS_INDEX_CACHE[cache_token] = {
                "cpu_index": cpu_index,
                "index_type": str(index_type),
                "nlist": int(nlist_eff),
            }
            _FAISS_INDEX_CACHE_ORDER.append(cache_token)
            while len(_FAISS_INDEX_CACHE_ORDER) > int(MAX_FAISS_INDEX_CACHE):
                old = _FAISS_INDEX_CACHE_ORDER.pop(0)
                _FAISS_INDEX_CACHE.pop(old, None)

    nprobe_eff = 1
    if str(index_type) == "ivf_flat":
        nprobe_eff = max(1, min(nprobe_target, int(nlist_eff)))
        try:
            if hasattr(cpu_index, "nprobe"):
                cpu_index.nprobe = int(nprobe_eff)
        except Exception:
            pass

    backend_device = "cpu"
    index = cpu_index
    req = str(device).strip().lower()
    want_cuda = req.startswith("cuda") or req == "auto"
    if want_cuda:
        has_gpu_api = hasattr(faiss, "StandardGpuResources") and hasattr(
            faiss, "index_cpu_to_gpu"
        )
        n_gpu = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
        if has_gpu_api and n_gpu > 0:
            gpu_id = _parse_cuda_device(req if req.startswith("cuda") else "cuda:0")
            if gpu_id >= n_gpu:
                warnings.warn(
                    f"Requested CUDA device {gpu_id} out of range (n_gpu={n_gpu}); using gpu:0."
                )
                gpu_id = 0
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                if index_type == "ivf_flat":
                    try:
                        if hasattr(index, "nprobe"):
                            index.nprobe = nprobe_eff
                    except Exception:
                        pass
                backend_device = f"cuda:{gpu_id}"
            except Exception as e:
                warnings.warn(f"FAISS GPU init failed ({e}); using CPU FAISS.")
        elif req.startswith("cuda"):
            warnings.warn("CUDA requested but FAISS GPU is unavailable; using CPU FAISS.")

    dist_raw, idx = index.search(query_use, int(n_neighbors))

    if metric == "cosine":
        # IP similarity -> cosine distance proxy.
        dist = np.maximum(1.0 - dist_raw, 0.0).astype(np.float32, copy=False)
    else:
        # L2 returns squared distances.
        dist = np.sqrt(np.maximum(dist_raw, 0.0)).astype(np.float32, copy=False)
    idx = idx.astype(np.int64, copy=False)
    bad = idx < 0
    if np.any(bad):
        dist = dist.copy()
        dist[bad] = np.nan

    meta = {
        "backend_used": "faiss",
        "device_used": backend_device,
        "index_type": index_type,
        "nlist": int(nlist_eff),
        "nprobe": int(nprobe_eff),
        "cache_key_used": (cache_key_use if cache_key_use else None),
        "cache_hit": bool(cache_hit),
    }
    return idx, dist, meta


def knn_search(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "cosine",
    backend: str = "auto",
    device: str = "auto",
    nprobe: int | None = None,
    cache_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Unified kNN search supporting FAISS and scikit-learn backends.

    Tries FAISS first (fast, approximate) and falls back to sklearn (exact) if
    FAISS is unavailable or raises. Both backends return consistent output shapes
    and distance semantics.

    Parameters
    ----------
    ref : np.ndarray, shape (n_ref, d)
        Reference embedding matrix.
    query : np.ndarray, shape (n_query, d)
        Query embedding matrix.
    n_neighbors : int
        Number of nearest neighbors to return per query point.
    metric : {'cosine', 'euclidean'}, default 'cosine'
        Distance metric. Cosine is computed as 1 - cosine_similarity.
    backend : {'auto', 'faiss', 'sklearn'}, default 'auto'
        Search backend. 'auto' tries FAISS first and falls back to sklearn.
        'faiss' also falls back to sklearn with a warning if FAISS fails.
        'sklearn' skips FAISS entirely.
    device : {'auto', 'cpu', 'cuda', 'cuda:<id>'}, default 'auto'
        Device for FAISS index. Ignored by the sklearn backend. 'auto' uses
        GPU if FAISS GPU support is available, otherwise CPU.
    nprobe : int, optional
        Number of IVF partitions to probe during FAISS search. Higher values
        improve recall at the cost of speed. Defaults to DEFAULT_FAISS_NPROBE.
        Ignored by the sklearn backend and FAISS Flat indices.
    cache_key : str, optional
        If provided, the FAISS index built for `ref` is cached in-memory under
        this key and reused on subsequent calls with the same (key, metric,
        n_ref, d). Useful when the same reference is queried repeatedly.

    Returns
    -------
    indices : np.ndarray, shape (n_query, n_neighbors), dtype int64
        Row indices into `ref` for each neighbor, ordered by ascending distance.
    distances : np.ndarray, shape (n_query, n_neighbors), dtype float32
        Corresponding distances. NaN where FAISS returns invalid indices (-1).
    meta : dict
        Diagnostic info including backend_used, device_used, index_type, and
        cache_hit. Useful for verifying which backend and device were actually used.
    """
    metric = str(metric).lower()
    backend_req = str(backend).lower()
    device_req = str(device).lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric}.")
    if backend_req not in {"auto", "faiss", "sklearn"}:
        raise ValueError("backend must be one of {'auto', 'faiss', 'sklearn'}.")

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    if ref.shape[0] < n_neighbors:
        raise ValueError(
            f"n_neighbors={n_neighbors} exceeds reference rows={ref.shape[0]}."
        )

    if backend_req in {"auto", "faiss"}:
        try:
            idx, dist, meta = _search_faiss(
                ref,
                query,
                n_neighbors,
                metric,
                device_req,
                nprobe=nprobe,
                cache_key=cache_key,
            )
            meta["backend_requested"] = backend_req
            meta["device_requested"] = device_req
            meta["nprobe_requested"] = (
                int(DEFAULT_FAISS_NPROBE) if nprobe is None else int(nprobe)
            )
            return idx, dist, meta
        except Exception as e:
            if backend_req == "faiss":
                warnings.warn(f"FAISS backend failed ({e}); falling back to sklearn.")
            elif backend_req == "auto":
                warnings.warn(f"FAISS unavailable ({e}); using sklearn backend.")

    idx, dist = _search_sklearn(ref, query, n_neighbors, metric)
    meta = {
        "backend_requested": backend_req,
        "device_requested": device_req,
        "backend_used": "sklearn",
        "device_used": "cpu",
    }
    return idx, dist, meta
