import os
import pathlib
import urllib.request
import tempfile

import anndata as ad
import scanpy as sc
from tqdm import tqdm

# --------------------------------------------------------------------
# Registry of known H5ADs (fill in your real URLs & filenames)
# --------------------------------------------------------------------

H5AD_SOURCES = {
    # Raw counts
    "raw": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_250402_raw.h5ad",
        "filename": "ZMAP_250402_raw.h5ad",
    },
    # Fully processed (full dataset + intermediate files)
    "processed": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251209_processed.h5ad",
        "filename": "ZMAP_251209_processed.h5ad",
    },
    # Fully processed but raw counts only
    "processed_slim": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251209_processed_slim.h5ad",
        "filename": "ZMAP_251209_processed_slim.h5ad",
    },
    # Fully processed but tpm counts only (best for plotting)
    "processed_slim_tpm": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251209_processed_slim_tpm.h5ad",
        "filename": "ZMAP_251209_processed_slim_tpm.h5ad",
    },
    # Processed slim / symphony reference
    "symphony": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_260103_symphony.h5ad",
        "filename": "ZMAP_260103_symphony.h5ad",
    },
}

# In-memory cache per (kind, url, backed)
_H5AD_CACHE: dict[tuple[str, str | None, bool], ad.AnnData] = {}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _default_h5ad_dir() -> pathlib.Path:
    """
    Default directory to store / cache H5ADs.

    Uses Google Drive when available (/content/drive/MyDrive/zmap/h5ad),
    so files persist across Colab sessions. Falls back to <cwd>/zmap/h5ad
    if Drive is not mounted.
    """
    drive_path = pathlib.Path("/content/drive/MyDrive/zmap/h5ad")
    if drive_path.parent.exists():
        drive_path.mkdir(parents=True, exist_ok=True)
        return drive_path

    print(
        "[ZMAP] Google Drive not detected at /content/drive/MyDrive — "
        "using local cache at <cwd>/zmap/h5ad. "
        "Mount Drive and re-run to enable persistent caching."
    )
    fallback = pathlib.Path.cwd() / "zmap" / "h5ad"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _open_url(url: str):
    """Open a URL with a browser-like User-Agent so Cloudflare is happy."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ZMAP/0.1; +https://example.org)"
        },
    )
    return urllib.request.urlopen(req)


def _stream_download(
    url: str,
    dest_path: pathlib.Path,
    *,
    chunk_size: int = 1 << 20,  # 1 MB
    show_progress: bool = True,
):
    """
    Stream a file from URL to dest_path with an optional tqdm progress bar.
    """
    print(f"[ZMAP] Downloading {url} → {dest_path}")
    resp = _open_url(url)
    total = None
    try:
        total = int(resp.headers.get("Content-Length", "0")) or None
    except Exception:
        total = None

    if show_progress:
        pbar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
        )
    else:
        pbar = None

    with dest_path.open("wb") as out:
        while True:
            block = resp.read(chunk_size)
            if not block:
                break
            out.write(block)
            if pbar is not None:
                pbar.update(len(block))

    if pbar is not None:
        pbar.close()
    resp.close()


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def download_zmap_h5ad(
    *,
    kind: str | None = "processed_slim_tpm",
    url: str | None = None,
    dest_dir: str | os.PathLike | None = None,
    filename: str | None = None,
    write_to_disk: bool = True,
    force_download: bool = False,
    chunk_size: int = 1 << 20,
    show_progress: bool = True,
) -> pathlib.Path:
    """
    Download a ZMAP H5AD file from the CDN, with local caching.

    Downloads the file to a persistent cache directory (Google Drive when
    available, otherwise a local directory). Subsequent calls with the same
    ``kind`` skip the download if the file already exists on disk.

    Most users should prefer :func:`load_zmap_h5ad`, which calls this function
    internally and also handles loading and preprocessing.

    Parameters
    ----------
    kind : str or None, default ``"processed_slim_tpm"``
        Preset dataset key. One of the keys in :data:`H5AD_SOURCES`
        (``"raw"``, ``"processed"``, ``"processed_slim"``,
        ``"processed_slim_tpm"``, ``"symphony"``). Ignored when ``url``
        is provided.
    url : str or None, default ``None``
        Explicit download URL. Overrides the registry URL looked up via
        ``kind``.
    dest_dir : path-like or None, default ``None``
        Directory to store the downloaded file. Defaults to
        ``/content/drive/MyDrive/zmap/h5ad`` when Google Drive is mounted,
        or ``<cwd>/zmap/h5ad`` otherwise.
    filename : str or None, default ``None``
        Override the filename used when saving to disk. Inferred from the
        registry or URL when not provided.
    write_to_disk : bool, default ``True``
        If ``False``, downloads to a temporary file that is not kept after
        loading.
    force_download : bool, default ``False``
        Re-download the file even if it already exists on disk.
    chunk_size : int, default ``1 << 20`` (1 MB)
        Download chunk size in bytes.
    show_progress : bool, default ``True``
        Display a ``tqdm`` progress bar during download.

    Returns
    -------
    pathlib.Path
        Path to the downloaded (or cached) H5AD file on disk.

    Raises
    ------
    ValueError
        If no URL can be resolved from ``kind`` or ``url``.

    Examples
    --------
    >>> path = zmap.ref.download_zmap_h5ad()
    >>> path = zmap.ref.download_zmap_h5ad(kind="symphony")
    >>> path = zmap.ref.download_zmap_h5ad(url="https://.../my.h5ad")
    """
    meta = H5AD_SOURCES.get(kind or "", {}) if url is None else {}
    final_url = url or meta.get("url")
    if final_url is None:
        raise ValueError(f"No URL provided and no registry entry for kind={kind!r}")

    if dest_dir is None:
        dest_dir_path = _default_h5ad_dir()
    else:
        dest_dir_path = pathlib.Path(dest_dir)
        dest_dir_path.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        fname = filename
    elif "filename" in meta:
        fname = meta["filename"]
    else:
        fname = pathlib.Path(urllib.request.urlparse(final_url).path).name or "zmap.h5ad"

    dest_path = dest_dir_path / fname

    if not write_to_disk:
        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        dest_path = pathlib.Path(tmp.name)
        tmp.close()
        _stream_download(final_url, dest_path, chunk_size=chunk_size, show_progress=show_progress)
        return dest_path

    if dest_path.exists() and not force_download:
        print(f"[ZMAP] Using cached file: {dest_path}")
        return dest_path

    _stream_download(final_url, dest_path, chunk_size=chunk_size, show_progress=show_progress)
    return dest_path


def preprocess_tpmlog(adata: ad.AnnData):
    """
    Add a ``tpm_log`` layer by normalizing raw counts to TPM + log1p.

    Checks whether ``adata.layers["raw_nolog"]`` exists and
    ``adata.layers["tpm_log"]`` does not. When both conditions are met,
    performs library-size normalization to counts per million followed by
    ``log1p``, and stores the result as ``adata.layers["tpm_log"]``.

    This is a convenience function called automatically by
    :func:`load_zmap_h5ad` when ``attempt_preprocess_tpmlog=True``.

    Parameters
    ----------
    adata : anndata.AnnData
        The dataset to preprocess. Modified in-place.

    Notes
    -----
    After this call, ``adata.X`` is cleared (set to ``None``) so that
    downstream code explicitly selects a layer rather than relying on
    a stale ``.X`` matrix.
    """
    if "raw_nolog" in adata.layers and "tpm_log" not in adata.layers:
        print("[ZMAP] Computing 'tpm_log' from 'raw_nolog' (normalize + log1p)")
        adata.X = adata.layers["raw_nolog"]
        sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
        sc.pp.log1p(adata)
        adata.layers["tpm_log"] = adata.X
        del adata.X


# --------------------------------------------------------------------
# High Level Wrapper (download if needed, load, preprocess)
# --------------------------------------------------------------------

def load_zmap_h5ad(
    *,
    kind: str | None = "processed_slim_tpm",
    url: str | None = None,
    dest_dir: str | os.PathLike | None = None,
    filename: str | None = None,
    write_to_disk: bool = True,
    use_cache: bool = True,
    force_download: bool = False,
    backed: bool | str = False,
    chunk_size: int = 1 << 20,
    show_progress: bool = True,
    attempt_preprocess_tpmlog: bool = True,
) -> ad.AnnData:
    """
    Load a ZMAP reference dataset into memory, downloading it if necessary.

    This is the primary entry point for accessing ZMAP reference data. On first
    call the file is downloaded and cached to Google Drive (if mounted) or a
    local directory. Subsequent calls in the same session are served from an
    in-memory cache and return instantly.

    Load priority (fastest to slowest):

    1. In-memory session cache — instantaneous, no I/O.
    2. File already on disk (Drive or local) — fast, no download.
    3. Fresh download from the ZMAP CDN.

    Parameters
    ----------
    kind : str or None, default ``"processed_slim_tpm"``
        Preset dataset to load. One of:

        - ``"processed_slim_tpm"`` — fully processed, TPM counts only.
          Best default for visualization and label transfer.
        - ``"processed_slim"``     — fully processed, raw counts only.
        - ``"processed"``          — fully processed, includes intermediate layers.
        - ``"raw"``                — raw counts, unprocessed.
        - ``"symphony"``           — Symphony reference used for query embedding.
          Required for ``annotate_with_zmap``.

        Ignored when ``url`` is provided.
    url : str or None, default ``None``
        Explicit download URL. Overrides ``kind``. Use this to load a custom
        or external H5AD not in the ZMAP registry.
    dest_dir : path-like or None, default ``None``
        Directory where the H5AD file is saved. Defaults to
        ``/content/drive/MyDrive/zmap/h5ad`` when Google Drive is mounted,
        or ``<cwd>/zmap/h5ad`` otherwise.
    filename : str or None, default ``None``
        Override the filename used when saving to disk. Inferred from the
        registry or URL when not provided.
    write_to_disk : bool, default ``True``
        If ``False``, downloads to a temporary file that is deleted after
        loading. Useful for one-off loads when disk space is constrained.
        Incompatible with ``backed=True``.
    use_cache : bool, default ``True``
        If ``True``, return the cached in-memory object on repeat calls.
        Set to ``False`` to force a fresh load from disk (e.g. after
        modifying the file externally).
    force_download : bool, default ``False``
        Re-download the file even if it already exists on disk.
    backed : bool or str, default ``False``
        Open the H5AD in backed (memory-mapped) mode. Pass ``True`` for
        read-only (``"r"``), or a mode string (e.g. ``"r+"``) for
        read-write. Backed mode avoids loading the full matrix into RAM
        but is slower for random access. Requires ``write_to_disk=True``.
    chunk_size : int, default ``1 << 20`` (1 MB)
        Download chunk size in bytes.
    show_progress : bool, default ``True``
        Display a ``tqdm`` progress bar while downloading.
    attempt_preprocess_tpmlog : bool, default ``True``
        If the loaded object has a ``raw_nolog`` layer but no ``tpm_log``
        layer, compute ``tpm_log`` via TPM normalization + log1p and add
        it as a layer. Has no effect if ``tpm_log`` is already present or
        if ``backed=True``.

    Returns
    -------
    anndata.AnnData
        The loaded reference dataset.

    Examples
    --------
    >>> adata_ref = zmap.ref.load_zmap_h5ad()                          # default: processed_slim_tpm
    >>> adata_ref = zmap.ref.load_zmap_h5ad(kind="symphony")           # for annotate_with_zmap
    >>> adata_ref = zmap.ref.load_zmap_h5ad(url="https://.../my.h5ad", filename="my.h5ad")
    """
    if backed and not write_to_disk:
        print("[ZMAP] backed=True requires write_to_disk=True; overriding.")
        write_to_disk = True

    # ------------------------------------------------------------------
    # 1) In-memory cache (fastest — same session)
    # ------------------------------------------------------------------
    cache_key = (kind or "custom", url, bool(backed))
    if use_cache and cache_key in _H5AD_CACHE:
        return _H5AD_CACHE[cache_key]

    # ------------------------------------------------------------------
    # 2) Download file if needed (no-op if already on Drive)
    # ------------------------------------------------------------------
    path = download_zmap_h5ad(
        kind=kind,
        url=url,
        dest_dir=dest_dir,
        filename=filename,
        write_to_disk=write_to_disk,
        force_download=force_download,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )

    # ------------------------------------------------------------------
    # 3) Load
    # ------------------------------------------------------------------
    if backed:
        backed_mode = "r" if backed is True else backed
        print(f"[ZMAP] Loading (backed={backed_mode!r}) from {path}")
        adata = sc.read_h5ad(path, backed=backed_mode)
    else:
        print(f"[ZMAP] Loading into memory from {path}")
        adata = ad.read_h5ad(path)

    # ------------------------------------------------------------------
    # 4) Optional preprocessing
    # ------------------------------------------------------------------
    if attempt_preprocess_tpmlog and not backed:
        preprocess_tpmlog(adata)

    # ------------------------------------------------------------------
    # 5) Clean up if not keeping persistent copy
    # ------------------------------------------------------------------
    if not write_to_disk:
        try:
            os.unlink(path)
        except OSError:
            pass

    if use_cache and not backed:
        _H5AD_CACHE[cache_key] = adata

    return adata
