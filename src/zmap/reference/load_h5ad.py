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


def _uncompressed_path(compressed_path: pathlib.Path) -> pathlib.Path:
    """
    Derive the path for an uncompressed copy of an h5ad file.
    e.g. ZMAP_processed_slim_tpm.h5ad -> ZMAP_processed_slim_tpm.uncompressed.h5ad
    """
    return compressed_path.with_suffix("").with_suffix(".uncompressed.h5ad")


def _resolve_expected_path(
    kind: str | None,
    url: str | None,
    dest_dir: str | os.PathLike | None,
    filename: str | None,
) -> pathlib.Path | None:
    """
    Derive the expected compressed file path using the same logic as
    download_zmap_h5ad, without performing any I/O.
    Returns None if the path cannot be determined.
    """
    meta = H5AD_SOURCES.get(kind or "", {}) if url is None else {}
    final_url = url or meta.get("url")
    if final_url is None:
        return None

    dest_dir_path = _default_h5ad_dir() if dest_dir is None else pathlib.Path(dest_dir)

    if filename is not None:
        fname = filename
    elif "filename" in meta:
        fname = meta["filename"]
    else:
        fname = pathlib.Path(urllib.request.urlparse(final_url).path).name or "zmap.h5ad"

    return dest_dir_path / fname


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
    Download an H5AD file.

    Parameters
    ----------
    kind
        One of H5AD_SOURCES keys, used to look up default URL and filename.
        Ignored if `url` is provided.
    url
        Optional explicit URL. If given, overrides the registry URL.
    dest_dir
        Directory to store the file. Default: /content/drive/MyDrive/zmap/h5ad.
    filename
        Optional filename override.
    write_to_disk
        If False, downloads to a temp file (not kept after load).
    force_download
        If True, re-download even if dest file exists.
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
    Add a 'tpm_log' layer if missing but 'raw_nolog' exists.
    Performs standard TPM normalization and log1p transform.
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
    cache_uncompressed: bool = True,
) -> ad.AnnData:
    """
    High-level loader for ZMAP H5ADs.

    Load priority (fastest to slowest):
        1. In-memory _H5AD_CACHE       (same session, instantaneous)
        2. Uncompressed .h5ad on Drive (warm cache, ~60s)
        3. Compressed .h5ad on Drive   (first load or forced, ~2min)
        4. Fresh download              (no local file at all)

    On first load, an uncompressed copy is written to Drive so that
    subsequent sessions hit path 2 instead of path 3.

    Parameters
    ----------
    cache_uncompressed
        Write an uncompressed copy after first load for faster future reads.
        Only active when Google Drive is mounted, write_to_disk=True, backed=False.
    """
    if backed and not write_to_disk:
        print("[ZMAP] backed=True requires write_to_disk=True; overriding.")
        write_to_disk = True

    # uncompressed cache only makes sense on Drive
    _do_uncompressed = (
        cache_uncompressed
        and write_to_disk
        and not backed
        and pathlib.Path("/content/drive/MyDrive").exists()
    )

    # ------------------------------------------------------------------
    # 1) In-memory cache (fastest — same session)
    # ------------------------------------------------------------------
    cache_key = (kind or "custom", url, bool(backed))
    if use_cache and cache_key in _H5AD_CACHE:
        return _H5AD_CACHE[cache_key]

    # ------------------------------------------------------------------
    # 2) Check for uncompressed copy FIRST — before any other Drive I/O
    # ------------------------------------------------------------------
    if _do_uncompressed and not force_download:
        expected_path = _resolve_expected_path(kind, url, dest_dir, filename)
        if expected_path is not None:
            uncompressed = _uncompressed_path(expected_path)
            if uncompressed.exists():
                print(f"[ZMAP] Loading uncompressed cache from {uncompressed}")
                adata = ad.read_h5ad(uncompressed)
                if use_cache:
                    _H5AD_CACHE[cache_key] = adata
                return adata

    # ------------------------------------------------------------------
    # 3) Download compressed file if needed
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
    # 4) Load compressed file
    # ------------------------------------------------------------------
    if backed:
        backed_mode = "r" if backed is True else backed
        print(f"[ZMAP] Loading (backed={backed_mode!r}) from {path}")
        adata = sc.read_h5ad(path, backed=backed_mode)
    else:
        print(f"[ZMAP] Loading into memory from {path}")
        adata = ad.read_h5ad(path)

    # ------------------------------------------------------------------
    # 5) Optional preprocessing
    # ------------------------------------------------------------------
    if attempt_preprocess_tpmlog and not backed:
        preprocess_tpmlog(adata)

    # ------------------------------------------------------------------
    # 6) Write uncompressed copy for fast future loads
    # ------------------------------------------------------------------
    if _do_uncompressed:
        uncompressed = _uncompressed_path(path)
        print(f"[ZMAP] Writing uncompressed cache → {uncompressed}")
        print("[ZMAP] (this only happens once; future loads will be much faster)")
        adata.write_h5ad(uncompressed, compression=None)

    # ------------------------------------------------------------------
    # 7) Clean up if not keeping persistent copy
    # ------------------------------------------------------------------
    if not write_to_disk:
        try:
            os.unlink(path)
        except OSError:
            pass

    if use_cache and not backed:
        _H5AD_CACHE[cache_key] = adata

    return adata
