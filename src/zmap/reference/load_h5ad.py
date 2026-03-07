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
    default_drive_path = pathlib.Path("/content/drive/MyDrive/zmap/h5ad")

    # Use Drive if it's mounted and accessible
    if default_drive_path.parent.exists():
        default_drive_path.mkdir(parents=True, exist_ok=True)
        return default_drive_path

    # Fallback for non-Colab or unmounted Drive
    print(
        "[ZMAP] Google Drive not detected at /content/drive/MyDrive — "
        "using local cache at <cwd>/zmap/h5ad. "
        "Mount Drive and re-run to enable persistent caching."
    )
    fallback_path = pathlib.Path.cwd() / "zmap" / "h5ad"
    fallback_path.mkdir(parents=True, exist_ok=True)
    return fallback_path


def _uncompressed_path(compressed_path: pathlib.Path) -> pathlib.Path:
    """
    Derive the path for an uncompressed copy of an h5ad file.
    e.g. ZMAP_processed_slim.h5ad -> ZMAP_processed_slim.uncompressed.h5ad
    """
    return compressed_path.with_suffix("").with_suffix(".uncompressed.h5ad")


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
    # Try to get total size from headers
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
    kind: str | None = "symphony",
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
        One of H5AD_SOURCES keys ('raw', 'processed', 'processed_slim'), used
        to look up default URL and filename.
        Ignored if `url` is provided.
    url
        Optional explicit URL. If given, overrides the registry URL.
    dest_dir
        Directory to store the file. Default: <cwd>/zmap/h5ads.
    filename
        Optional filename. If None, uses registry filename (if kind known),
        otherwise tries to infer from the URL.
    write_to_disk
        If False, still uses a temporary file (for read_h5ad), but does not
        keep a persistent copy.
    force_download
        If True, re-download even if dest file exists (only relevant when
        write_to_disk=True).
    """

    # Determine metadata
    meta = H5AD_SOURCES.get(kind or "", {}) if url is None else {}
    final_url = url or meta.get("url")
    if final_url is None:
        raise ValueError("No URL provided and no registry entry for kind={kind!r}")

    # Determine dest dir / filename
    if dest_dir is None:
        dest_dir_path = _default_h5ad_dir()
    else:
        dest_dir_path = pathlib.Path(dest_dir)
        dest_dir_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if "filename" in meta:
            fname = meta["filename"]
        else:
            # Try to guess filename from URL
            fname = pathlib.Path(urllib.request.urlparse(final_url).path).name or "zmap.h5ad"
    else:
        fname = filename

    dest_path = dest_dir_path / fname

    # If we don't want a persistent copy, use a temp file
    if not write_to_disk:
        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        dest_path = pathlib.Path(tmp.name)
        tmp.close()
        _stream_download(
            final_url,
            dest_path,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
        return dest_path

    # Persistent copy
    if dest_path.exists() and not force_download:
        print(f"[ZMAP] Using cached file: {dest_path}")
        return dest_path

    _stream_download(
        final_url,
        dest_path,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
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

    On first load, downloads the (compressed) h5ad, runs any preprocessing,
    then writes an uncompressed copy alongside it. Subsequent loads read the
    uncompressed copy directly, skipping download and decompression entirely.

    Parameters
    ----------
    cache_uncompressed
        If True (default), write an uncompressed copy of the h5ad to disk on
        first load and prefer it on subsequent loads. Faster to read but uses
        more disk space. Has no effect when write_to_disk=False or backed=True.

    Examples
    --------
    adata = load_zmap_h5ad(kind="symphony")
    adata_raw = load_zmap_h5ad(kind="raw")
    adata_custom = load_zmap_h5ad(
        url="https://.../my_custom.h5ad", filename="my_custom.h5ad"
    )
    """
    # backed mode requires a real file on disk
    if backed and not write_to_disk:
        print("[ZMAP] backed=True requires write_to_disk=True; overriding.")
        write_to_disk = True

    # uncompressed cache is only meaningful for persistent, non-backed loads
    _do_uncompressed = cache_uncompressed and write_to_disk and not backed and pathlib.Path("/content/drive/MyDrive").exists()

    # ------------------------------------------------------------------
    # 1) In-memory cache check (fastest path — same session)
    # ------------------------------------------------------------------
    cache_key = (kind or "custom", url, bool(backed))
    if use_cache and cache_key in _H5AD_CACHE:
        return _H5AD_CACHE[cache_key]

    # ------------------------------------------------------------------
    # 2) Resolve the compressed file path (without downloading yet)
    #    so we can derive the uncompressed path and check it first.
    # ------------------------------------------------------------------
    if _do_uncompressed:
        meta = H5AD_SOURCES.get(kind or "", {}) if url is None else {}
        final_url = url or meta.get("url")

        if dest_dir is None:
            dest_dir_path = _default_h5ad_dir()
        else:
            dest_dir_path = pathlib.Path(dest_dir)

        if filename is not None:
            fname = filename
        elif "filename" in meta:
            fname = meta["filename"]
        elif final_url is not None:
            fname = pathlib.Path(urllib.request.urlparse(final_url).path).name or "zmap.h5ad"
        else:
            fname = "zmap.h5ad"

        compressed_path = dest_dir_path / fname
        uncompressed = _uncompressed_path(compressed_path)

        # Warm path: uncompressed file already exists and download not forced
        if uncompressed.exists() and not force_download:
            print(f"[ZMAP] Loading uncompressed cache from {uncompressed}")
            adata = ad.read_h5ad(uncompressed)
            if use_cache:
                _H5AD_CACHE[cache_key] = adata
            return adata

    # ------------------------------------------------------------------
    # 3) Download or reuse compressed file
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
    # 4) Load
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
        print(f"[ZMAP] Writing uncompressed cache → {uncompressed}")
        print("[ZMAP] (this only happens once; future loads will be much faster)")
        adata.write_h5ad(uncompressed, compression=None)

    # ------------------------------------------------------------------
    # 7) Clean up compressed file if not keeping persistent copy
    # ------------------------------------------------------------------
    if not write_to_disk:
        try:
            os.unlink(path)
        except OSError:
            pass

    if use_cache and not backed:
        _H5AD_CACHE[cache_key] = adata

    return adata