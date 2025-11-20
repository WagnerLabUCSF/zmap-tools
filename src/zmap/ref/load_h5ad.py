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
    # Fully processed (large)
    "processed": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251008_processed.h5ad",
        "filename": "ZMAP_251008_processed.h5ad",
    },
    # Fully processed for plotting only (slim)
    "processed_slim": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251008_processed_slim.h5ad",
        "filename": "ZMAP_251008_processed_slim.h5ad",
    },
    # Processed slim / symphony reference
    "symphony": {
        "url": "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/h5ad/ZMAP_251016_symphony.h5ad",
        "filename": "ZMAP_251016_symphony.h5ad",
    },
}

# In-memory cache per (kind, url, backed)
_H5AD_CACHE: dict[tuple[str, str | None, bool], ad.AnnData] = {}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _default_h5ad_dir() -> pathlib.Path:
    """
    Default directory to store H5ADs:
        <cwd>/zmap/h5ads
    In Colab, this will be /content/zmap/h5ads.
    """
    root = pathlib.Path.cwd()
    d = root / "zmap" / "h5ads"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        print(f"[ZMAP] Using existing file: {dest_path}")
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
        adata.X = adata.layers["raw_nolog"].copy()
        sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
        sc.pp.log1p(adata)
        adata.layers["tpm_log"] = adata.X.copy()
        del adata.X

def load_zmap_h5ad(
    *,
    kind: str | None = "processed_slim",
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
    High-level loader for ZMAP H5ADs.

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

    cache_key = (kind or "custom", url, bool(backed))
    if use_cache and cache_key in _H5AD_CACHE:
        return _H5AD_CACHE[cache_key]

    # ----------------------------------------------------------------------
    # Download or reuse existing file
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    if backed:
        backed_mode = "r" if backed is True else backed
        print(f"[ZMAP] Loading (backed={backed_mode!r}) from {path}")
        adata = sc.read_h5ad(path, backed=backed_mode)
    else:
        print(f"[ZMAP] Loading into memory from {path}")
        adata = ad.read_h5ad(path)

    # ----------------------------------------------------------------------
    # Optional preprocessing step
    # ----------------------------------------------------------------------
    if attempt_preprocess_tpmlog and not backed:
        preprocess_tpmlog(adata)

    # ----------------------------------------------------------------------
    # Clean up if we didn't want a persistent copy
    # ----------------------------------------------------------------------
    if not write_to_disk:
        try:
            os.unlink(path)
        except OSError:
            pass

    if use_cache and not backed:
        _H5AD_CACHE[cache_key] = adata

    return adata


