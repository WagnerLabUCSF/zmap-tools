``zmap.reference`` — Reference Data Loading
=============================================

.. module:: zmap.reference

Functions for downloading, caching, and loading ZMAP reference datasets
and consensus marker tables. Aliased as ``zmap.ref``.


H5AD Loading
------------

.. autofunction:: load_zmap_h5ad

.. autofunction:: download_zmap_h5ad

.. autofunction:: preprocess_tpmlog


Consensus Markers
-----------------

.. autofunction:: load_consensus_markers


Data Registry
-------------

The following preset keys are available for the ``kind`` parameter:

- ``"raw"`` — raw counts, unprocessed.
- ``"processed"`` — fully processed, includes intermediate layers.
- ``"processed_slim"`` — fully processed, raw counts only.
- ``"processed_slim_tpm"`` — fully processed, TPM counts only (default).
- ``"symphony"`` — Symphony reference for query embedding and label transfer.
