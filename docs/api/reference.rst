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

.. autodata:: zmap.reference.load_h5ad.H5AD_SOURCES
   :annotation: = {...}

   Dictionary mapping preset names (``"raw"``, ``"processed"``,
   ``"processed_slim"``, ``"processed_slim_tpm"``, ``"symphony"``) to
   their CDN URLs and filenames.
