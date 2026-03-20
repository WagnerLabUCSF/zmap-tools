``zmap.predict`` — Label Transfer & Annotation
================================================

.. module:: zmap.predict

End-to-end pipeline and lower-level functions for transferring ZMAP
reference labels to query single-cell datasets via kNN voting in
Symphony/Harmony embedding space.


Full Pipeline
-------------

.. autofunction:: annotate_with_zmap


Preprocessing
-------------

.. autofunction:: preprocess_adata_query


kNN Label Transfer
------------------

.. autofunction:: predict_labels_kNN


Post-processing & Summaries
----------------------------

.. autofunction:: summarize_knn_run

.. autofunction:: aggregate_by_cluster

.. autofunction:: build_cell_annotations_table


Visualization
-------------

.. autofunction:: plot_embedding_with_ondata_labels

.. autofunction:: plot_colorbar_histogram

.. autofunction:: sync_zmap_colors

.. autofunction:: map_query_labels
