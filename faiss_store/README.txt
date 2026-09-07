This directory is intentionally empty in the delivered project.

The FAISS index and pickles that used to ship here were built from the OLD
3-column dataset (scheme_name, category, description) and are not
compatible with the new enriched dataset (18 columns, including the
structured eligibility fields the rest of this upgrade depends on). Rather
than ship stale files that would silently disable eligibility checking
(app.py would load them without error but every eligibility field would
just be missing), this directory is left empty and app.py/evaluate.py/
test_retrieval.py all fail with a clear message telling you to rebuild it.

Regenerate these files with:
    python data_cleaning.py
    python enrich_eligibility.py
    python build_index.py

This requires faiss-cpu, sentence-transformers, and their model weights
(all-MiniLM-L6-v2, downloaded on first use) -- see requirements.txt.
