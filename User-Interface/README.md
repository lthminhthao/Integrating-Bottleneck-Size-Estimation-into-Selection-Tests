# Naive-BN-test interface with the BN test (with bootstrap function)

This package supersedes the previous recreated interface. `app_neutrality.py`
is based directly on the supplied original app. Its landing text, authors,
license statement, funding, references, original controls, upload tabs, demo
code, output layout, download names, plot style and fold-change calculation
are retained. No custom theme or extra refitting controls are introduced.

## Run

From this extracted folder, with Python 3.11 or 3.12:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app_neutrality.py
```

The new backend requires a C++17 compiler (g++ or clang++) and a writable
working directory. On macOS install Xcode command-line tools; on Linux install
g++. For Streamlit Community Cloud upload `packages.txt` alongside the Python
files and requirements, and select `app_neutrality.py` as the entry point.

Alternatively, run `BN-test_app.ipynb` to regenerate the app and backend.
The notebook's original launch cell includes an optional localtunnel call;
only run that cell if you want to launch the app and attempt a public tunnel.

## Exact changes

1. Replace the old test/estimator imports with the functions from
   `neutrality_test_bootstrap.py`, copied byte-for-byte from the supplied
   `neutrality_test_bootstrap(4).py`.
2. Add Number of bootstrap replicates (default 100), passed as `n_boot`.
3. Call `neutrality_bootstrap`, unpack its results/diagnostics, and show the
   replicate count while running. Bottlenecks are estimated once and supplied
   to the test; `reestimate_bottleneck=False`, matching the function default.
   Seed is 12345. No optional refitting control is added.
4. Preserve valid infinite bottleneck estimates rather than dropping them or
   converting them to integers. Preserve the original zero-count-recipient
   exclusion explicitly, because the new estimator raises on empty samples.
5. Add `agreement_LR` and `direction_agreement` to the displayed results so the
   new gated statistic is visible alongside the raw LR. Full CSV downloads
   contain all columns returned by the new function.
6. Label the plot's existing raw-p-value axis as bootstrap p-value (the old
   label said adjusted p-value although the plotted column was `pval`).
7. Require SciPy >=1.11 and add g++ to packages.txt for the compiled backend.
   Update the notebook's backend, app and sanity-check cells consistently;
   clear old execution outputs so they do not show obsolete results.

The full app diff is in `changes.diff`. The original coarse-grid and refine
controls remain as requested, but the supplied new estimator accepts and
ignores those two legacy arguments. Its own internal search is unchanged.

The original descriptive text and labels are retained as requested; this UI
update does not revalidate scientific claims in that text. The supplied
backend documents its statistical limitations.

## Checks completed

- Original header and synthetic demo code match the source exactly.
- The notebook regenerates the exact updated app source.
- The backend matches the supplied bootstrap file byte for byte.
- Streamlit AppTest: demo, bootstrap-count input, results and plot execute.
- App results match a direct backend call with the same data and settings.
- Infinite bottleneck and zero-count-recipient handling pass.

Tests used three bootstrap replicates for a short software smoke check;
these checks are not a statistical validation study.
