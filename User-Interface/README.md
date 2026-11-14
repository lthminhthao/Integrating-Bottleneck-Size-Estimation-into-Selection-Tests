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

## Checks completed

- Original header and synthetic demo code match the source exactly.
- The notebook regenerates the exact updated app source.
- The backend matches the supplied bootstrap file byte for byte.
- Streamlit AppTest: demo, bootstrap-count input, results and plot execute.
- App results match a direct backend call with the same data and settings.
- Infinite bottleneck and zero-count-recipient handling pass.

Tests used three bootstrap replicates for a short software smoke check;
these checks are not a statistical validation study.
