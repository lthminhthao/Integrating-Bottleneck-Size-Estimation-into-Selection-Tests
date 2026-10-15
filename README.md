# Integrating Bottleneck Size Estimation into Selection Tests for Biological Diversity Data

Code accompanying the study by Thi Minh Thao Le and Erida Gjini. The Bottleneck-Neutrality (BN) test compares donor and recipient variant frequencies while incorporating a bottleneck parameter for each recipient.

The current implementation uses Dirichlet–multinomial maximum likelihood to estimate the bottleneck parameter and beta-binomial marginal likelihoods to test each variant. The likelihood-ratio test uses a chi-square reference distribution with one degree of freedom, followed by multiple-testing correction (Benjamini–Hochberg by default). It does not implement bootstrap calibration.

Paper: [Integrating Bottleneck Size into Selection Tests for Biological Diversity Data](https://doi.org/10.1101/2026.07.07.737025).

## Repository contents

| Location | Purpose |
| --- | --- |
| `User-Interface/app_neutrality.py` | Streamlit interface for a synthetic demo or uploaded count data. |
| `User-Interface/requirements.txt` | Dependencies for the interface. |
| `BNtest/neutrality_test_Dox.ipynb` | 24 hpi induced-mouse analysis, using Pre2 as donor. |
| `BNtest/neutrality_test_noDox.ipynb` | 24 hpi uninduced-mouse analysis, using Pre1 as donor. |
| `BNtest/neutrality_test.py` | Reusable `neutrality_lrt` function. |
| `BNtest/compare_sgRNA_Dox.ipynb` | Compare detected depleted sgRNAs with Table S5. |
| `BNtest/Table-S7-sgRNA-Raw-Counts.csv` | Raw-count input for the 24 hpi analysis notebooks. |
| `BNtest/Leonard_Nb_24hpi.csv` | Supplied per-mouse bottleneck estimates used by those notebooks. |
| `bottleneck_computation/Bottleneck_Leonard.ipynb` | Recompute bottleneck estimates for 24 and 48 hpi. |
| `bottleneck_computation/Nb_Comparison.ipynb`, `bottleneck_computation/bottleneck_ratio.ipynb` | Additional bottleneck comparison analyses. |

The example data originate from Liu et al., *Exploration of bacterial bottlenecks and Streptococcus pneumoniae pathogenesis by CRISPRi-Seq*, Cell Host & Microbe 29(1):107–120 (2021).

## Installation

Install Python 3 and Git, then run:

```bash
git clone https://github.com/lthminhthao/Integrating-Bottleneck-Size-Estimation-into-Selection-Tests.git
cd Integrating-Bottleneck-Size-Estimation-into-Selection-Tests
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r User-Interface/requirements.txt
python -m pip install jupyterlab openpyxl
```

The requirements file includes Streamlit, NumPy, pandas, SciPy, statsmodels and Matplotlib. JupyterLab runs the notebooks; openpyxl reads Table S5. Dependency versions are not pinned in the repository.

## Quick start: interactive demo

From the repository root, run:

```bash
python -m streamlit run User-Interface/app_neutrality.py
```

Open the local URL printed in the terminal.

1. Select **Run demo**.
2. Click **Generate demo data**. This creates eight variants and six recipients with a fixed random seed.
3. Click **Estimate bottleneck size ($N_b$) and apply BN test**.
4. Inspect the per-recipient bottleneck estimates, summary counts, result table and scatter plot.
5. Use the download buttons to save the results:

| Download | Contents |
| --- | --- |
| `bottleneck_Nb_per_recipient.csv` | Recipient names and estimated bottleneck parameters. |
| `BN_test_results.csv` | Full variant-level results. |
| `down_in_recipient_significant.csv` | Rows with `reject_FDR = True` and `direction = down_in_recipient`. |
| `BN_test_scatter.png` | Scatter plot of frequency changes and significance. |

Downloads are saved by your browser. The demo illustrates the workflow; it is not a reproduction of the experimental dataset. Stop the local app with Ctrl+C in the terminal.

## Run the 24 hpi experimental analyses

From the repository root, start JupyterLab inside `BNtest` so that the notebooks can find their relative input paths and Python modules:

```bash
cd BNtest
python -m jupyterlab
```

Open either notebook below, select a Python kernel from the installed environment, and run all cells in order.

| Notebook | Comparison | CSV written to `BNtest` |
| --- | --- | --- |
| `neutrality_test_Dox.ipynb` | Pre2 donor versus `Mice_plusDox_*` recipients at 24 hpi | `neutrality_test_24hpi_Dox.csv` |
| `neutrality_test_noDox.ipynb` | Pre1 donor versus `Mice_noDox_*` recipients at 24 hpi | `neutrality_test_24hpi_noDox.csv` |

Both notebooks read `Table-S7-sgRNA-Raw-Counts.csv` and `Leonard_Nb_24hpi.csv` from the same directory. Keep the original Table S7 headers: the code uses the first header row for experimental groups and the next row for sample names. Bottleneck values are matched to recipients by mouse name.

The notebooks use the supplied bottleneck estimates; they do not recompute them. They compare each recipient group with its donor, not the induced group directly with the uninduced group.

The final cell saves **significant rows only**, including both directions. Running it replaces the existing CSV with the same name. To inspect and save all results, add a cell:

```python
display(res.head(20))
print(res.groupby(["reject_FDR", "direction"]).size())
res.to_csv("neutrality_results_all.csv", index=False)

depleted = res.loc[
    res["reject_FDR"] & res["direction"].eq("down_in_recipient")
]
display(depleted)
depleted.to_csv("neutrality_results_depleted.csv", index=False)
```

Use distinct output names if saving both conditions. After running the induced-mouse notebook, run `compare_sgRNA_Dox.ipynb` in the same directory to print the overlap with the in-vivo essentiality calls in the supplied Table S5 workbook.

## Recompute bottleneck estimates (optional)

Start JupyterLab from `bottleneck_computation`, open `Bottleneck_Leonard.ipynb`, and run its cells in order. It reads the local copy of Table S7 and writes `Leonard_Nb_24hpi.csv` and `Leonard_Nb_48hpi.csv` in that directory.

To use the newly computed 24 hpi estimates in the neutrality notebooks, copy `Leonard_Nb_24hpi.csv` into `BNtest`, replacing the supplied estimates only if this is intended. The two folders contain separate input files; keep track of the input copy used when comparing runs.

The estimator searches an integer grid with a default upper bound of 5,000,000, followed by local refinement. The 48 hpi notebook uses Pre1 for Dox and Pre2 for noDox, whereas the 24 hpi mapping is the reverse. Check this mapping against the experimental design before adapting the analysis.

## Analyze your own data with the interface

Choose **Upload your data** and upload two CSV files. Both must have a first column of unique variant IDs and matching variant rows. Use finite, nonnegative integer counts, with positive total counts for the donor and usable recipients.

Example donor file:

```csv
variant,donor
sgRNA0001,100
sgRNA0002,200
sgRNA0003,300
```

Example recipient file:

```csv
variant,recipient_1,recipient_2
sgRNA0001,50,80
sgRNA0002,220,190
sgRNA0003,330,350
```

These examples illustrate the file layout, not a dataset for scientific validation. Upload one donor count column and one or more recipient columns. Do not upload the original multi-header Table S7 directly into these fields: first extract the donor and recipients into this layout.

Adjust parameters in the sidebar if needed, then click the estimation/test button. Defaults include `fdr_alpha = 0.05`, `fdr_method = fdr_bh`, donor pseudocount `1e-6`, and `min_p = 1e-8`.

## Interpret the results

| Column | Meaning |
| --- | --- |
| `feature` (or `variant` in the app) | Variant or sgRNA identifier. |
| `p_donor` | Donor frequency after pseudocount normalization. |
| `q_hat_recipient_center` | Fitted recipient center frequency across hosts. |
| `LR` | Likelihood-ratio statistic. |
| `pval` | Raw chi-square-calibrated p-value. |
| `qval_FDR` | Multiple-testing-adjusted p-value; BH by default. |
| `reject_FDR` | Whether the null is rejected at the selected correction threshold. |
| `direction` | Direction of the fitted frequency change; use together with `reject_FDR`. |

A significant depleted candidate has `reject_FDR = True` and `direction = down_in_recipient`. A non-rejection does not establish neutrality, and statistical depletion alone does not establish gene essentiality. Frequencies are relative abundances, so enrichment need not imply an increase in absolute abundance. Features skipped by the donor-frequency filter have missing test statistics/p-values and `reject_FDR = False`.

The current test treats supplied bottleneck estimates as fixed. Numerical results can depend on inputs, bottleneck values, filtering, and software versions; retain these settings with exported results.

## Troubleshooting and current implementation notes

- **File or module not found:** launch JupyterLab from the notebook's directory. Keep its input files and helper modules together.
- **Missing Nb entries:** check the mouse IDs, `condition`, `hpi`, and `lung` fields in `Leonard_Nb_24hpi.csv` against the recipient names selected by the notebook.
- **Excel import error:** install `openpyxl` in the notebook kernel's environment.
- **No usable recipients in the app:** check for zero-total recipient columns. The app drops these recipients.
- **No-dox comparison notebook:** `compare_sgRNA_noDox.ipynb` currently expects `neutrality_noDox_FDR0.05.csv` and an `sgRNA` column. To use the output above, change its input filename to `neutrality_test_24hpi_noDox.csv` and its result-column access from `df["sgRNA"]` to `df["feature"]`. Its final in-vivo overlap print statement should print `joint_invivo`, not `joint`.
- **App plot label:** the current app computes the vertical coordinate from raw `pval`, although the axis label says `p_FDR`. Use `qval_FDR` and `reject_FDR` in the results table to assess adjusted significance. The app's “Neutral” counter means tested variants for which the null was not rejected.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See [LICENSE](LICENSE). Since the LGPL builds on the GNU General Public License v3.0, a copy of the GPL is included in [COPYING](COPYING).
