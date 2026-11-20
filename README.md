# Integrating Bottleneck Size Estimation into Selection Tests

Code accompanying the manuscript by **Thi Minh Thao Le and Erida Gjini**. The Bottleneck-Neutrality (BN) test assesses donor-to-recipient frequency changes against a neutral model incorporating bottleneck effects.

[Manuscript](https://doi.org/10.1101/2026.07.07.737025)

## Repository contents

| Folder | Contents and purpose |
| --- | --- |
| `BNtest/` | Bootstrap BN test implementation, analysis of the 24 hpi experimental data, candidate comparisons, result plots, and simulation comparisons with DESeq2. |
| `bottleneck_computation/` | Bottleneck estimation, comparison with the Abel-based estimator, and analysis of ratios between estimates at 24 and 48 hpi. |
| `User-Interface/` | Streamlit application for running the BN test on example or uploaded data, with dependencies and setup instructions. |
| `data-normalise/` | Standardizes sgRNA identifiers to four-digit format; does not normalize read counts. |
| `naive-bottleneck-and-naiveBNtest/` | Earlier bottleneck estimation and neutrality-testing code, retained for comparison with the revised bootstrap method. |

Experimental data are from Liu et al. (2021), *Exploration of Bacterial Bottlenecks and Streptococcus pneumoniae Pathogenesis by CRISPRi-Seq*.

## Usage

Open the relevant notebook and follow its instructions. Run notebooks from their containing folders so that relative input paths resolve correctly. For the interactive application, see [User-Interface/README.md](User-Interface/README.md).

Significant results indicate departures from the specified neutral model; they do not alone establish gene essentiality or a causal fitness effect.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See [LICENSE](LICENSE). Since the LGPL builds on the GNU General Public License v3.0, a copy of the GPL is included in [COPYING](COPYING).
