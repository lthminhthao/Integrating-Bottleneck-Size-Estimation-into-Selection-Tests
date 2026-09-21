import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from neutrality_test import neutrality_lrt
from bottleneck_function import bottleneck_from_two_timepoints

st.set_page_config(page_title="Bottleneck Neutrality Test", layout="wide")

# ============================ HEADER / LANDING ============================
st.title("Bottleneck Neutrality Test - From Donor to Recipient Diversity "
         "Accounting for Population Bottleneck")
st.caption("Software version v1 (September 2026)")

st.markdown("**Thi Minh Thao Le and Erida Gjini**")

st.markdown(
    "Building on existing computational approaches (refs 1-3), we develop a new "
    "software framework that integrates explicit bottleneck size estimation into "
    "neutrality testing for biological diversity data. Designed for variant frequency "
    "datasets, our method accounts for sequencing errors and sampling biases, enabling "
    "more accurate and interpretable detection of selection signatures. The framework has "
    "been validated using previously published *Streptococcus pneumoniae* in vivo "
    "experimental data (Liu et al., ref 4), where it successfully reproduced established "
    "fitness results while identifying additional genes associated with infection and "
    "pathogenesis. By explicitly modeling bottleneck effects, the bottleneck-neutrality "
    "(BN) test refines the identification of candidate genes under selection and helps "
    "distinguish genetic drift from selection. Its flexible design makes it a robust and "
    "broadly applicable tool for analyzing evolutionary dynamics across diverse biological "
    "systems."
)

st.markdown("For more information on the method see our paper:")
st.markdown(
    "Le, TMT, Gjini E. (2026) "
    "[Integrating Bottleneck Size into Selection Tests for Biological Diversity Data]"
    "(https://doi.org/10.1101/2026.07.07.737025) bioRxiv 2026.07.07.737025"
)

st.markdown(
    "**Funding:** This project has received funding from the European Union's Horizon "
    "Europe research and innovation programme under grant agreement No 101080528."
)

with st.expander("References"):
    st.markdown(
        "1. Leonard *et al.* (2017). Transmission bottleneck size estimation from pathogen "
        "deep-sequencing data, with an application to human influenza A virus. "
        "*Journal of Virology*, 91(14):10-1128.\n\n"
        "2. Abel S, Abel zur Wiesch P, Davis BM, Waldor MK (2015). Analysis of bottlenecks "
        "in experimental models of infection. *PLoS Pathogens*, 11(6):e1004823.\n\n"
        "3. Krimbas *et al.* (1971). The genetics of *Dacus oleae*. V. Changes of esterase "
        "polymorphism in a natural population following insecticide control-selection or "
        "drift? *Evolution*, pages 454-460.\n\n"
        "4. Liu *et al.* (2020). Exploration of bacterial bottlenecks and *Streptococcus "
        "pneumoniae* pathogenesis by CRISPRi-Seq. *Cell Host & Microbe*, 29(1):107-120."
    )

st.markdown("**Give it a try here in our specially-designed user interface!**")
st.markdown("You can upload your own data, run a random demo, or run analysis on the "
            "data used in our paper.")
st.markdown("---")
# ========================== END HEADER / LANDING ==========================

with st.sidebar:
    st.header("Parameters")
    fdr_alpha   = st.slider("FDR alpha", 0.01, 0.20, 0.05, 0.01)
    pseudocount = st.number_input("Donor pseudocount", value=1e-6, format="%.2e")
    min_p       = st.number_input("min_p (skip near-zero donor freqs)", value=1e-8, format="%.2e")
    fdr_method  = st.selectbox("FDR method", ["fdr_bh", "fdr_by", "holm", "bonferroni"])
    with st.expander(r"Advanced: $N_b$ estimator"):
        nb_max = int(st.number_input("nb_max", value=5000000, step=100000))
        n_grid = int(st.number_input("coarse grid points", value=220, step=20))
        refine = int(st.number_input("refine points", value=220, step=20))
    st.markdown("---")
    st.markdown("**Input format** - CSV files, columns = samples/recipients, rows = variants "
                "(each variant defined by one or a set of features, e.g. presence of a gene).")

# ---- Method / data description (shown just before the demo) ----
st.markdown(
    "The data consist of variant counts in the **donor** environment (or $t=t_0$) and "
    "variant counts in the **recipient** environment ($t=t_1$). There can be multiple "
    "recipient environments (e.g. different hosts or realisations of the experiment). "
    "Each variant could be defined by one or a set of features (e.g. presence of a gene, "
    "or multiple genes)."
)
st.markdown(
    "The model uses a likelihood-based approach to disentangle drift from diffusion by "
    "coupling donor-recipient data, accounting for eventual bottlenecks in the growth or "
    "transmission process."
)
st.markdown(
    "The first step is to estimate the bottleneck size $N_b$. The second step is to apply "
    "the **Bottleneck-Neutrality (BN) test** to identify variants that display a selection "
    "signature."
)

tab_upload, tab_demo = st.tabs(["Upload your data", "Run demo"])

donor_df = None
recipient_df = None

with tab_upload:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Donor / inoculum counts")
        st.caption("One column of raw counts; one row per variant.")
        donor_file = st.file_uploader("Upload donor CSV", type="csv", key="donor")
        if donor_file:
            donor_raw = pd.read_csv(donor_file, index_col=0)
            st.dataframe(donor_raw.head(), use_container_width=True)
            donor_df = donor_raw.T
    with col2:
        st.subheader("Recipient counts")
        st.caption("Columns = recipients, rows = variants (same variant order as donor).")
        recip_file = st.file_uploader("Upload recipient CSV", type="csv", key="recip")
        if recip_file:
            recip_raw = pd.read_csv(recip_file, index_col=0)
            st.dataframe(recip_raw.head(), use_container_width=True)
            recipient_df = recip_raw.T

with tab_demo:
    st.markdown(
        "Generates **synthetic** data: 8 variants, 6 recipients. "
        "Variants A and B are shifted in recipients to simulate non-neutrality. "
        "Shown in the documented layout (rows = variants, columns = samples)."
    )
    if st.button("Generate demo data"):
        rng = np.random.default_rng(42)
        K, M = 8, 6
        variants = [f"Variant {chr(65+i)}" for i in range(K)]
        donor_counts_demo = rng.integers(50, 500, size=K).astype(float)
        donor_df = pd.DataFrame([donor_counts_demo], columns=variants, index=["donor"])
        p = donor_counts_demo / donor_counts_demo.sum()
        p_recip = p.copy()
        p_recip[0] *= 3; p_recip[1] *= 0.2
        p_recip /= p_recip.sum()
        rows = []
        for _ in range(M):
            n = rng.integers(800, 1200)
            rows.append(rng.multinomial(n, p_recip))
        recipient_df = pd.DataFrame(rows, columns=variants, index=[f"recipient_{i+1}" for i in range(M)])
        st.session_state["donor_df"] = donor_df
        st.session_state["recipient_df"] = recipient_df
        st.success("Demo data generated!")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Donor** (rows = variants)"); st.dataframe(donor_df.T, use_container_width=True)
        with c2:
            st.write("**Recipients** (rows = variants)"); st.dataframe(recipient_df.T, use_container_width=True)

if donor_df is None and "donor_df" in st.session_state:
    donor_df = st.session_state["donor_df"]
if recipient_df is None and "recipient_df" in st.session_state:
    recipient_df = st.session_state["recipient_df"]

if donor_df is not None and recipient_df is not None:
    st.markdown("---")
    if st.button(r"Estimate bottleneck size ($N_b$) and apply BN test", type="primary"):
        if donor_df.shape[0] == 1:
            donor_series = donor_df.iloc[0]
        elif donor_df.shape[1] == 1:
            donor_series = donor_df.iloc[:, 0]
        else:
            donor_series = donor_df.iloc[0]
            st.warning("Donor has multiple samples; using the first.")

        if set(recipient_df.columns) == set(donor_series.index):
            recipient_df = recipient_df.reindex(columns=donor_series.index)

        donor_vec = donor_series.to_numpy(dtype=float)
        rec_names = list(recipient_df.index)
        nb_list = []
        prog = st.progress(0.0, text="Estimating bottleneck size N_b per recipient ...")
        for k, name in enumerate(rec_names):
            x = recipient_df.loc[name].to_numpy()
            nb_list.append(bottleneck_from_two_timepoints(
                donor_vec, x, donor_pseudocount=pseudocount,
                nb_max=nb_max, n_grid=n_grid, refine=refine))
            prog.progress((k + 1) / len(rec_names))
        prog.empty()
        nb_arr = np.array(nb_list, dtype=float)

        valid = np.isfinite(nb_arr)
        if not valid.all():
            st.warning(f"{int((~valid).sum())} recipient(s) had zero total counts and were dropped.")
        recipient_used = recipient_df.loc[valid]
        nb_used = nb_arr[valid]
        if recipient_used.shape[0] == 0:
            st.error("No usable recipients (all had zero counts).")
            st.stop()

        nb_table = pd.DataFrame({"recipient": list(recipient_used.index),
                                 "Nb_hat": nb_used.astype(int)})

        with st.spinner("Applying the BN test ..."):
            results = neutrality_lrt(
                donor_counts=donor_series,
                recipient_counts=recipient_used,
                bottleneck=nb_used,
                pseudocount=pseudocount,
                min_p=min_p,
                fdr_alpha=fdr_alpha,
                fdr_method=fdr_method,
            )
        results = results.rename(columns={"feature": "variant"})
        st.session_state["results"] = results
        st.session_state["nb_table"] = nb_table

    if "results" in st.session_state:
        results = st.session_state["results"]

        if "nb_table" in st.session_state:
            nb_table = st.session_state["nb_table"]
            st.markdown("### 1. Estimated bottleneck size ($N_b$) per recipient")
            cA, cB = st.columns([3, 1])
            with cA:
                st.dataframe(nb_table, use_container_width=True, hide_index=True)
            with cB:
                st.metric(r"Mean $N_b$", f"{nb_table['Nb_hat'].mean():.0f}")
                st.metric(r"Median $N_b$", f"{nb_table['Nb_hat'].median():.0f}")
            st.download_button(r"Download $N_b$ per recipient (CSV)",
                               nb_table.to_csv(index=False).encode(),
                               "bottleneck_Nb_per_recipient.csv", "text/csv")

        n_tested   = results["pval"].notna().sum()
        n_rejected = results["reject_FDR"].sum()
        n_down_sig = int(((results["direction"] == "down_in_recipient") & results["reject_FDR"]).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Variants tested", int(n_tested))
        m2.metric(f"Selected (FDR {int(fdr_alpha*100)}%)", int(n_rejected))
        m3.metric("Neutral", int(n_tested - n_rejected))
        m4.metric("Sig. depleted", n_down_sig)

        st.markdown("### 2. Results table for variant frequency changes")
        display_cols = ["variant","p_donor","q_hat_recipient_center","LR","pval","qval_FDR","reject_FDR","direction"]
        disp = results[display_cols].copy()
        for c in ["p_donor","q_hat_recipient_center"]:
            disp[c] = disp[c].map(lambda v: f"{v:.4e}" if pd.notna(v) else "")
        disp["LR"] = disp["LR"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        for c in ["pval","qval_FDR"]:
            disp[c] = disp[c].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        down_df = results[(results["direction"] == "down_in_recipient") & (results["reject_FDR"])].copy()
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button("Download full results CSV",
                               results.to_csv(index=False).encode(),
                               "BN_test_results.csv", "text/csv")
        with dcol2:
            st.download_button(f"Download significant down_in_recipient ({len(down_df)}) CSV",
                               down_df.to_csv(index=False).encode(),
                               "down_in_recipient_significant.csv", "text/csv")
        st.caption("The down_in_recipient file lists only SIGNIFICANT depletions "
                   "(reject_FDR = True and recipient frequency below donor frequency).")

        st.markdown("### 3. Interpreting variant frequency changes: log2 fold change")
        eps = 1e-12
        plot_df = results.dropna(subset=["pval","q_hat_recipient_center","p_donor"]).copy()
        plot_df["log2_fc"] = np.log2((plot_df["q_hat_recipient_center"] + eps) / (plot_df["p_donor"] + eps))
        plot_df["neglog10_p"] = -np.log10(plot_df["pval"])
        rej = plot_df[plot_df["reject_FDR"]]
        hline = -np.log10(rej["pval"].max()) if len(rej) else -np.log10(fdr_alpha)
        fig, ax = plt.subplots(figsize=(6.4, 5))
        b = plot_df[~plot_df["reject_FDR"]]; r = plot_df[plot_df["reject_FDR"]]
        ax.scatter(b["log2_fc"], b["neglog10_p"], s=14, c="#3b7fbf", alpha=0.75, edgecolors="none")
        ax.scatter(r["log2_fc"], r["neglog10_p"], s=34, c="#e8261f", alpha=0.95, edgecolors="none")
        ax.axvline(0.0, ls="--", lw=1, color="#6b7a99")
        ax.axhline(hline, ls="--", lw=1, color="#6b7a99")
        ax.set_xlabel(r"$log_2(\hat{q}_j / D_j)$", fontsize=12)
        ax.set_ylabel(r"$-log_{10}(p_{FDR})$", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(f"Red = selected at FDR {fdr_alpha}. Vertical dashed line: no change. "
                   f"Horizontal dashed line: FDR significance boundary. Left of centre = depleted in recipients.")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download scatter (PNG)", buf.getvalue(), "BN_test_scatter.png", "image/png")
else:
    st.info("Upload donor + recipient CSVs, or click Generate demo data to get started.")
