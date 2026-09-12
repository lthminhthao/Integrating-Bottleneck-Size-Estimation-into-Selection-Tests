
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from neutrality_test import neutrality_lrt

# -- Page config --------------------------------------------------------------
st.set_page_config(page_title="Neutrality LRT", layout="wide")
st.title("Neutrality Test - Donor to Recipient Transmission")
st.markdown(
    """
    Test whether each feature (taxon / gene) is **neutral** during transmission:
    - **H0**: recipient center frequency = donor frequency
    - **H1**: recipient center frequency is free
    Uses a likelihood-ratio test (LRT) with Dirichlet-Multinomial marginals
    and Benjamini-Hochberg FDR correction.
    """
)

# -- Sidebar - parameters -----------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    bottleneck_input = st.text_input(
        "Bottleneck size Nb (scalar or comma-separated per recipient)",
        value="100",
        help="A single number (same Nb for all recipients) or one value per recipient sample."
    )
    fdr_alpha   = st.slider("FDR alpha", 0.01, 0.20, 0.05, 0.01)
    pseudocount = st.number_input("Donor pseudocount", value=1e-6, format="%.2e")
    min_p       = st.number_input("min_p (skip near-zero donor freqs)", value=1e-8, format="%.2e")
    fdr_method  = st.selectbox("FDR method", ["fdr_bh", "fdr_by", "holm", "bonferroni"])
    st.markdown("---")
    st.markdown("**Input format** - CSV files, columns = samples/recipients, rows = features (taxa / genes).")

# -- Data input tabs ----------------------------------------------------------
tab_upload, tab_demo = st.tabs(["Upload your data", "Run demo"])

def parse_bottleneck(text, M):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        return float(parts[0])
    arr = np.array([float(p) for p in parts])
    if len(arr) != M:
        st.error(f"Bottleneck has {len(arr)} values but there are {M} recipients.")
        st.stop()
    return arr

donor_df     = None
recipient_df = None

with tab_upload:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Donor counts")
        st.caption("One column of raw counts; one row per feature.")
        donor_file = st.file_uploader("Upload donor CSV", type="csv", key="donor")
        if donor_file:
            donor_raw = pd.read_csv(donor_file, index_col=0)   # rows = features
            st.dataframe(donor_raw.head(), use_container_width=True)
            donor_df = donor_raw.T                             # -> 1 row (sample), cols = features
    with col2:
        st.subheader("Recipient counts")
        st.caption("Columns = recipients, rows = features (same feature order as donor).")
        recip_file = st.file_uploader("Upload recipient CSV", type="csv", key="recip")
        if recip_file:
            recip_raw = pd.read_csv(recip_file, index_col=0)   # rows = features, cols = recipients
            st.dataframe(recip_raw.head(), use_container_width=True)
            recipient_df = recip_raw.T                         # -> rows = recipients, cols = features

with tab_demo:
    st.markdown(
        """
        Generates **synthetic** data: 8 features, 6 recipients.
        Features A and B are shifted in recipients to simulate non-neutrality.
        Shown in the documented layout (rows = features, columns = samples).
        """
    )
    if st.button("Generate demo data"):
        rng = np.random.default_rng(42)
        K, M = 8, 6
        features = [f"Feature_{chr(65+i)}" for i in range(K)]
        donor_counts_demo = rng.integers(50, 500, size=K).astype(float)
        donor_df = pd.DataFrame([donor_counts_demo], columns=features, index=["donor"])
        p = donor_counts_demo / donor_counts_demo.sum()
        p_recip = p.copy()
        p_recip[0] *= 3; p_recip[1] *= 0.2
        p_recip /= p_recip.sum()
        rows = []
        for _ in range(M):
            n = rng.integers(800, 1200)
            rows.append(rng.multinomial(n, p_recip))
        recipient_df = pd.DataFrame(rows, columns=features,
                                    index=[f"recipient_{i+1}" for i in range(M)])
        st.session_state["donor_df"]     = donor_df
        st.session_state["recipient_df"] = recipient_df
        st.success("Demo data generated!")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Donor** (rows = features)"); st.dataframe(donor_df.T, use_container_width=True)
        with col2:
            st.write("**Recipients** (rows = features)"); st.dataframe(recipient_df.T, use_container_width=True)

# pull demo data into local vars if present (stored as recipients-as-rows for the model)
if donor_df is None and "donor_df" in st.session_state:
    donor_df     = st.session_state["donor_df"]
if recipient_df is None and "recipient_df" in st.session_state:
    recipient_df = st.session_state["recipient_df"]

# -- Run test -----------------------------------------------------------------
if donor_df is not None and recipient_df is not None:
    st.markdown("---")
    if st.button("Run neutrality LRT", type="primary"):
        with st.spinner("Running LRT ..."):
            if donor_df.shape[0] == 1:
                donor_series = donor_df.iloc[0]
            elif donor_df.shape[1] == 1:
                donor_series = donor_df.iloc[:, 0]
            else:
                donor_series = donor_df.iloc[0]
                st.warning("Donor has multiple samples; using the first.")

            M = recipient_df.shape[0]
            Nb = parse_bottleneck(bottleneck_input, M)

            results = neutrality_lrt(
                donor_counts=donor_series,
                recipient_counts=recipient_df,
                bottleneck=Nb,
                pseudocount=pseudocount,
                min_p=min_p,
                fdr_alpha=fdr_alpha,
                fdr_method=fdr_method,
            )
            st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]

        n_tested   = results["pval"].notna().sum()
        n_rejected = results["reject_FDR"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Features tested", int(n_tested))
        m2.metric(f"Rejected (FDR {int(fdr_alpha*100)}%)", int(n_rejected))
        m3.metric("Not rejected (neutral)", int(n_tested - n_rejected))

        st.subheader("Results table")
        display_cols = ["feature","p_donor","q_hat_recipient_center",
                        "LR","pval","qval_FDR","reject_FDR","direction"]
        styled = results[display_cols].style.format(
            {"p_donor":"{:.4f}","q_hat_recipient_center":"{:.4f}",
             "LR":"{:.3f}","pval":"{:.2e}","qval_FDR":"{:.2e}"}
        ).apply(
            lambda col: ["background-color: #ffd6d6" if v else "" for v in col],
            subset=["reject_FDR"]
        )
        st.dataframe(styled, use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode()
        st.download_button("Download results CSV", csv_bytes,
                           "neutrality_results.csv", "text/csv")

        st.subheader("Visualisations")
        plot_tab1, plot_tab2, plot_tab3 = st.tabs(
            ["p-value distribution", "Volcano (LR vs -log10 p)", "Freq shift"]
        )

        with plot_tab1:
            fig, ax = plt.subplots(figsize=(7, 3))
            pv = results["pval"].dropna()
            ax.hist(pv, bins=20, color="steelblue", edgecolor="white")
            ax.axvline(fdr_alpha, color="red", linestyle="--", label=f"alpha={fdr_alpha}")
            ax.set_xlabel("p-value"); ax.set_ylabel("Count")
            ax.set_title("p-value histogram"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with plot_tab2:
            fig, ax = plt.subplots(figsize=(7, 4))
            sub = results.dropna(subset=["pval","LR"])
            colors = ["#e74c3c" if r else "#95a5a6" for r in sub["reject_FDR"]]
            ax.scatter(sub["LR"], -np.log10(sub["pval"]), c=colors, alpha=0.8, edgecolors="none")
            for _, row in sub[sub["reject_FDR"]].iterrows():
                ax.annotate(str(row["feature"]), (row["LR"], -np.log10(row["pval"])),
                            fontsize=7, ha="left", va="bottom")
            ax.set_xlabel("Likelihood-ratio statistic")
            ax.set_ylabel("-log10(p-value)")
            ax.set_title("Volcano plot")
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", label="Rejected"),
                Line2D([0],[0], marker="o", color="w", markerfacecolor="#95a5a6", label="Not rejected"),
            ]
            ax.legend(handles=legend_elements)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with plot_tab3:
            fig, ax = plt.subplots(figsize=(8, 4))
            sub = results.dropna(subset=["q_hat_recipient_center"])
            x = np.arange(len(sub))
            ax.bar(x, sub["q_hat_recipient_center"] - sub["p_donor"],
                   color=["#e74c3c" if r else "#3498db" for r in sub["reject_FDR"]])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["feature"].astype(str), rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("q_hat - p_donor  (frequency shift)")
            ax.set_title("Frequency shift per feature (red = rejected)")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
else:
    st.info("Upload donor + recipient CSVs, or click Generate demo data to get started.")
