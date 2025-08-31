import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

DEFAULT_PROMPT = """You are a seasoned financial analyst specializing in portfolio construction and risk management. 
Analyze the given portfolio (holdings, weights, and pairwise correlations) and present insights in Markdown tables only, with no extra commentary. 

Your output must include:

1. **Economic Regime Analysis**  
   Table format with columns: Regime | Expected Portfolio Performance | Key Drivers

2. **Risk Assessment**  
   Table format with columns: Risk Factor | Description | Why It Matters

3. **Diversification & Rebalancing Guidance**  
   Table format with columns: Recommendation | Rationale

Be concise, practical, and actionable. Keep recommendations broad (e.g., "consider adding uncorrelated commodities" rather than naming specific tickers). Do not output text outside the tables.
"""


# Load environment variables
load_dotenv()

st.set_page_config(page_title="Portfolio Correlation Analyzer", layout="wide")

st.title("📊 Portfolio Correlation Analyzer (5Y)")

# User input
st.sidebar.header("Your Portfolio")
tickers_input = st.sidebar.text_input("Enter tickers separated by commas", "BRK-B, GS, GOOGL, AMZN, ASML, UNH, GLD")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    # Fetch 5 years of data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)

    st.write(f"Fetching 5Y data for: {', '.join(tickers)}")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)

        # Handle single ticker vs multiple tickers
        if isinstance(raw_data, pd.DataFrame) and isinstance(raw_data.columns, pd.MultiIndex):
            if "Adj Close" in raw_data.columns.get_level_values(0):
                data = raw_data.xs("Adj Close", level=0, axis=1)
            elif "Close" in raw_data.columns.get_level_values(0):
                data = raw_data.xs("Close", level=0, axis=1)
            else:
                st.error("Neither 'Adj Close' nor 'Close' found in downloaded data.")
                st.stop()
        elif "Adj Close" in raw_data.columns:
            data = raw_data[["Adj Close"]]
            data.columns = tickers if len(tickers) == 1 else data.columns
        elif "Close" in raw_data.columns:
            data = raw_data[["Close"]]
            data.columns = tickers if len(tickers) == 1 else data.columns
        elif isinstance(raw_data, pd.Series):
            data = raw_data.to_frame(name=tickers[0])
        else:
            st.error("Unexpected data format returned by yfinance.")
            st.stop()

        # Drop missing values
        data = data.dropna()

        if data.empty:
            st.error("No valid price data found for the given tickers.")
            st.stop()

        # Compute daily returns
        returns = data.pct_change().dropna()

        # Compute correlation matrix
        corr_matrix = returns.corr()

        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)

        # Compute top correlated pairs (by absolute correlation) and visualize as graph
        if len(tickers) > 1:
            # Build list of unique pairs (i < j)
            pairs = []
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a = cols[i]
                    b = cols[j]
                    c = corr_matrix.iloc[i, j]
                    pairs.append((a, b, c))

            pairs_df = pd.DataFrame(pairs, columns=["asset_a", "asset_b", "corr"])
            # Rank by absolute correlation
            pairs_df["abs_corr"] = pairs_df["corr"].abs()
            pairs_df = pairs_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

            top_n = min(10, len(pairs_df))
            st.subheader(f"Top {top_n} Correlated Pairs")
            st.write("Pairs ranked by absolute correlation (higher = more tightly linked).")
            display_df = pairs_df.head(top_n)[["asset_a", "asset_b", "corr"]].copy()
            display_df["corr"] = display_df["corr"].map(lambda v: f"{v:.3f}")
            st.table(display_df)


            # ---- Full portfolio network visualization (all asset pairs) ----
            st.subheader("Full Portfolio Correlation Network")
            st.write("All assets shown; edge color = signed correlation, edge width/alpha = intensity. Use the slider to hide weak links.")

            # choose a visually distinct diverging colormap and center normalization
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            cmap = cm.get_cmap("seismic")

            # slider to filter by minimum absolute correlation to display an edge
            min_edge = st.sidebar.slider("Min absolute correlation to show edge", 0.0, 1.0, 0.15, 0.01)

            G_all = nx.Graph()
            for t in tickers:
                G_all.add_node(t)

            # add all unique pairs with attributes
            for _, row in pairs_df.iterrows():
                if abs(row["corr"]) >= min_edge:
                    G_all.add_edge(row["asset_a"], row["asset_b"], weight=abs(row["corr"]), corr=row["corr"])

            # node size by entanglement (mean abs corr to others)
            entanglement = corr_matrix.abs().mean(axis=1)
            # scale node sizes for plotting
            node_sizes = [300 + 1200 * entanglement.get(n, 0) for n in G_all.nodes()]

            fig_all, ax_all = plt.subplots(figsize=(12, 10))
            # layout: larger graphs benefit from larger k; scale with sqrt of nodes
            num_nodes = G_all.number_of_nodes()
            k_param = 0.8 if num_nodes <= 10 else 1.2 * (num_nodes ** 0.5) / 5
            pos_all = nx.spring_layout(G_all, seed=42, k=k_param)

            # prepare edge colors and widths
            edge_colors_all = [cmap(norm(G_all[u][v]["corr"])) for u, v in G_all.edges()]
            edge_weights_all = [max(0.5, G_all[u][v]["weight"] * 6) for u, v in G_all.edges()]
            alphas = [0.3 + 0.7 * G_all[u][v]["weight"] for u, v in G_all.edges()]

            # draw nodes
            nodes = nx.draw_networkx_nodes(G_all, pos_all, node_size=node_sizes, node_color="#88c0d0", ax=ax_all)
            nx.draw_networkx_labels(G_all, pos_all, font_size=9, ax=ax_all)

            # draw edges individually to set alpha per-edge
            for (u, v), color, w, a in zip(G_all.edges(), edge_colors_all, edge_weights_all, alphas):
                nx.draw_networkx_edges(G_all, pos_all, edgelist=[(u, v)], width=w, edge_color=[color], alpha=a, ax=ax_all)

            # add colorbar for signed correlation
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_all.colorbar(sm, ax=ax_all, fraction=0.03, pad=0.04)
            cbar.set_label('signed correlation')

            ax_all.set_title('Full Portfolio Correlation Network')
            ax_all.axis('off')
            st.pyplot(fig_all)

        # Heatmap visualization
        st.subheader("Heatmap Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

        # Portfolio health assessment
        st.subheader("Portfolio Health Check")
        
        # FIX: Correctly calculate the average of the off-diagonal correlation coefficients
        if len(tickers) > 1:
            # Use the upper triangle (excluding the diagonal) to get all unique off-diagonal correlations
            off_diagonal_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            avg_corr = off_diagonal_corr.mean()
        else:
            avg_corr = 1.0 # Single ticker has no other tickers to correlate with
        
        # Display health check based on the corrected average correlation
        if avg_corr > 0.7:
            st.error(f"⚠️ High correlation detected (avg: {avg_corr:.2f}). Your portfolio may lack diversification.")
        elif avg_corr > 0.4:
            st.warning(f"⚠️ Moderate correlation (avg: {avg_corr:.2f}). Consider adding uncorrelated assets.")
        else:
            st.success(f"✅ Good diversification (avg: {avg_corr:.2f}). Your portfolio looks healthy!")

        # --- Diversification Scoring (Avg absolute correlation -> star score) ---
        st.subheader("Diversification Score")
        if len(tickers) > 1:
            # compute average absolute off-diagonal correlation
            off_diag_abs = np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])
            avg_abs_corr = float(off_diag_abs.mean()) if off_diag_abs.size > 0 else 0.0

            def stars_for(val: float) -> str:
                if val <= 0.20:
                    return "🌟🌟🌟🌟🌟  Excellent diversification (true Dalio-style uncorrelated bets)"
                if val <= 0.40:
                    return "🌟🌟🌟🌟  Good diversification"
                if val <= 0.60:
                    return "🌟🌟🌟  Moderate, acceptable but could improve"
                if val <= 0.80:
                    return "🌟🌟  Weak diversification, assets move together"
                return "🌟  Poor, ‘fake diversification’"

            st.write(f"Average absolute pairwise correlation: {avg_abs_corr:.3f}")
            st.write(stars_for(avg_abs_corr))

            # Suggested weighting guidance removed: keep the basic diversification score only
            st.info("For simpler workflows, regime-weighted adjustments have been removed. Use the diversification score above and the suggested equal-weight principles to rebalance assets across economic exposures.")

            # --- Financial analysis via GenAI (optional) ---
            st.subheader("Run GenAI analysis on holdings")
            st.write("Optionally send your current holdings and correlation info to a GenAI model (e.g. Gemini) for scenario analysis. Provide your API key below. No requests are sent without your explicit consent.")

            st.sidebar.markdown("## GenAI analysis (optional)")
            # genai_key = st.sidebar.text_input("GenAI API Key", "", type="password")
            consent = st.checkbox("I consent to send this portfolio data to the GenAI model")

            # allow user to optionally provide current weights (comma-separated) or assume equal
            weights_input = st.text_input("Optional: current weights (comma-separated, sum not required)", "")
            if weights_input.strip():
                try:
                    weights = [float(w) for w in weights_input.split(",")]
                    if len(weights) != len(tickers):
                        st.warning("Number of weights doesn't match number of tickers — equal weights will be assumed.")
                        weights = None
                except Exception:
                    st.warning("Could not parse weights — equal weights will be assumed.")
                    weights = None
            else:
                weights = None

            if weights is None:
                weights = [1.0 / len(tickers)] * len(tickers)

            st.write("Current weights used for analysis:", {t: round(w, 4) for t, w in zip(tickers, weights)})

            # Prepare JSON-safe correlation matrix
            corr_json = corr_matrix.applymap(lambda x: float(x)).to_dict()

            # System prompt input
            st.markdown("**System prompt for the GenAI model**")
            system_prompt = st.text_area("System prompt", value=DEFAULT_PROMPT, height=160)

            if st.button("Run GenAI Analysis"):
                if not consent:
                    st.error("You must give consent before the data can be sent.")
                # elif not genai_key:
                #     st.error("Please provide your GenAI API key in the sidebar.")
                else:
                    payload = {
                        "tickers": tickers,
                        "weights": weights,
                        "correlation_matrix": corr_json,
                        "avg_abs_pairwise_correlation": float(avg_abs_corr),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }

                    prompt_text = system_prompt + "\n\nPortfolio data (JSON):\n" + json.dumps(payload, indent=2)

                    st.write("Sending prompt preview (trimmed):")
                    st.code(prompt_text[:2000])

                    try:
                        # set API key in environment for the client (non-persistent)
                        # os.environ["GEMINI_API_KEY"] = genai_key
                        try:
                            from google import genai
                        except Exception as ie:
                            st.error("GenAI client library not installed. Install the official SDK (e.g., `pip install google-genai`).")
                            raise ie

                        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                        with st.spinner("Running GenAI analysis..."):
                            response = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt_text,
                            )

                        # response object may vary; try to show text if available
                        text = getattr(response, "text", None) or (response if isinstance(response, str) else None)
                        if not text:
                            # try dict-like
                            try:
                                text = response["candidates"][0]["content"][0]["text"]
                            except Exception:
                                text = str(response)

                        st.subheader("GenAI model output")
                        if text:
                            st.markdown(text, unsafe_allow_html=False)
                        else:
                            st.info("No output received from the model.")
                    except Exception as e:
                        st.error(f"GenAI request failed: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check the tickers and your internet connection.")
