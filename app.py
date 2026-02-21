import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Simulator", layout="wide")

st.title("🩺 Interactive Diabetes Progression Simulator")
st.caption("Logistic progression + intervention comparison")

# -------------------------
# Helper: simulate logistic (Euler discretization)
# -------------------------
def simulate_logistic(days, r, K, D0):
    t = np.arange(0, days + 1)
    D = np.zeros_like(t, dtype=float)
    D[0] = D0

    for i in range(1, len(t)):
        D[i] = D[i-1] + r * D[i-1] * (1 - D[i-1] / K)

        # keep it non-negative
        if D[i] < 0:
            D[i] = 0.0

    return t, D

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Days", 30, 365, 180)

st.sidebar.subheader("Baseline (No Intervention)")
r_base = st.sidebar.slider("Progression rate r (baseline)", 0.01, 0.20, 0.05)
K_base = st.sidebar.slider("Carrying capacity K (baseline)", 0.50, 2.00, 1.00)
D0 = st.sidebar.slider("Initial disease level D0", 0.01, 0.50, 0.10)

st.sidebar.subheader("Intervention")
enable = st.sidebar.checkbox("Enable intervention", value=True)

adherence = st.sidebar.slider("Adherence (0 = none, 1 = perfect)", 0.0, 1.0, 0.7)
treatment_strength = st.sidebar.slider("Treatment strength (reduces progression)", 0.0, 1.0, 0.5)

# Intervention effect: reduce r, optionally reduce K a little
# (Interpretation: lifestyle/medication slows progression; sustained changes reduce long-term burden.)
r_int = r_base * (1 - adherence * treatment_strength)
K_int = K_base * (1 - 0.2 * adherence * treatment_strength)  # small reduction in max burden

# -------------------------
# Run simulations
# -------------------------
t, D_base = simulate_logistic(days, r_base, K_base, D0)

if enable:
    _, D_int = simulate_logistic(days, r_int, K_int, D0)
else:
    D_int = None

# -------------------------
# Layout: plot + explanation
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots()
    ax.plot(t, D_base, label="Baseline (no intervention)")

    if D_int is not None:
        ax.plot(t, D_int, label="With intervention")

    ax.set_xlabel("Days")
    ax.set_ylabel("Disease level (normalized)")
    ax.set_title("Logistic Progression Model")
    ax.legend()

    st.pyplot(fig)

with col2:
    st.subheader("Interpretation")
    st.write(
        """
This is a simple *progression* model:

- **r** controls how fast disease burden increases.
- **K** is the long-run maximum disease burden.
- **Intervention** reduces r (slows progression) depending on adherence.
        """
    )

    st.markdown("### Current values")
    st.write(f"Baseline r = {r_base:.3f}, K = {K_base:.2f}")
    if enable:
        st.write(f"Intervention r = {r_int:.3f}, K = {K_int:.2f}")
    else:
        st.write("Intervention is OFF")

    st.markdown("### Quick takeaway")
    if enable and D_int is not None:
        st.write("If the intervention curve stays lower, your strategy is reducing disease burden over time.")
    else:
        st.write("Turn on intervention to compare strategies.")
