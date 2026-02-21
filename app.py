import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Simulator", layout="wide")

st.title("🩺 Interactive Diabetes Progression Simulator")
st.caption("Logistic progression + intervention comparison | Glucose–Insulin ODE (Euler)")

# -------------------------
# Model selector
# -------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Progression", "Glucose–Insulin ODE System"]
)

# -------------------------
# Helper 1: Logistic progression (Euler discretization)
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
# Helper 2: Glucose–Insulin ODE system (Euler method)
# dG/dt = intake - s*G*I - kg*G
# dI/dt = alpha*G - ki*I
# -------------------------
def simulate_glucose_insulin(days, dt, G0, I0, intake, s, alpha, kg, ki):
    t = np.arange(0, days + dt, dt)

    G = np.zeros_like(t, dtype=float)
    I = np.zeros_like(t, dtype=float)
    G[0] = G0
    I[0] = I0

    for i in range(1, len(t)):
        dG = intake - s * G[i-1] * I[i-1] - kg * G[i-1]
        dI = alpha * G[i-1] - ki * I[i-1]

        G[i] = G[i-1] + dt * dG
        I[i] = I[i-1] + dt * dI

        # avoid negative values
        if G[i] < 0:
            G[i] = 0.0
        if I[i] < 0:
            I[i] = 0.0

    return t, G, I

# =========================
# SIDEBAR: shared settings
# =========================
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Days", 30, 365, 180)

# =========================
# MODEL 1: Logistic
# =========================
if model_choice == "Logistic Progression":

    st.sidebar.subheader("Baseline (No Intervention)")
    r_base = st.sidebar.slider("Progression rate r (baseline)", 0.01, 0.20, 0.05)
    K_base = st.sidebar.slider("Carrying capacity K (baseline)", 0.50, 2.00, 1.00)
    D0 = st.sidebar.slider("Initial disease level D0", 0.01, 0.50, 0.10)

    st.sidebar.subheader("Intervention")
    enable = st.sidebar.checkbox("Enable intervention", value=True)
    adherence = st.sidebar.slider("Adherence (0 = none, 1 = perfect)", 0.0, 1.0, 0.7)
    treatment_strength = st.sidebar.slider("Treatment strength (reduces progression)", 0.0, 1.0, 0.5)

    # Intervention effect
    r_int = r_base * (1 - adherence * treatment_strength)
    K_int = K_base * (1 - 0.2 * adherence * treatment_strength)

    # Run simulations
    t, D_base = simulate_logistic(days, r_base, K_base, D0)
    if enable:
        _, D_int = simulate_logistic(days, r_int, K_int, D0)
    else:
        D_int = None

    # Layout
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

# =========================
# MODEL 2: Glucose–Insulin ODE
# =========================
else:
    st.sidebar.subheader("ODE Parameters (Euler)")

    dt = st.sidebar.selectbox("Time step (dt)", [0.1, 0.5, 1.0], index=1)

    G0 = st.sidebar.slider("Initial Glucose (G0)", 50.0, 250.0, 120.0)
    I0 = st.sidebar.slider("Initial Insulin (I0)", 1.0, 80.0, 15.0)

    intake = st.sidebar.slider("Glucose intake rate", 0.0, 10.0, 3.0)
    s = st.sidebar.slider("Insulin sensitivity (s)", 0.0001, 0.01, 0.0020)
    alpha = st.sidebar.slider("Pancreatic response (alpha)", 0.01, 1.0, 0.20)
    kg = st.sidebar.slider("Glucose decay (kg)", 0.01, 1.0, 0.10)
    ki = st.sidebar.slider("Insulin clearance (ki)", 0.01, 1.0, 0.10)

    # Run simulation
    t, G, I = simulate_glucose_insulin(days, dt, G0, I0, intake, s, alpha, kg, ki)

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()
        ax.plot(t, G, label="Glucose G(t)")
        ax.plot(t, I, label="Insulin I(t)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Level")
        ax.set_title("Glucose–Insulin Dynamics (Euler Method)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Interpretation")
        st.write(
            """
This is a **2-state ODE system**:

- **G(t)** = glucose level
- **I(t)** = insulin level

Equations:

- dG/dt = intake − s·G·I − kg·G  
- dI/dt = alpha·G − ki·I
            """
        )

        st.markdown("### What parameters mean")
        st.write("- **intake** increases glucose (diet).")
        st.write("- **s** is insulin sensitivity (higher s reduces glucose faster).")
        st.write("- **alpha** controls how strongly insulin responds to glucose.")
        st.write("- **kg** is natural glucose clearance.")
        st.write("- **ki** is insulin clearance.")

        st.markdown("### Next upgrade")
        st.write("We will add **RK4** and a **Euler vs RK4** comparison next.")
