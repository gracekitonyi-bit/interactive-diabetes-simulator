import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Simulator", layout="wide")

st.title("🩺 Interactive Diabetes Progression Simulator")
st.caption("Logistic progression + Glucose–Insulin ODE model")

# --------------------------------------------------
# Model selector
# --------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Progression", "Glucose–Insulin ODE System"]
)

# --------------------------------------------------
# Logistic model (Euler)
# --------------------------------------------------
def simulate_logistic(days, r, K, D0):
    t = np.arange(0, days + 1)
    D = np.zeros_like(t, dtype=float)
    D[0] = D0

    for i in range(1, len(t)):
        D[i] = D[i-1] + r * D[i-1] * (1 - D[i-1] / K)
        if D[i] < 0:
            D[i] = 0.0

    return t, D


# --------------------------------------------------
# Glucose–Insulin ODE system (Euler method)
# dG/dt = intake - sGI - kgG
# dI/dt = alpha G - kiI
# --------------------------------------------------
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

        if G[i] < 0:
            G[i] = 0.0
        if I[i] < 0:
            I[i] = 0.0

    return t, G, I


# --------------------------------------------------
# Shared simulation settings
# --------------------------------------------------
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Days", 30, 365, 180)

# ==================================================
# MODEL 1: Logistic Progression
# ==================================================
if model_choice == "Logistic Progression":

    st.sidebar.subheader("Baseline (No Intervention)")
    r_base = st.sidebar.slider("Progression rate r", 0.01, 0.20, 0.05)
    K_base = st.sidebar.slider("Carrying capacity K", 0.50, 2.00, 1.00)
    D0 = st.sidebar.slider("Initial disease level D0", 0.01, 0.50, 0.10)

    st.sidebar.subheader("Intervention")
    enable = st.sidebar.checkbox("Enable intervention", value=True)
    adherence = st.sidebar.slider("Adherence (0-1)", 0.0, 1.0, 0.7)
    treatment_strength = st.sidebar.slider("Treatment strength", 0.0, 1.0, 0.5)

    # Apply intervention effect
    r_int = r_base * (1 - adherence * treatment_strength)
    K_int = K_base * (1 - 0.2 * adherence * treatment_strength)

    # Run simulations
    t, D_base = simulate_logistic(days, r_base, K_base, D0)

    if enable:
        _, D_int = simulate_logistic(days, r_int, K_int, D0)
    else:
        D_int = None

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()
        ax.plot(t, D_base, label="Baseline")
        if D_int is not None:
            ax.plot(t, D_int, label="With intervention")
        ax.set_xlabel("Days")
        ax.set_ylabel("Disease level")
        ax.set_title("Logistic Progression Model")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Model Interpretation")
        st.latex(r"\frac{dD}{dt} = rD\left(1 - \frac{D}{K}\right)")
        st.write("• r controls growth speed")
        st.write("• K is long-run maximum")
        st.write("• Intervention reduces r")

# ==================================================
# MODEL 2: Glucose–Insulin ODE
# ==================================================
else:

    st.sidebar.subheader("ODE Parameters (Euler)")
    dt = st.sidebar.selectbox("Time step (dt)", [0.1, 0.5, 1.0], index=1)

    G0 = st.sidebar.slider("Initial Glucose G0", 50.0, 250.0, 120.0)
    I0 = st.sidebar.slider("Initial Insulin I0", 1.0, 80.0, 15.0)

    intake = st.sidebar.slider("Glucose intake", 0.0, 10.0, 3.0)
    s = st.sidebar.slider("Insulin sensitivity (s)", 0.0001, 0.01, 0.0020)
    alpha = st.sidebar.slider("Pancreatic response (alpha)", 0.01, 1.0, 0.20)
    kg = st.sidebar.slider("Glucose decay (kg)", 0.01, 1.0, 0.10)
    ki = st.sidebar.slider("Insulin clearance (ki)", 0.01, 1.0, 0.10)

    # Run simulation
    t, G, I = simulate_glucose_insulin(days, dt, G0, I0, intake, s, alpha, kg, ki)

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
        st.subheader("Model Interpretation")

        st.latex(r"\frac{dG}{dt} = \text{intake} - sGI - k_g G")
        st.latex(r"\frac{dI}{dt} = \alpha G - k_i I")

        st.markdown("### State Variables")
        st.markdown(r"- $G(t)$ = glucose level")
        st.markdown(r"- $I(t)$ = insulin level")

        st.markdown("### Interpretation")
        st.markdown(r"""
- High $G$ increases insulin via $\alpha G$
- Insulin reduces glucose via $sGI$
- System converges to equilibrium
        """)

        st.markdown("### Next Step")
        st.write("Next upgrade: RK4 method + Euler vs RK4 comparison.")
