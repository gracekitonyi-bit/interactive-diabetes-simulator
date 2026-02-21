import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Simulator", layout="wide")

st.title("🩺 Interactive Diabetes Progression Simulator")
st.caption("Logistic progression + Glucose–Insulin ODE model (Euler vs RK4)")

# --------------------------------------------------
# Model selector
# --------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Progression", "Glucose–Insulin ODE System"]
)

# --------------------------------------------------
# Logistic model (Euler discretization)
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
# Glucose–Insulin ODE RHS
# dG/dt = intake - sGI - kgG
# dI/dt = alpha G - kiI
# --------------------------------------------------
def f_glucose_insulin(G, I, intake, s, alpha, kg, ki):
    dG = intake - s * G * I - kg * G
    dI = alpha * G - ki * I
    return dG, dI

# --------------------------------------------------
# Euler solver
# --------------------------------------------------
def simulate_glucose_insulin_euler(days, dt, G0, I0, intake, s, alpha, kg, ki):
    t = np.arange(0, days + dt, dt)
    G = np.zeros_like(t, dtype=float)
    I = np.zeros_like(t, dtype=float)
    G[0], I[0] = G0, I0

    for i in range(1, len(t)):
        dG, dI = f_glucose_insulin(G[i-1], I[i-1], intake, s, alpha, kg, ki)
        G[i] = G[i-1] + dt * dG
        I[i] = I[i-1] + dt * dI
        G[i] = max(G[i], 0.0)
        I[i] = max(I[i], 0.0)

    return t, G, I

# --------------------------------------------------
# RK4 solver
# --------------------------------------------------
def simulate_glucose_insulin_rk4(days, dt, G0, I0, intake, s, alpha, kg, ki):
    t = np.arange(0, days + dt, dt)
    G = np.zeros_like(t, dtype=float)
    I = np.zeros_like(t, dtype=float)
    G[0], I[0] = G0, I0

    for i in range(1, len(t)):
        G_prev, I_prev = G[i-1], I[i-1]

        k1G, k1I = f_glucose_insulin(G_prev, I_prev, intake, s, alpha, kg, ki)
        k2G, k2I = f_glucose_insulin(G_prev + 0.5*dt*k1G, I_prev + 0.5*dt*k1I, intake, s, alpha, kg, ki)
        k3G, k3I = f_glucose_insulin(G_prev + 0.5*dt*k2G, I_prev + 0.5*dt*k2I, intake, s, alpha, kg, ki)
        k4G, k4I = f_glucose_insulin(G_prev + dt*k3G, I_prev + dt*k3I, intake, s, alpha, kg, ki)

        G[i] = G_prev + (dt/6.0) * (k1G + 2*k2G + 2*k3G + k4G)
        I[i] = I_prev + (dt/6.0) * (k1I + 2*k2I + 2*k3I + k4I)

        G[i] = max(G[i], 0.0)
        I[i] = max(I[i], 0.0)

    return t, G, I

# --------------------------------------------------
# Shared simulation settings
# --------------------------------------------------
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Days", 30, 365, 180)

# ==================================================
# MODEL 1: Logistic
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

    r_int = r_base * (1 - adherence * treatment_strength)
    K_int = K_base * (1 - 0.2 * adherence * treatment_strength)

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
        st.write("• Intervention reduces r (slows progression)")

# ==================================================
# MODEL 2: Glucose–Insulin ODE (Euler vs RK4)
# ==================================================
else:
    st.sidebar.subheader("ODE Parameters")
    dt = st.sidebar.selectbox("Time step (dt)", [0.1, 0.25, 0.5, 1.0], index=2)

    method = st.sidebar.radio(
        "Numerical Method",
        ["Euler", "RK4", "Compare Euler vs RK4"],
        index=2
    )

    G0 = st.sidebar.slider("Initial Glucose G0", 50.0, 250.0, 120.0)
    I0 = st.sidebar.slider("Initial Insulin I0", 1.0, 80.0, 15.0)

    intake = st.sidebar.slider("Glucose intake", 0.0, 10.0, 3.0)
    s = st.sidebar.slider("Insulin sensitivity (s)", 0.0001, 0.01, 0.0020)
    alpha = st.sidebar.slider("Pancreatic response (alpha)", 0.01, 1.0, 0.20)
    kg = st.sidebar.slider("Glucose decay (kg)", 0.01, 1.0, 0.10)
    ki = st.sidebar.slider("Insulin clearance (ki)", 0.01, 1.0, 0.10)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()

        if method == "Euler":
            t, G, I = simulate_glucose_insulin_euler(days, dt, G0, I0, intake, s, alpha, kg, ki)
            ax.plot(t, G, label="Glucose (Euler)")
            ax.plot(t, I, label="Insulin (Euler)")
            ax.set_title("Glucose–Insulin Dynamics (Euler)")

        elif method == "RK4":
            t, G, I = simulate_glucose_insulin_rk4(days, dt, G0, I0, intake, s, alpha, kg, ki)
            ax.plot(t, G, label="Glucose (RK4)")
            ax.plot(t, I, label="Insulin (RK4)")
            ax.set_title("Glucose–Insulin Dynamics (RK4)")

        else:
            t, G_e, I_e = simulate_glucose_insulin_euler(days, dt, G0, I0, intake, s, alpha, kg, ki)
            _, G_r, I_r = simulate_glucose_insulin_rk4(days, dt, G0, I0, intake, s, alpha, kg, ki)

            ax.plot(t, G_e, label="Glucose (Euler)")
            ax.plot(t, G_r, linestyle="--", label="Glucose (RK4)")
            ax.plot(t, I_e, label="Insulin (Euler)")
            ax.plot(t, I_r, linestyle="--", label="Insulin (RK4)")
            ax.set_title("Euler vs RK4 Comparison")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Level")
        ax.legend()
        st.pyplot(fig)

        if method == "Compare Euler vs RK4":
            # Simple numerical difference metric
            diff_G = np.max(np.abs(G_e - G_r))
            diff_I = np.max(np.abs(I_e - I_r))
            st.info(f"Max absolute difference: Glucose = {diff_G:.3f}, Insulin = {diff_I:.3f}")

    with col2:
        st.subheader("Model Interpretation")

        st.latex(r"\frac{dG}{dt} = \text{intake} - sGI - k_g G")
        st.latex(r"\frac{dI}{dt} = \alpha G - k_i I")

        st.markdown("### Why Euler vs RK4 matters")
        st.markdown(r"""
- **Euler** is simple but can be inaccurate for larger $dt$  
- **RK4** is much more accurate for the same $dt$  
- Comparing them shows **numerical error**
        """)

        st.markdown("### What you should observe")
        st.markdown(r"""
- For **small dt (0.1)** Euler and RK4 look very similar  
- For **large dt (1.0)** Euler may deviate more  
        """)
