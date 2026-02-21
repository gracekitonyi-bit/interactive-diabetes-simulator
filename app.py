import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Simulator", layout="wide")

st.title("🩺 Interactive Diabetes Progression Simulator")
st.caption("Educational simulator: Logistic progression + Glucose–Insulin ODE (Euler vs RK4 + intervention)")

# --------------------------------------------------
# Sidebar: audience + model selector
# --------------------------------------------------
audience_mode = st.sidebar.radio(
    "Audience Mode",
    ["Community (Simple)", "Student/Research (Detailed)"],
    index=0
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Progression", "Glucose–Insulin ODE System"]
)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Days", 30, 365, 180)

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
# dG/dt = intake - s*G*I - kg*G
# dI/dt = alpha*G - ki*I
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
        G[i] = max(G[i-1] + dt * dG, 0.0)
        I[i] = max(I[i-1] + dt * dI, 0.0)

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
        Gp, Ip = G[i-1], I[i-1]

        k1G, k1I = f_glucose_insulin(Gp, Ip, intake, s, alpha, kg, ki)
        k2G, k2I = f_glucose_insulin(Gp + 0.5*dt*k1G, Ip + 0.5*dt*k1I, intake, s, alpha, kg, ki)
        k3G, k3I = f_glucose_insulin(Gp + 0.5*dt*k2G, Ip + 0.5*dt*k2I, intake, s, alpha, kg, ki)
        k4G, k4I = f_glucose_insulin(Gp + dt*k3G, Ip + dt*k3I, intake, s, alpha, kg, ki)

        G[i] = max(Gp + (dt/6.0) * (k1G + 2*k2G + 2*k3G + k4G), 0.0)
        I[i] = max(Ip + (dt/6.0) * (k1I + 2*k2I + 2*k3I + k4I), 0.0)

    return t, G, I

# --------------------------------------------------
# Community helper summaries (educational, non-diagnostic)
# --------------------------------------------------
def curve_summary(t, G, I):
    peak_G = float(np.max(G))
    end_G = float(G[-1])
    peak_I = float(np.max(I))
    end_I = float(I[-1])

    n = len(G)
    tail = G[int(0.8*n):]
    stability = float(np.std(tail))  # smaller = more stable

    return {"peak_G": peak_G, "end_G": end_G, "peak_I": peak_I, "end_I": end_I, "stability": stability}

def community_guidance(summary):
    peak_G = summary["peak_G"]
    stability = summary["stability"]

    msgs = []
    msgs.append("Educational tool only — not diagnosis, not medical advice, not prescriptions.")

    if peak_G > 180:
        msgs.append("In this simulation, glucose peaks very high. This can happen with higher intake or low sensitivity.")
    elif peak_G > 140:
        msgs.append("In this simulation, glucose peaks moderately high. Reducing intake or increasing sensitivity may lower the peak.")
    else:
        msgs.append("In this simulation, glucose stays relatively controlled (lower peak).")

    if stability > 5:
        msgs.append("Glucose is less stable near the end. Try smaller dt (0.1) and explore which parameter drives instability.")
    else:
        msgs.append("Glucose becomes fairly stable as the simulation progresses (approaches equilibrium).")

    tips = [
        "General tips: reduce sugary drinks, choose high-fiber carbs (beans, vegetables, whole grains).",
        "Pair carbs with protein/healthy fats to reduce spikes.",
        "Regular movement (walking/exercise) can improve insulin sensitivity over time.",
        "If you have symptoms or you’re on medication, talk to a health professional before making changes."
    ]
    return msgs, tips

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

    r_int = r_base * (1 - adherence * treatment_strength)
    K_int = K_base * (1 - 0.2 * adherence * treatment_strength)

    t, D_base = simulate_logistic(days, r_base, K_base, D0)
    D_int = None
    if enable:
        _, D_int = simulate_logistic(days, r_int, K_int, D0)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()
        ax.plot(t, D_base, label="Baseline")
        if D_int is not None:
            ax.plot(t, D_int, label="With intervention")
        ax.set_xlabel("Days")
        ax.set_ylabel("Progression level (normalized)")
        ax.set_title("Logistic Progression Model")
        ax.legend()
        st.pyplot(fig)

    with col2:
        if audience_mode == "Student/Research (Detailed)":
            st.subheader("Interpretation (Research)")
            st.latex(r"\frac{dD}{dt} = rD\left(1 - \frac{D}{K}\right)")
            st.write("• r controls progression speed")
            st.write("• K is the long-run maximum")
            st.write("• Intervention reduces r (slows progression)")
        else:
            st.subheader("What this means (Simple)")
            st.write("This curve is a simple way to show ‘progression’ over time.")
            st.write("• Higher r → progression rises faster.")
            st.write("• Intervention reduces r → progression rises more slowly.")
            st.info("Educational tool only — not medical advice.")

# ==================================================
# MODEL 2: Glucose–Insulin ODE System
# ==================================================
else:
    st.sidebar.subheader("ODE Parameters")

    dt = st.sidebar.selectbox("Time step (dt)", [0.1, 0.25, 0.5, 1.0], index=2)

    method = st.sidebar.radio(
        "Numerical Method",
        ["Euler", "RK4", "Compare Euler vs RK4 (with error curves)"],
        index=2
    )

    st.sidebar.subheader("Initial Conditions")
    G0 = st.sidebar.slider("Initial Glucose G0", 50.0, 250.0, 120.0)
    I0 = st.sidebar.slider("Initial Insulin I0", 1.0, 80.0, 15.0)

    st.sidebar.subheader("Baseline Parameters")
    intake = st.sidebar.slider("Glucose intake (diet signal)", 0.0, 10.0, 3.0)
    s = st.sidebar.slider("Insulin sensitivity (higher is better)", 0.0001, 0.01, 0.0020)
    alpha = st.sidebar.slider("Insulin response strength", 0.01, 1.0, 0.20)
    kg = st.sidebar.slider("Glucose natural clearance", 0.01, 1.0, 0.10)
    ki = st.sidebar.slider("Insulin clearance", 0.01, 1.0, 0.10)

    # -------------------------
    # Intervention scenario (educational)
    # -------------------------
    st.sidebar.subheader("Intervention Scenario (Optional)")
    enable_intervention = st.sidebar.checkbox("Enable intervention scenario", value=True)
    adherence_int = st.sidebar.slider("Adherence to intervention (0-1)", 0.0, 1.0, 0.7)

    # Diet reduces intake; Activity improves sensitivity. Both are scaled by adherence.
    diet_reduction = st.sidebar.slider("Diet change (reduces intake)", 0.0, 1.0, 0.4)
    activity_increase = st.sidebar.slider("Activity change (improves sensitivity)", 0.0, 1.0, 0.4)

    intake_int = intake * (1 - adherence_int * diet_reduction)
    s_int = s * (1 + adherence_int * activity_increase)

    # Keep in safe range
    s_int = min(s_int, 0.02)

    col1, col2 = st.columns([2, 1])

    # -------------------------
    # Plot main trajectories
    # -------------------------
    with col1:
        fig, ax = plt.subplots()

        # Baseline RK4 always available (best reference)
        t_rk, G_rk, I_rk = simulate_glucose_insulin_rk4(days, dt, G0, I0, intake, s, alpha, kg, ki)

        if method == "Euler":
            t_e, G_e, I_e = simulate_glucose_insulin_euler(days, dt, G0, I0, intake, s, alpha, kg, ki)
            ax.plot(t_e, G_e, label="Glucose (Euler)")
            ax.plot(t_e, I_e, label="Insulin (Euler)")
            ax.set_title("Glucose–Insulin Dynamics (Euler)")

        elif method == "RK4":
            ax.plot(t_rk, G_rk, label="Glucose (RK4)")
            ax.plot(t_rk, I_rk, label="Insulin (RK4)")
            ax.set_title("Glucose–Insulin Dynamics (RK4)")

        else:
            # Compare Euler vs RK4
            t_e, G_e, I_e = simulate_glucose_insulin_euler(days, dt, G0, I0, intake, s, alpha, kg, ki)
            ax.plot(t_e, G_e, label="Glucose (Euler)")
            ax.plot(t_rk, G_rk, linestyle="--", label="Glucose (RK4)")
            ax.plot(t_e, I_e, label="Insulin (Euler)")
            ax.plot(t_rk, I_rk, linestyle="--", label="Insulin (RK4)")
            ax.set_title("Euler vs RK4 Comparison")

            diff_G = float(np.max(np.abs(G_e - G_rk)))
            diff_I = float(np.max(np.abs(I_e - I_rk)))
            st.info(f"Numerical error (max abs): Glucose={diff_G:.3f}, Insulin={diff_I:.3f}")

        # Add intervention overlay (using RK4 for clarity)
        if enable_intervention:
            _, G_int, I_int = simulate_glucose_insulin_rk4(days, dt, G0, I0, intake_int, s_int, alpha, kg, ki)
            ax.plot(t_rk, G_int, linestyle=":", label="Glucose (Intervention RK4)")
            ax.plot(t_rk, I_int, linestyle=":", label="Insulin (Intervention RK4)")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Level")
        ax.legend()
        st.pyplot(fig)

        # -------------------------
        # Error curves plot (Euler - RK4)
        # -------------------------
        if method == "Compare Euler vs RK4 (with error curves)":
            err_G = G_e - G_rk
            err_I = I_e - I_rk

            fig2, ax2 = plt.subplots()
            ax2.plot(t_e, err_G, label="Error in Glucose: Euler − RK4")
            ax2.plot(t_e, err_I, label="Error in Insulin: Euler − RK4")
            ax2.axhline(0.0)
            ax2.set_xlabel("Time (days)")
            ax2.set_ylabel("Error")
            ax2.set_title("Numerical Error Curves (Euler − RK4)")
            ax2.legend()
            st.pyplot(fig2)

    # -------------------------
    # Interpretation panel
    # -------------------------
    with col2:
        if audience_mode == "Student/Research (Detailed)":
            st.subheader("Interpretation (Research)")
            st.latex(r"\frac{dG}{dt} = \text{intake} - sGI - k_g G")
            st.latex(r"\frac{dI}{dt} = \alpha G - k_i I")
            st.markdown("**RK4** is a higher-accuracy reference. **Euler** may deviate more when dt is large.")
            st.markdown("The error curves show where Euler accumulates numerical error.")

            if enable_intervention:
                st.markdown("---")
                st.markdown("### Intervention mapping (educational)")
                st.write(f"Baseline intake = {intake:.2f} → Intervention intake = {intake_int:.2f}")
                st.write(f"Baseline sensitivity s = {s:.4f} → Intervention sensitivity s = {s_int:.4f}")

        else:
            st.subheader("What this means (Simple)")
            st.write("This simulation shows how blood sugar (glucose) and insulin can change over time.")
            st.write("• Eating/‘intake’ pushes glucose up.")
            st.write("• Insulin helps bring glucose down.")
            st.write("• ‘Sensitivity’ means how well insulin helps reduce glucose.")

            st.warning("Educational tool only — not diagnosis, not prescriptions, not medical advice.")

            summary = curve_summary(t_rk, G_rk, I_rk)
            msgs, tips = community_guidance(summary)

            st.markdown("### Quick summary (from this simulation)")
            st.write(f"Peak glucose: **{summary['peak_G']:.1f}**")
            st.write(f"End glucose: **{summary['end_G']:.1f}**")
            st.write(f"Peak insulin: **{summary['peak_I']:.1f}**")

            if enable_intervention:
                _, G_int_s, I_int_s = simulate_glucose_insulin_rk4(days, dt, G0, I0, intake_int, s_int, alpha, kg, ki)
                summary_int = curve_summary(t_rk, G_int_s, I_int_s)
                st.markdown("### Intervention comparison (educational)")
                st.write(f"Peak glucose (baseline) = **{summary['peak_G']:.1f}**")
                st.write(f"Peak glucose (intervention) = **{summary_int['peak_G']:.1f}**")

                if summary_int["peak_G"] < summary["peak_G"]:
                    st.success("In this simulation, the intervention reduces the glucose peak.")
                else:
                    st.info("In this simulation, the intervention does not reduce the glucose peak much. Try adjusting diet/activity sliders.")

            with st.expander("General lifestyle tips (not medical advice)"):
                for tip in tips:
                    st.write("• " + tip)

            with st.expander("How to know your real insulin level"):
                st.write("• Most people need a lab test to measure insulin (e.g., fasting insulin).")
                st.write("• Doctors may also check C-peptide, fasting glucose, and HbA1c.")
                st.write("• If you are concerned, speak with a clinician for proper testing.")
