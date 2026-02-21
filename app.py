import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Simulator", layout="wide")

# --------------------------------------------------
# App state navigation helper
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "👥 Community View", "🧪 Simulator"],
    index=["🏠 Home", "👥 Community View", "🧪 Simulator"].index(st.session_state.page),
)

# Keep session_state synced
st.session_state.page = page

audience_mode = st.sidebar.radio(
    "Audience Mode",
    ["Community (Simple)", "Student/Research (Detailed)"],
    index=0
)

st.sidebar.markdown("---")

# --------------------------------------------------
# Helpers: safe local image loader
# --------------------------------------------------
def show_image_if_exists(path: str, caption: str = "", use_container_width: bool = True):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=use_container_width)
        return True
    return False

# --------------------------------------------------
# Helpers: models
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

def f_glucose_insulin(G, I, intake, s, alpha, kg, ki):
    dG = intake - s * G * I - kg * G
    dI = alpha * G - ki * I
    return dG, dI

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

def curve_summary(t, G, I):
    peak_G = float(np.max(G))
    end_G = float(G[-1])
    peak_I = float(np.max(I))
    n = len(G)
    tail = G[int(0.8*n):]
    stability = float(np.std(tail))
    return {"peak_G": peak_G, "end_G": end_G, "peak_I": peak_I, "stability": stability}

def community_guidance(summary):
    peak_G = summary["peak_G"]
    stability = summary["stability"]
    msgs = ["Educational tool only — not diagnosis, not prescriptions, not medical advice."]
    if peak_G > 180:
        msgs.append("Glucose peaks very high in this simulation. This can happen with higher intake or low sensitivity.")
    elif peak_G > 140:
        msgs.append("Glucose peaks moderately high in this simulation.")
    else:
        msgs.append("Glucose stays relatively controlled in this simulation.")
    if stability > 5:
        msgs.append("Glucose is less stable near the end (try smaller dt = 0.1).")
    else:
        msgs.append("Glucose becomes fairly stable over time.")
    tips = [
        "Choose water instead of sugary drinks.",
        "Add vegetables + fiber (beans, whole grains) to reduce spikes.",
        "Walk or move daily (even 20–30 minutes).",
        "If you have symptoms or take medication, consult a clinician before making changes.",
    ]
    return msgs, tips

# --------------------------------------------------
# CSS: nicer cards + hover animation
# --------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 18px 18px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.15));
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 16px;
    }
    .hero h1 { margin: 0; font-size: 44px; }
    .hero p { margin: 8px 0 0 0; font-size: 16px; opacity: 0.9; }

    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        margin-right: 8px;
        font-size: 13px;
        opacity: 0.95;
    }

    .card {
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        transition: transform 0.15s ease, border 0.15s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-3px);
        border: 1px solid rgba(255,255,255,0.22);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# PAGE: HOME
# ==================================================
if st.session_state.page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="pill">🩺 Diabetes learning</div>
            <div class="pill">📈 Visual simulation</div>
            <div class="pill">🧪 Euler vs RK4</div>
            <h1>Interactive Diabetes Simulator</h1>
            <p>Explore how blood sugar (glucose) and insulin may change over time — with simple explanations for everyone.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👥 Community View")
        st.write("Simple guidance, weekly plan checklist, pictures, and examples.")
        if st.button("Open Community View →", use_container_width=True):
            go_to("👥 Community View")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Simulator")
        st.write("Try the full simulator (ODE + RK4 + intervention + error curves).")
        if st.button("Open Simulator →", use_container_width=True):
            go_to("🧪 Simulator")
        st.markdown('</div>', unsafe_allow_html=True)

    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ℹ️ What is insulin?")
        st.write("Most people don’t measure insulin daily. Learn what it means.")
        with st.expander("Read a simple explanation"):
            st.write(
                """
**Glucose** is blood sugar. It rises after meals, especially carbs/sugary drinks.  
**Insulin** is a hormone that helps move glucose into cells.

Most people measure **glucose** at home, but **insulin** usually needs a **lab test** (fasting insulin, C-peptide).  
This app is **educational** — it helps you understand patterns, not diagnose disease.
                """
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Tip: If you’re here for simple guidance, start with **Community View**.")

# ==================================================
# PAGE: COMMUNITY VIEW (photos + water tracker + checklist)
# ==================================================
elif st.session_state.page == "👥 Community View":
    st.markdown(
        """
        <div class="hero">
            <div class="pill">🍽️ Food</div>
            <div class="pill">💧 Water</div>
            <div class="pill">🚶 Movement</div>
            <h1>Community View</h1>
            <p>Simple guidance, pictures, and a weekly plan checklist. (Educational — not medical advice.)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning("Educational tool only. Not diagnosis. Not prescriptions. For symptoms/medications, consult a clinician.")

    # --- Photo strip (works if you add assets)
    st.subheader("📸 Healthy habits (examples)")
    c1, c2, c3 = st.columns(3)

    with c1:
        ok = show_image_if_exists("assets/healthy_plate.jpg", "Healthy plate idea")
        if not ok:
            st.info("Add an image: `assets/healthy_plate.jpg` (healthy dish photo).")

    with c2:
        ok = show_image_if_exists("assets/water.jpg", "Water first")
        if not ok:
            st.info("Add an image: `assets/water.jpg` (water bottle/glass).")

    with c3:
        ok = show_image_if_exists("assets/walk.jpg", "Daily movement")
        if not ok:
            st.info("Add an image: `assets/walk.jpg` (walking/exercise photo).")

    st.markdown("---")

    # --- 2 liters water tracker (interactive)
    st.subheader("💧 Water goal (2 liters/day)")
    st.caption("Educational habit tracker. Adjust to see the progress bar.")
    liters = st.slider("How much water did you drink today (liters)?", 0.0, 2.5, 1.0, 0.1)
    st.progress(min(liters / 2.0, 1.0))
    if liters >= 2.0:
        st.success("Nice! You hit the 2L goal today ✅")
    else:
        st.write(f"Goal remaining: **{max(0.0, 2.0 - liters):.1f} L**")

    st.markdown("---")

    # --- Weekly plan checklist
    if "weekly" not in st.session_state:
        st.session_state.weekly = {
            "Water instead of sugary drinks (3+ days)": False,
            "Vegetables with meals (4+ days)": False,
            "20–30 min walk (4+ days)": False,
            "Whole grains instead of refined carbs (3+ days)": False,
            "Consistent sleep routine (4+ days)": False,
        }

    st.subheader("✅ Weekly Plan Checklist")
    done = 0
    for k in list(st.session_state.weekly.keys()):
        st.session_state.weekly[k] = st.checkbox(k, value=st.session_state.weekly[k])
        done += 1 if st.session_state.weekly[k] else 0
    st.progress(done / max(1, len(st.session_state.weekly)))
    st.write(f"Progress: **{done} / {len(st.session_state.weekly)}**")

    st.markdown("---")

    st.subheader("📌 If your curve looks like this… (examples)")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("### 🔺 Big glucose peak")
        st.write("Often higher intake or lower sensitivity.")
        st.write("Try in Simulator: reduce intake, increase sensitivity.")
    with e2:
        st.markdown("### 🟡 Slow return")
        st.write("Glucose takes longer to stabilize.")
        st.write("Try: smaller dt and improve sensitivity.")
    with e3:
        st.markdown("### 🟢 Stable & controlled")
        st.write("Lower peaks and steady behavior.")
        st.write("Keep habits consistent.")

    st.markdown("---")
    st.subheader("Ready to explore?")
    if st.button("Go to Simulator →", use_container_width=True):
        go_to("🧪 Simulator")

# ==================================================
# PAGE: SIMULATOR (keep your powerful model)
# ==================================================
else:
    st.markdown(
        """
        <div class="hero">
            <div class="pill">🧪 ODE</div>
            <div class="pill">📉 Error curves</div>
            <div class="pill">🧠 Intervention</div>
            <h1>Simulator</h1>
            <p>Explore models and interventions. Community mode simplifies explanations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_choice_sim = st.sidebar.selectbox(
        "Model (Simulator)",
        ["Logistic Progression", "Glucose–Insulin ODE System"]
    )

    st.sidebar.header("Simulator Controls")
    days_sim = st.sidebar.slider("Simulation Days (Simulator)", 30, 365, 180)

    if model_choice_sim == "Logistic Progression":
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

        t, D_base = simulate_logistic(days_sim, r_base, K_base, D0)
        D_int = None
        if enable:
            _, D_int = simulate_logistic(days_sim, r_int, K_int, D0)

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
            else:
                st.subheader("Simple meaning")
                st.write("Higher r → faster progression. Intervention slows it.")
                st.info("Educational only.")

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
        s = st.sidebar.slider("Insulin sensitivity", 0.0001, 0.01, 0.0020)
        alpha = st.sidebar.slider("Insulin response strength", 0.01, 1.0, 0.20)
        kg = st.sidebar.slider("Glucose natural clearance", 0.01, 1.0, 0.10)
        ki = st.sidebar.slider("Insulin clearance", 0.01, 1.0, 0.10)

        st.sidebar.subheader("Intervention Scenario (Optional)")
        enable_intervention = st.sidebar.checkbox("Enable intervention scenario", value=True)
        adherence_int = st.sidebar.slider("Adherence (0-1)", 0.0, 1.0, 0.7)
        diet_reduction = st.sidebar.slider("Diet change (reduces intake)", 0.0, 1.0, 0.4)
        activity_increase = st.sidebar.slider("Activity change (improves sensitivity)", 0.0, 1.0, 0.4)

        intake_int = intake * (1 - adherence_int * diet_reduction)
        s_int = min(s * (1 + adherence_int * activity_increase), 0.02)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots()

            t_rk, G_rk, I_rk = simulate_glucose_insulin_rk4(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)

            if method == "Euler":
                t_e, G_e, I_e = simulate_glucose_insulin_euler(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)
                ax.plot(t_e, G_e, label="Glucose (Euler)")
                ax.plot(t_e, I_e, label="Insulin (Euler)")
            elif method == "RK4":
                ax.plot(t_rk, G_rk, label="Glucose (RK4)")
                ax.plot(t_rk, I_rk, label="Insulin (RK4)")
            else:
                t_e, G_e, I_e = simulate_glucose_insulin_euler(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)
                ax.plot(t_e, G_e, label="Glucose (Euler)")
                ax.plot(t_rk, G_rk, linestyle="--", label="Glucose (RK4)")
                ax.plot(t_e, I_e, label="Insulin (Euler)")
                ax.plot(t_rk, I_rk, linestyle="--", label="Insulin (RK4)")

                err_G = G_e - G_rk
                err_I = I_e - I_rk
                st.caption("Error curves are shown below the main plot.")

            if enable_intervention:
                _, G_int, I_int = simulate_glucose_insulin_rk4(days_sim, dt, G0, I0, intake_int, s_int, alpha, kg, ki)
                ax.plot(t_rk, G_int, linestyle=":", label="Glucose (Intervention RK4)")
                ax.plot(t_rk, I_int, linestyle=":", label="Insulin (Intervention RK4)")

            ax.set_title("Glucose–Insulin Dynamics")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Level")
            ax.legend()
            st.pyplot(fig)

            if method == "Compare Euler vs RK4 (with error curves)":
                fig2, ax2 = plt.subplots()
                ax2.plot(t_e, err_G, label="Error Glucose (Euler − RK4)")
                ax2.plot(t_e, err_I, label="Error Insulin (Euler − RK4)")
                ax2.axhline(0.0)
                ax2.set_title("Numerical Error Curves")
                ax2.set_xlabel("Time (days)")
                ax2.set_ylabel("Error")
                ax2.legend()
                st.pyplot(fig2)

        with col2:
            if audience_mode == "Student/Research (Detailed)":
                st.subheader("Equations (Research)")
                st.latex(r"\frac{dG}{dt} = \text{intake} - sGI - k_g G")
                st.latex(r"\frac{dI}{dt} = \alpha G - k_i I")
            else:
                st.subheader("Simple meaning")
                st.write("Food/intake raises glucose. Insulin helps reduce glucose.")
                st.warning("Educational only — not medical advice.")
