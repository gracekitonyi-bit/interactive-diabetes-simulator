import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Simulator", layout="wide")

# --------------------------------------------------
# Navigation
# --------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "👥 Community View", "🧪 Simulator"], index=0)

audience_mode = st.sidebar.radio(
    "Audience Mode",
    ["Community (Simple)", "Student/Research (Detailed)"],
    index=0
)

st.sidebar.markdown("---")

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

# --------------------------------------------------
# Helpers: community summaries (educational)
# --------------------------------------------------
def curve_summary(t, G, I):
    peak_G = float(np.max(G))
    end_G = float(G[-1])
    peak_I = float(np.max(I))
    end_I = float(I[-1])
    n = len(G)
    tail = G[int(0.8*n):]
    stability = float(np.std(tail))
    return {"peak_G": peak_G, "end_G": end_G, "peak_I": peak_I, "end_I": end_I, "stability": stability}

def community_guidance(summary):
    peak_G = summary["peak_G"]
    stability = summary["stability"]
    msgs = []
    msgs.append("Educational tool only — not diagnosis, not prescriptions, not medical advice.")
    if peak_G > 180:
        msgs.append("In this simulation, glucose peaks very high (possible high intake or low sensitivity).")
    elif peak_G > 140:
        msgs.append("In this simulation, glucose peaks moderately high.")
    else:
        msgs.append("In this simulation, glucose stays relatively controlled.")
    if stability > 5:
        msgs.append("Glucose is less stable near the end (try smaller dt = 0.1, or adjust parameters).")
    else:
        msgs.append("Glucose becomes fairly stable over time (approaches equilibrium).")
    tips = [
        "Reduce sugary drinks; choose water or unsweetened tea.",
        "Prefer high-fiber carbs (beans, vegetables, whole grains).",
        "Add protein/healthy fats with carbs to reduce spikes.",
        "Regular walking/exercise can improve insulin sensitivity over time.",
        "If you have symptoms or take medication, consult a clinician before changing anything."
    ]
    return msgs, tips

# --------------------------------------------------
# STYLE: big hero section (simple CSS)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 18px 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.15));
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 16px;
    }
    .hero h1 { margin: 0; font-size: 42px; }
    .hero p { margin: 8px 0 0 0; font-size: 16px; opacity: 0.9; }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        margin-right: 8px;
        font-size: 13px;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# PAGE 1: HOME (catchy first sight)
# ==================================================
if page == "🏠 Home":
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
        st.subheader("👥 Community View")
        st.write("Simple guidance, weekly plan checklist, and examples.")
        if st.button("Open Community View →"):
            st.session_state["nav_to"] = "community"

    with colB:
        st.subheader("🧪 Simulator")
        st.write("Try the full simulator (Logistic + ODE + RK4 + intervention).")
        if st.button("Open Simulator →"):
            st.session_state["nav_to"] = "sim"

    with colC:
        st.subheader("ℹ️ What is insulin?")
        st.write("Most people don’t measure insulin daily. Learn what it means.")
        with st.expander("Read a simple explanation"):
            st.write(
                """
**Glucose** is blood sugar. It rises after you eat (especially carbs/sugary drinks).  
**Insulin** is a hormone that helps move glucose from the blood into cells.

Most people measure **glucose** at home (glucometer).  
**Insulin** usually requires a **lab test** (fasting insulin, C-peptide).  

This app is **educational** — it helps you *see* relationships, not diagnose disease.
                """
            )

    st.info("Tip: If you’re here for simple guidance, start with **Community View**.")

    # optional auto-nav buttons (works only within session)
    if st.session_state.get("nav_to") == "community":
        st.session_state["nav_to"] = None
        st.sidebar.radio("Go to", ["🏠 Home", "👥 Community View", "🧪 Simulator"], index=1)
    if st.session_state.get("nav_to") == "sim":
        st.session_state["nav_to"] = None
        st.sidebar.radio("Go to", ["🏠 Home", "👥 Community View", "🧪 Simulator"], index=2)

# ==================================================
# PAGE 2: COMMUNITY VIEW (catchy + checklists + examples)
# ==================================================
elif page == "👥 Community View":
    st.markdown(
        """
        <div class="hero">
            <div class="pill">🍽️ Food</div>
            <div class="pill">🚶 Movement</div>
            <div class="pill">💧 Hydration</div>
            <h1>Community View</h1>
            <p>Simple explanations, practical habits, and a weekly checklist. (Educational — not medical advice.)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning("This page is educational only. It does not diagnose diabetes and does not give prescriptions.")

    # Icons / “tiles”
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🍽️ Healthy Plate")
        st.write("Half vegetables, quarter protein, quarter whole grains.")
    with c2:
        st.subheader("💧 Water First")
        st.write("Replace sugary drinks with water/unsweetened tea.")
    with c3:
        st.subheader("🚶 Move Daily")
        st.write("Even a 20–30 min walk can help glucose control over time.")

    st.markdown("---")

    # Weekly plan checklist (stored in session)
    if "weekly" not in st.session_state:
        st.session_state.weekly = {
            "Water instead of sugary drinks (3+ days)": False,
            "Vegetables with meals (4+ days)": False,
            "20–30 min walk (4+ days)": False,
            "Whole grains instead of refined carbs (3+ days)": False,
            "Consistent sleep routine (4+ days)": False,
            "Check glucose (if you monitor) (as advised)": False,
        }

    st.subheader("✅ Weekly Plan Checklist")
    st.caption("Tick what you plan to work on this week (saved while this page stays open).")

    done_count = 0
    for k in list(st.session_state.weekly.keys()):
        st.session_state.weekly[k] = st.checkbox(k, value=st.session_state.weekly[k])
        done_count += 1 if st.session_state.weekly[k] else 0

    st.progress(done_count / max(len(st.session_state.weekly), 1))
    st.write(f"Progress: **{done_count} / {len(st.session_state.weekly)}**")

    st.markdown("---")

    # “If your curve looks like this…” examples
    st.subheader("📌 If your curve looks like this…")
    st.caption("These are *educational patterns* you might see in the Simulator.")

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.markdown("### 🔺 Big glucose peak")
        st.write("Often driven by higher intake or lower sensitivity.")
        st.write("Try: reduce intake slider, increase sensitivity via activity slider.")
    with ex2:
        st.markdown("### 🟡 Slow return to normal")
        st.write("Glucose takes longer to stabilize.")
        st.write("Try: smaller dt, or improve sensitivity.")
    with ex3:
        st.markdown("### 🟢 Stable & controlled")
        st.write("Lower peaks and steady behavior.")
        st.write("Keep habits consistent; avoid big sugar spikes.")

    with st.expander("🍎 Simple diet ideas (examples)"):
        st.write("• Breakfast: oats + peanuts + fruit (small portion)")
        st.write("• Lunch: beans + vegetables + brown rice/whole grain")
        st.write("• Dinner: vegetables + fish/eggs + small starch portion")
        st.write("• Snacks: nuts, yogurt (unsweetened), fruit (portion control)")

    with st.expander("When to seek professional help"):
        st.write("• Frequent thirst, frequent urination, weight loss, blurred vision")
        st.write("• Very high measured glucose readings")
        st.write("• If you are pregnant, on medication, or have complications")
        st.write("Always consult a clinician for diagnosis and treatment.")

    st.info("Ready to explore? Go to **🧪 Simulator** and try the intervention sliders.")

# ==================================================
# PAGE 3: SIMULATOR (your full model)
# ==================================================
else:
    st.markdown(
        """
        <div class="hero">
            <div class="pill">🧪 ODE</div>
            <div class="pill">📉 Error curves</div>
            <div class="pill">🧠 Intervention</div>
            <h1>Simulator</h1>
            <p>Explore models, parameters, and interventions. Community mode simplifies explanations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Choose model inside Simulator page
    model_choice_sim = st.sidebar.selectbox(
        "Model (Simulator)",
        ["Logistic Progression", "Glucose–Insulin ODE System"]
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Simulator Controls")
    days_sim = st.sidebar.slider("Simulation Days (Simulator)", 30, 365, 180)

    # -------------------------
    # Logistic
    # -------------------------
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
                st.write("• r controls progression speed")
                st.write("• K is long-run maximum")
                st.write("• Intervention reduces r (slows progression)")
            else:
                st.subheader("What this means (Simple)")
                st.write("Higher r → faster progression.")
                st.write("Intervention → slows progression.")
                st.info("Educational tool only — not medical advice.")

    # -------------------------
    # ODE
    # -------------------------
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

        st.sidebar.subheader("Intervention Scenario (Optional)")
        enable_intervention = st.sidebar.checkbox("Enable intervention scenario", value=True)
        adherence_int = st.sidebar.slider("Adherence to intervention (0-1)", 0.0, 1.0, 0.7)
        diet_reduction = st.sidebar.slider("Diet change (reduces intake)", 0.0, 1.0, 0.4)
        activity_increase = st.sidebar.slider("Activity change (improves sensitivity)", 0.0, 1.0, 0.4)

        intake_int = intake * (1 - adherence_int * diet_reduction)
        s_int = s * (1 + adherence_int * activity_increase)
        s_int = min(s_int, 0.02)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots()

            # RK4 baseline reference
            t_rk, G_rk, I_rk = simulate_glucose_insulin_rk4(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)

            if method == "Euler":
                t_e, G_e, I_e = simulate_glucose_insulin_euler(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)
                ax.plot(t_e, G_e, label="Glucose (Euler)")
                ax.plot(t_e, I_e, label="Insulin (Euler)")
                ax.set_title("Glucose–Insulin Dynamics (Euler)")

            elif method == "RK4":
                ax.plot(t_rk, G_rk, label="Glucose (RK4)")
                ax.plot(t_rk, I_rk, label="Insulin (RK4)")
                ax.set_title("Glucose–Insulin Dynamics (RK4)")

            else:
                t_e, G_e, I_e = simulate_glucose_insulin_euler(days_sim, dt, G0, I0, intake, s, alpha, kg, ki)
                ax.plot(t_e, G_e, label="Glucose (Euler)")
                ax.plot(t_rk, G_rk, linestyle="--", label="Glucose (RK4)")
                ax.plot(t_e, I_e, label="Insulin (Euler)")
                ax.plot(t_rk, I_rk, linestyle="--", label="Insulin (RK4)")
                ax.set_title("Euler vs RK4 Comparison")

                diff_G = float(np.max(np.abs(G_e - G_rk)))
                diff_I = float(np.max(np.abs(I_e - I_rk)))
                st.info(f"Numerical error (max abs): Glucose={diff_G:.3f}, Insulin={diff_I:.3f}")

            if enable_intervention:
                _, G_int, I_int = simulate_glucose_insulin_rk4(days_sim, dt, G0, I0, intake_int, s_int, alpha, kg, ki)
                ax.plot(t_rk, G_int, linestyle=":", label="Glucose (Intervention RK4)")
                ax.plot(t_rk, I_int, linestyle=":", label="Insulin (Intervention RK4)")

            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Level")
            ax.legend()
            st.pyplot(fig)

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

        with col2:
            if audience_mode == "Student/Research (Detailed)":
                st.subheader("Interpretation (Research)")
                st.latex(r"\frac{dG}{dt} = \text{intake} - sGI - k_g G")
                st.latex(r"\frac{dI}{dt} = \alpha G - k_i I")
                st.write("RK4 is higher accuracy; Euler can deviate for larger dt.")
                if enable_intervention:
                    st.markdown("---")
                    st.write(f"Baseline intake = {intake:.2f} → Intervention intake = {intake_int:.2f}")
                    st.write(f"Baseline s = {s:.4f} → Intervention s = {s_int:.4f}")
            else:
                st.subheader("What this means (Simple)")
                st.write("• Intake pushes glucose up.")
                st.write("• Insulin helps reduce glucose.")
                st.write("• Sensitivity means how well insulin works.")
                st.warning("Educational tool only — not diagnosis or medical advice.")

                summary = curve_summary(t_rk, G_rk, I_rk)
                msgs, tips = community_guidance(summary)

                st.markdown("### Quick summary (simulation)")
                st.write(f"Peak glucose: **{summary['peak_G']:.1f}**")
                st.write(f"End glucose: **{summary['end_G']:.1f}**")
                st.write(f"Peak insulin: **{summary['peak_I']:.1f}**")

                if enable_intervention:
                    _, G_int_s, I_int_s = simulate_glucose_insulin_rk4(days_sim, dt, G0, I0, intake_int, s_int, alpha, kg, ki)
                    summary_int = curve_summary(t_rk, G_int_s, I_int_s)
                    st.markdown("### Intervention comparison (educational)")
                    st.write(f"Peak glucose (baseline) = **{summary['peak_G']:.1f}**")
                    st.write(f"Peak glucose (intervention) = **{summary_int['peak_G']:.1f}**")
                    if summary_int["peak_G"] < summary["peak_G"]:
                        st.success("In this simulation, the intervention reduces the glucose peak.")
                    else:
                        st.info("Try increasing diet/activity sliders to see stronger effects.")

                with st.expander("General lifestyle tips (not medical advice)"):
                    for tip in tips:
                        st.write("• " + tip)
