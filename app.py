import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Simulator")

st.title("🩺 Interactive Diabetes Progression Simulator")

st.sidebar.header("Model Parameters")

days = st.sidebar.slider("Simulation Days", 30, 365, 180)
r = st.sidebar.slider("Progression Rate (r)", 0.01, 0.2, 0.05)
K = st.sidebar.slider("Maximum Disease Burden (K)", 0.5, 2.0, 1.0)
D0 = st.sidebar.slider("Initial Disease Level", 0.01, 0.5, 0.1)

# Time grid
t = np.linspace(0, days, days)

# Logistic model
D = np.zeros_like(t)
D[0] = D0

for i in range(1, len(t)):
    D[i] = D[i-1] + r * D[i-1] * (1 - D[i-1] / K)

# Plot
fig, ax = plt.subplots()
ax.plot(t, D)
ax.set_xlabel("Days")
ax.set_ylabel("Disease Level")
ax.set_title("Logistic Progression Model")

st.pyplot(fig)
