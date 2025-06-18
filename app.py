import streamlit as st
from polynomial_solver import polynomial_solver_tab
from linear_equations_solver import linear_equations_solver_tab

# Streamlit App
st.title("Math Solver")

# Create tabs
tab1, tab2 = st.tabs(["Polynomial Solver", "Linear Equations Solver"])

# Polynomial Solver Tab
with tab1:
    polynomial_solver_tab()

# Linear Equations Solver Tab
with tab2:
    linear_equations_solver_tab()