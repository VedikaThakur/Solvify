import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pix2tex.cli import LatexOCR
import re
import random
from PIL import ImageEnhance

def parse_linear_equations(eqn_str):
    try:
        x, y = sp.symbols('x y')
        possible_separators = [',', '\\\\', '&', r'\n']
        eqns = None
        for sep in possible_separators:
            if re.search(sep, eqn_str):
                eqns = [e.strip() for e in re.split(sep, eqn_str) if e.strip()]
                break
        if not eqns:
            eqns = [eqn_str.strip()]
        if len(eqns) != 2:
            raise ValueError(f"Expected exactly two equations, but found {len(eqns)}. Please separate equations with a comma, newline, or LaTeX separator (e.g., 'x + y = 2, x - y = 0').")
        equations = []
        for eqn in eqns:
            sides = eqn.split('=')
            if len(sides) != 2:
                raise ValueError(f"Invalid equation format: {eqn}. Each equation must contain exactly one '=' sign.")
            expr = sp.sympify(sides[0].replace(' ', '') + '-(' + sides[1].replace(' ', '') + ')', locals={'x': x, 'y': y, 'pi': sp.pi, 'E': sp.E})
            equations.append(expr)
        return equations, x, y
    except Exception as e:
        st.error(f"Error parsing linear equations: {e}")
        return None, None, None

def preprocess_image(image):
    # Convert to grayscale and enhance contrast
    image = image.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast
    return image

def extract_equations_from_image(image):
    try:
        # Preprocess the image
        image = preprocess_image(image)
        model = LatexOCR()
        latex_text = model(image)
        st.write(f"**Raw LaTeX extracted**: {latex_text}")
        latex_text = latex_text.replace('$', '').strip()
        latex_text = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', latex_text, flags=re.DOTALL)
        latex_text = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', latex_text, flags=re.DOTALL)
        latex_text = re.sub(r'\\\[|\\\]', '', latex_text)
        latex_text = re.sub(r'=', '=', latex_text)  # Retain '=' for parsing
        latex_text = re.sub(r'\\left\(|\\right\)', '', latex_text)
        latex_text = latex_text.replace('\\cdot', '*')
        latex_text = re.sub(r'\^{(.*?)}', r'^\1', latex_text)
        latex_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', latex_text)
        latex_text = re.sub(r'\{|\}', '', latex_text)
        latex_text = re.sub(r'\\pi', 'pi', latex_text)
        latex_text = re.sub(r'\\sqrt\{(\d+)\}', r'sqrt(\1)', latex_text)
        latex_text = re.sub(r'\\e', 'E', latex_text)
        return latex_text
    except Exception as e:
        st.error(f"Error processing image: {e}. Ensure the image is clear, with high contrast and legible text.")
        return None

def solve_linear_equations(equations, x, y):
    steps = []
    try:
        steps.append(f"**Given equations**:")
        steps.append(f"1. ${sp.latex(equations[0])} = 0$")
        steps.append(f"2. ${sp.latex(equations[1])} = 0$")

        steps.append("**Solving using elimination method**:")
        eq1 = sp.Poly(equations[0], x, y).as_dict()
        eq2 = sp.Poly(equations[1], x, y).as_dict()
        a1 = float(eq1.get((1, 0), 0))
        b1 = float(eq1.get((0, 1), 0))
        c1 = float(eq1.get((0, 0), 0))
        a2 = float(eq2.get((1, 0), 0))
        b2 = float(eq2.get((0, 1), 0))
        c2 = float(eq2.get((0, 0), 0))

        steps.append(f"Multiply first equation by {b2} and second by {b1} to eliminate y:")
        eq1_scaled = b2 * equations[0]
        eq2_scaled = b1 * equations[1]
        steps.append(f"1. ${sp.latex(eq1_scaled)} = 0$")
        steps.append(f"2. ${sp.latex(eq2_scaled)} = 0$")

        new_eq = eq1_scaled - eq2_scaled
        steps.append(f"Subtract second from first: ${sp.latex(new_eq)} = 0$")
        x_solution = sp.solve(new_eq, x)
        if not x_solution:
            if new_eq == 0:
                steps.append("Equations are dependent (infinitely many solutions).")
                return None, None, steps
            else:
                steps.append("No solution (inconsistent system).")
                return None, None, steps
        x_val = x_solution[0]
        steps.append(f"Solve for x: $x = {sp.latex(x_val)}$")

        y_eq = equations[0].subs(x, x_val)
        y_solution = sp.solve(y_eq, y)
        if not y_solution:
            steps.append("Error solving for y.")
            return None, None, steps
        y_val = y_solution[0]
        steps.append(f"Substitute x into first equation: ${sp.latex(y_eq)} = 0$")
        steps.append(f"Solve for y: $y = {sp.latex(y_val)}$")

        solution = (float(x_val.evalf()), float(y_val.evalf()))
        steps.append(f"**Solution**: $(x, y) = ({round(solution[0], 3)}, {round(solution[1], 3)})$")
        return solution, solution, steps
    except Exception as e:
        st.error(f"Error solving linear equations: {e}")
        return None, None, steps

def plot_linear_equations(equations, x, y, solution):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        x_vals = np.linspace(-50, 50, 100)
        for i, eq in enumerate(equations):
            y_expr = sp.solve(eq, y)
            if not y_expr:
                x_expr = sp.solve(eq, x)
                if x_expr:
                    x_val = float(x_expr[0])
                    ax.axvline(x_val, label=f'Equation {i+1}')
                    continue
                else:
                    st.error("Invalid equation for plotting.")
                    return
            y_func = sp.lambdify(x, y_expr[0], 'numpy')
            y_vals = y_func(x_vals)
            y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)
            ax.plot(x_vals, y_vals, label=f'Equation {i+1}: {sp.latex(eq)} = 0')
        
        if solution:
            ax.plot(solution[0], solution[1], 'ro', label=f'Solution ({round(solution[0], 3)}, {round(solution[1], 3)})')
        
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Graph of Linear Equations')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting linear equations: {e}")

def generate_practice_linear_equations(equations):
    x, y = sp.symbols('x y')
    max_coeff = 1
    for eq in equations:
        coeffs = sp.Poly(eq, x, y).as_dict()
        for coeff in coeffs.values():
            max_coeff = max(max_coeff, abs(float(coeff)))
    coeff_range = (-int(max_coeff), int(max_coeff))
    while True:
        a1 = random.randint(-5, 5)
        b1 = random.randint(-5, 5)
        c1 = random.randint(coeff_range[0], coeff_range[1])
        a2 = random.randint(-5, 5)
        b2 = random.randint(-5, 5)
        c2 = random.randint(coeff_range[0], coeff_range[1])
        if a1*b2 - a2*b1 != 0:
            break
    eq1 = a1*x + b1*y - c1
    eq2 = a2*x + b2*y - c2
    return [eq1, eq2], x, y

def linear_equations_solver_tab():
    st.header("Linear Equations Solver")
    input_method = st.radio("Choose input method:", ("Text", "Image"), key="linear_input")

    equations = None
    x, y = sp.symbols('x y')

    if input_method == "Text":
        eqn_str = st.text_input("Enter two linear equations separated by a comma (e.g., x + y = 2, x - y = 0):")
        if eqn_str:
            equations, x, y = parse_linear_equations(eqn_str)
    elif input_method == "Image":
        st.write("Upload two images, each containing one linear equation.")
        col1, col2 = st.columns(2)
        eqn1_str = None  # Initialize to avoid unbound variable
        eqn2_str = None  # Initialize to avoid unbound variable
        with col1:
            uploaded_file1 = st.file_uploader("Upload image for first equation:", type=["png", "jpg", "jpeg"], key="eq1_image")
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="First Equation Image", use_container_width=True)
                eqn1_str = extract_equations_from_image(image1)
        with col2:
            uploaded_file2 = st.file_uploader("Upload image for second equation:", type=["png", "jpg", "jpeg"], key="eq2_image")
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Second Equation Image", use_container_width=True)
                eqn2_str = extract_equations_from_image(image2)
        
        if uploaded_file1 and uploaded_file2 and eqn1_str and eqn2_str:
            st.write(f"**Extracted first equation**: {eqn1_str}")
            st.write(f"**Extracted second equation**: {eqn2_str}")
            # Provide manual correction option if extraction is poor
            corrected_eqn1 = st.text_input("Correct first equation if needed (leave blank to use extracted):", value=eqn1_str, key="corr_eq1")
            corrected_eqn2 = st.text_input("Correct second equation if needed (leave blank to use extracted):", value=eqn2_str, key="corr_eq2")
            eqn_str = f"{corrected_eqn1 if corrected_eqn1 else eqn1_str}, {corrected_eqn2 if corrected_eqn2 else eqn2_str}"
            equations, x, y = parse_linear_equations(eqn_str)
        elif uploaded_file1 and uploaded_file2 and (not eqn1_str or not eqn2_str):
            st.error("Failed to extract equations from one or both images. Please ensure images are clear and retry, or switch to text input.")

    if equations:
        st.subheader("Step-by-Step Solution for Input Equations")
        solution, _, steps = solve_linear_equations(equations, x, y)
        for step in steps:
            st.markdown(step)
        
        if solution:
            st.subheader("Graph of the Input Equations")
            plot_linear_equations(equations, x, y, solution)
        
        st.subheader("Practice with Similar Equations")
        if st.button("Generate Similar Practice Equations"):
            practice_eqns, x, y = generate_practice_linear_equations(equations)
            st.write(f"**Generated practice equations**:")
            st.write(f"1. ${sp.latex(practice_eqns[0])} = 0$")
            st.write(f"2. ${sp.latex(practice_eqns[1])} = 0$")
            st.subheader("Step-by-Step Solution for Practice Equations")
            practice_solution, _, practice_steps = solve_linear_equations(practice_eqns, x, y)
            for step in practice_steps:
                st.markdown(step)
            if practice_solution:
                st.subheader("Graph of the Practice Equations")
                plot_linear_equations(practice_eqns, x, y, practice_solution)

    st.markdown("**Note**: For image input, upload two separate images, each containing one equation (e.g., 'x + y = 2' in the first image, 'x - y = 0' in the second). Ensure images are clear, with high contrast and legible handwriting. For text input, enter two equations separated by a comma (e.g., x + y = 2, x - y = 0).")