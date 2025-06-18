import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pix2tex.cli import LatexOCR
import re
import random

def generate_practice_polynomial(input_poly):
    x = sp.symbols('x')
    poly = sp.Poly(input_poly, x)
    degree = poly.degree()
    coeffs = poly.all_coeffs()
    max_coeff = max(abs(float(c)) for c in coeffs) if coeffs else 10
    coeff_range = (-int(max_coeff), int(max_coeff))
    
    if degree == 2:
        a = random.randint(1, max(1, coeff_range[1]))
        b = random.randint(coeff_range[0], coeff_range[1])
        c = random.randint(coeff_range[0], coeff_range[1])
        input_coeffs = poly.all_coeffs()
        input_discriminant = input_coeffs[1]**2 - 4*input_coeffs[0]*input_coeffs[2]
        discriminant = b**2 - 4*a*c
        while (input_discriminant >= 0 and discriminant < 0) or (input_discriminant < 0 and discriminant >= 0):
            b = random.randint(coeff_range[0], coeff_range[1])
            c = random.randint(coeff_range[0], coeff_range[1])
            discriminant = b**2 - 4*a*c
        return a*x**2 + b*x + c, x
    else:
        coeffs = [random.randint(coeff_range[0], coeff_range[1]) for _ in range(degree + 1)]
        coeffs[0] = random.randint(1, max(1, coeff_range[1]))
        practice_poly = sum(c * x**i for i, c in enumerate(coeffs[::-1]))
        return practice_poly, x

def parse_polynomial(poly_str):
    try:
        poly_str = poly_str.replace('=', '').replace('−', '-').replace('\\', '').strip()
        poly_str = re.sub(r'\\pi', 'pi', poly_str)
        poly_str = re.sub(r'\\sqrt\{(\d+)\}', r'sqrt(\1)', poly_str)
        poly_str = re.sub(r'\\e', 'E', poly_str)
        poly_str = poly_str.replace('^', '**').replace(' ', '')
        poly_str = re.sub(r'[\{\}]', '', poly_str)
        x = sp.symbols('x')
        pi, e = sp.symbols('pi E', real=True)
        poly = sp.sympify(poly_str, locals={'x': x, 'pi': pi, 'E': e})
        return poly, x
    except Exception as e:
        st.error(f"Error parsing polynomial: {e}. Please enter a valid polynomial (e.g., x**2 - 1, pi*x**2 + sqrt(2)*x - e, or x**3 - 7*x**2 + 7*x - 7).")
        return None, None

def extract_equations_from_image(image):
    try:
        model = LatexOCR()
        latex_text = model(image)
        st.write(f"**Raw LaTeX extracted**: {latex_text}")
        latex_text = latex_text.replace('$', '').strip()
        latex_text = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', latex_text, flags=re.DOTALL)
        latex_text = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', latex_text, flags=re.DOTALL)
        latex_text = re.sub(r'\\\[|\\\]', '', latex_text)
        latex_text = re.sub(r'=', '', latex_text)
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
        st.error(f"Error processing image: {e}")
        return None

def solve_polynomial(poly, x):
    steps = []
    try:
        steps.append(f"**Given polynomial**: ${sp.latex(poly)}$")
        
        roots = sp.solve(poly, x)
        real_roots = []
        
        factors = sp.factor(poly, extension=sp.sqrt(57))
        if str(factors) == str(poly):
            steps.append(f"**Factored form**: Polynomial is irreducible over rationals. Computing numerical factorization.")
            factor_str = ""
            exact_real_root = None
            for root in roots:
                num_root = root.evalf(15)
                im_part = float(sp.im(num_root)) if sp.im(num_root) else 0.0
                re_part = float(sp.re(num_root)) if sp.re(num_root) else 0.0
                if abs(im_part) < 1e-10:
                    real_root = round(re_part, 3)
                    factor_str += f"(x - {real_root})"
                    real_roots.append(real_root)
                    exact_real_root = root
            if real_roots and exact_real_root is not None and sp.Poly(poly, x).degree() >= 2:
                linear_factor = x - exact_real_root
                quotient, remainder = sp.div(poly, linear_factor)
                if remainder == 0:
                    coeffs = sp.Poly(quotient, x).all_coeffs()
                    a = round(float(coeffs[0].evalf()), 3) if len(coeffs) > 0 else 1.0
                    b = round(float(coeffs[1].evalf()), 3) if len(coeffs) > 1 else 0.0
                    c = round(float(coeffs[2].evalf()), 3) if len(coeffs) > 2 else 0.0
                    quad_factor = f"{a}x^2 + {b}x + {c}" if b >= 0 else f"{a}x^2 - {abs(b)}x + {c}"
                    factor_str += f"({quad_factor})"
                    steps.append(f"**Numerical factored form**: ${factor_str}$")
                else:
                    linear_factor = x - real_roots[0]
                    quotient, remainder = sp.div(poly, linear_factor, domain='RR')
                    if abs(float(remainder.evalf())) < 1e-10:
                        coeffs = sp.Poly(quotient, x).all_coeffs()
                        a = round(float(coeffs[0].evalf()), 3) if len(coeffs) > 0 else 1.0
                        b = round(float(coeffs[1].evalf()), 3) if len(coeffs) > 1 else 0.0
                        c = round(float(coeffs[2].evalf()), 3) if len(coeffs) > 2 else 0.0
                        quad_factor = f"{a}x^2 + {b}x + {c}" if b >= 0 else f"{a}x^2 - {abs(b)}x + {c}"
                        factor_str += f"({quad_factor})"
                        steps.append(f"**Numerical factored form**: ${factor_str}$")
                    else:
                        steps.append(f"**Numerical factored form**: ${factor_str}$ (Quadratic factor not computed due to numerical precision.)")
            else:
                steps.append(f"**Numerical factored form**: ${factor_str}$")
        else:
            steps.append(f"**Factored form**: ${sp.latex(factors)}$")

        steps.append("**Roots found by solving f(x) = 0**:")
        real_roots = []
        for i, root in enumerate(roots):
            num_root = root.evalf(15)
            im_part = float(sp.im(num_root)) if sp.im(num_root) else 0.0
            re_part = float(sp.re(num_root)) if sp.re(num_root) else 0.0
            if abs(im_part) < 1e-10:
                rounded_root = f"{round(re_part, 3)}"
            else:
                rounded_im = round(im_part, 3)
                rounded_re = round(re_part, 3)
                sign = '+' if rounded_im >= 0 else '-'
                rounded_root = f"{rounded_re} {sign} {abs(rounded_im)}i"
            steps.append(f"Root {i+1}: x = {rounded_root}")
            if abs(im_part) < 1e-10:
                real_roots.append(round(re_part, 3))
        
        real_roots = sorted(list(set(real_roots)))
        steps.append(f"**Real roots**: {real_roots}")
        return roots, real_roots, steps
    except Exception as e:
        st.error(f"Error solving polynomial: {e}")
        return None, None, steps

def plot_polynomial(poly, x, real_roots):
    try:
        f = sp.lambdify(x, poly, 'numpy')
        if real_roots:
            x_min = min(real_roots) - 3
            x_max = max(real_roots) + 3
            if x_max - x_min < 5.0:
                x_min -= 2
                x_max += 2
        else:
            x_min, x_max = -10.0, 10.0
        x_vals = np.linspace(x_min, x_max, 500)
        y_vals = f(x_vals)
        y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, y_vals, label=f'f(x) = {sp.latex(poly)}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Polynomial Graph')
        ax.legend()

        for root in real_roots:
            ax.plot(root, 0, 'ro', label=f'Root at x={root:.3f}')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        st.pyplot(fig)
        if not real_roots:
            st.write("Note: No real roots to plot. Graph shows polynomial behavior.")
        elif len(real_roots) < len(sp.solve(poly, x)):
            st.write("Note: Only real roots are plotted. Complex roots are not shown on the graph.")
    except Exception as e:
        st.error(f"Error plotting polynomial: {e}")

def polynomial_solver_tab():
    st.header("Polynomial Solver")
    input_method = st.radio("Choose input method:", ("Text", "Image"), key="poly_input")

    poly = None
    x = sp.symbols('x')

    if input_method == "Text":
        poly_str = st.text_input("Enter a polynomial (e.g., x**2 - 1, pi*x**2 + sqrt(2)*x - e, or x**3 - 7*x**2 + 7*x - 7):")
        if poly_str:
            poly, x = parse_polynomial(poly_str)
    elif input_method == "Image":
        uploaded_file = st.file_uploader("Upload an image containing a polynomial:", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            poly_str = extract_equations_from_image(image)
            if poly_str:
                st.write(f"Extracted polynomial: {poly_str}")
                poly, x = parse_polynomial(poly_str)

    if poly:
        st.subheader("Step-by-Step Solution for Input Polynomial")
        roots, real_roots, steps = solve_polynomial(poly, x)
        for step in steps:
            st.markdown(step)
        
        if roots is not None:
            st.subheader("Graph of the Input Polynomial")
            plot_polynomial(poly, x, real_roots)
        
        st.subheader("Practice with a Similar Polynomial")
        if st.button("Generate Similar Practice Polynomial"):
            practice_poly, x = generate_practice_polynomial(poly)
            st.write(f"**Generated practice polynomial**: ${sp.latex(practice_poly)}$")
            st.subheader("Step-by-Step Solution for Practice Polynomial")
            practice_roots, practice_real_roots, practice_steps = solve_polynomial(practice_poly, x)
            for step in practice_steps:
                st.markdown(step)
            if practice_roots is not None:
                st.subheader("Graph of the Practice Polynomial")
                plot_polynomial(practice_poly, x, practice_real_roots)

    st.markdown("**Note**: For image input, ensure the polynomial is clearly written without '=0' (e.g., x^2 + 2x - 5) and use pi, sqrt(n), or e for constants. For text input, use ** for powers (e.g., x**2) and pi, sqrt(n), or e (e.g., pi*x**2 + sqrt(2)*x - e).")
