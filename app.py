import streamlit as st
import math
import cmath
import html
import numpy as np
import control as ct

st.set_page_config(page_title="Projeto de Controladores", layout="wide", initial_sidebar_state="expanded")

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #182326;
    --accent: #0f4c42;
    --accent-2: #c96b5b;
    --paper: #f7f2ea;
    --paper-2: #f1e8dd;
    --line: #ded4c6;
}

.stApp {
    background: radial-gradient(1100px 700px at 12% 8%, #ffffff 0%, var(--paper) 55%, var(--paper-2) 100%);
    color: var(--ink);
}

.block-container {
    padding-top: 2.2rem;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: visible; }

/* =========================================================
   BOTÕES DE RECOLHER/EXPANDIR (>> e <<) E ÍCONES EM GERAL
   ========================================================= */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
.stNumberInput button {
    opacity: 1 !important;
    visibility: visible !important;
    color: var(--ink) !important;
}

/* Força a cor escura no desenho (SVG) das setas */
button svg, 
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--ink) !important;
    color: var(--ink) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3, h4 {
    font-family: 'IBM Plex Serif', serif;
    letter-spacing: -0.4px;
    color: var(--ink);
}

body, p, div, span, label, input, textarea {
    font-family: 'Space Grotesk', sans-serif;
}

/* =========================================================
   TEXTOS DA SIDEBAR (MENU LATERAL)
   ========================================================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1e8dd 100%);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: var(--ink) !important;
}

/* =========================================================
   FORÇA A COR ESCURA NOS TÍTULOS DAS ENTRADAS DE DADOS
   ========================================================= */
div[data-testid="stWidgetLabel"] p,
.stTextInput label p, 
.stNumberInput label p,
.stSelectbox label p,
.stRadio label p,
.stCheckbox label p {
    color: var(--ink) !important;
    font-weight: 600 !important;
}

/* =========================================================
   ESTILIZA AS CAIXAS ONDE O USUÁRIO DIGITA
   ========================================================= */
.stTextInput input, 
.stTextArea textarea, 
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}

input::placeholder {
    color: #888888 !important;
    opacity: 1 !important;
}

.hero {
    border: 1px solid rgba(15, 76, 66, 0.18);
    border-radius: 18px;
    padding: 20px 24px;
    background: linear-gradient(120deg, rgba(255,255,255,0.9) 0%, rgba(247,242,234,0.9) 55%, rgba(239,231,220,0.9) 100%);
    box-shadow: 0 12px 30px rgba(31,42,46,0.06);
    margin-bottom: 18px;
    animation: fadeInUp 0.5s ease-out;
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 6px;
}

.stButton > button {
    background: rgba(15, 76, 66, 0.10) !important;
    color: var(--ink) !important;
    border: 1px solid rgba(15, 76, 66, 0.22) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: rgba(15, 76, 66, 0.18) !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
)

# ==========================================
# FUNÇÕES MATEMÁTICAS AUXILIARES
# ==========================================

def get_zpk_latex(sys):
    """Gera a fração LaTeX na forma fatorada (Polos e Zeros explícitos)."""
    sys_tf = ct.tf(sys)
    zeros = sys_tf.zeros()
    poles = sys_tf.poles()
    K = sys_tf.num[0][0][0] / sys_tf.den[0][0][0]
    
    def format_roots(roots):
        if len(roots) == 0:
            return "1"
        s_str = ""
        skip_idx = []
        for i, r in enumerate(roots):
            if i in skip_idx: continue
            if abs(r.imag) < 1e-8: 
                r_real = float(r.real)
                if abs(r_real) < 1e-8: s_str += "s" 
                elif r_real < 0: s_str += f"(s + {abs(r_real):.4g})"
                else: s_str += f"(s - {r_real:.4g})"
            else: 
                conj_idx = -1
                for j in range(i+1, len(roots)):
                    if j not in skip_idx and abs(roots[j].real - r.real) < 1e-8 and abs(roots[j].imag + r.imag) < 1e-8:
                        conj_idx = j
                        break
                if conj_idx != -1:
                    w_n_sq = float(r.real**2 + r.imag**2)
                    two_zeta_wn = float(-2 * r.real)
                    term = "(s^2"
                    if abs(two_zeta_wn) > 1e-8:
                        term += f" + {two_zeta_wn:.4g}s" if two_zeta_wn > 0 else f" - {abs(two_zeta_wn):.4g}s"
                    term += f" + {w_n_sq:.4g})"
                    s_str += term
                    skip_idx.append(conj_idx)
                else:
                    s_str += f"(s - ({r.real:.4g} + j{r.imag:.4g}))"
        return s_str

    num_str = format_roots(zeros)
    den_str = format_roots(poles)
    k_str = "" if abs(K - 1.0) < 1e-8 else f"{K:.4g}"
        
    if num_str == "1" and k_str == "": num_final = "1"
    elif num_str == "1": num_final = k_str
    else: num_final = k_str + num_str if k_str else num_str
    return rf"\frac{{{num_final}}}{{{den_str}}}"

def poly_to_latex(coefs):
    terms = []
    degree = len(coefs) - 1
    for i, coef in enumerate(coefs):
        if abs(coef) < 1e-10: continue
        power = degree - i
        if power == 0: term = f"{coef:g}" 
        else:
            coef_str = "" if coef == 1 else ("-" if coef == -1 else f"{coef:g}")
            term = f"{coef_str}s" if power == 1 else f"{coef_str}s^{{{power}}}"
        terms.append(term)
    if not terms: return "0"
    result = terms[0]
    for term in terms[1:]:
        result += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return result

def get_tf_latex(sys):
    sys_tf = ct.tf(sys) 
    num_str = poly_to_latex(sys_tf.num[0][0])
    den_str = poly_to_latex(sys_tf.den[0][0])
    return rf"\frac{{{num_str}}}{{{den_str}}}"

def _leading_gain(tf):
    return tf.num[0][0][0] / tf.den[0][0][0]

def _fmt_num(x):
    if np.iscomplexobj(x) and abs(x.imag) > 1e-9:
        return f"{x.real:+.2f}{x.imag:+.2f}j"
    return f"{float(np.real(x)):+.2f}"

def _normalize_angle_deg(angle):
    return (angle + 180) % 360 - 180

def _calc_sd_from_mp_ts(mp_max, ts_req, criterio_ts):
    mp_decimal = mp_max / 100.0
    zeta = -math.log(mp_decimal) / math.sqrt(math.pi**2 + math.log(mp_decimal)**2)
    fator_ts = 4.0 if criterio_ts <= 2.5 else 3.0
    sigma = fator_ts / ts_req
    omega_n = sigma / zeta
    omega_d = omega_n * math.sqrt(1 - zeta**2)
    s_d = -sigma + 1j * omega_d
    return zeta, sigma, omega_n, omega_d, s_d

def _calc_sd_from_zeta_omega(zeta, omega_n):
    sigma = zeta * omega_n
    omega_d = omega_n * math.sqrt(1 - zeta**2)
    s_d = -sigma + 1j * omega_d
    return sigma, omega_d, s_d

def _calc_zc(sigma, omega_d, phi_c_deg):
    if abs(omega_d) < 1e-12 or abs(abs(phi_c_deg) - 180) < 1e-9:
        return sigma
    phi_c_rad = math.radians(phi_c_deg)
    if abs(math.tan(phi_c_rad)) < 1e-12:
        return sigma
    return sigma + omega_d / math.tan(phi_c_rad)

def _print_contribuicoes_angulo(zeros, polos, s_d):
    angulos_zeros = []
    angulos_polos = []

    def format_complex_latex(c):
        if abs(c.imag) < 1e-8: return f"{c.real:.4g}"
        sinal = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4g} {sinal} j{abs(c.imag):.4g}"

    st.markdown("**Contribuição de Ângulo dos Zeros:**")
    if len(zeros) == 0:
        st.markdown("- *Não há zeros no sistema base.*")
    else:
        for z in zeros:
            vetor = s_d - z
            ang = math.degrees(cmath.phase(vetor))
            angulos_zeros.append(ang)
            z_str, vetor_str = format_complex_latex(z), format_complex_latex(vetor)
            st.latex(rf"\angle(s_d - ({z_str})) = \angle({vetor_str}) = {ang:.2f}^\circ")

    st.markdown("**Contribuição de Ângulo dos Polos:**")
    if len(polos) == 0:
        st.markdown("- *Não há polos no sistema base.*")
    else:
        for p in polos:
            vetor = s_d - p
            ang = math.degrees(cmath.phase(vetor))
            angulos_polos.append(ang)
            p_str, vetor_str = format_complex_latex(p), format_complex_latex(vetor)
            st.latex(rf"\angle(s_d - ({p_str})) = \angle({vetor_str}) = {ang:.2f}^\circ")

    soma_zeros, soma_polos = sum(angulos_zeros), sum(angulos_polos)
    ang_total = soma_zeros - soma_polos

    zeros_str = " + ".join([f"{a:.2f}^\circ" for a in angulos_zeros]) if angulos_zeros else "0.00^\circ"
    polos_str = " + ".join([f"{a:.2f}^\circ" for a in angulos_polos]) if angulos_polos else "0.00^\circ"

    st.markdown("**Somando as contribuições:**")
    st.latex(rf"\Sigma\angle \text{{zeros}} = {zeros_str} = {soma_zeros:.2f}^\circ")
    st.latex(rf"\Sigma\angle \text{{polos}} = {polos_str} = {soma_polos:.2f}^\circ")
    st.latex(rf"\Sigma\angle \text{{total}} = {soma_zeros:.2f}^\circ - ({soma_polos:.2f}^\circ) = {ang_total:.2f}^\circ")
    return ang_total

def _find_cancellations(zeros, polos, tol=1e-6):
    return [z for z in zeros for p in polos if abs(z - p) < tol]

def _poly_to_str(coeffs, var):
    coeffs = [float(np.real(c)) for c in coeffs]
    degree = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-12: continue
        power = degree - i
        coeff_abs = abs(c)
        if power == 0: term = f"{coeff_abs:.4f}"
        elif power == 1: term = f"{coeff_abs:.4f}{var}"
        else: term = f"{coeff_abs:.4f}{var}^{power}"
        sign = "-" if c < 0 else "+"
        parts.append((sign, term))
    if not parts: return "0"
    sign, term = parts[0]
    expr = f"{term}" if sign == "+" else f"-{term}"
    for sign, term in parts[1:]:
        expr += f" {sign} {term}"
    return expr

def _tf_to_latex(tf_obj, var="s"):
    return rf"\frac{{{_poly_to_str(tf_obj.num[0][0], var)}}}{{{_poly_to_str(tf_obj.den[0][0], var)}}}"

def _poly_subst_euler(coeffs, T):
    degree = len(coeffs) - 1
    p, poly = np.poly1d([1.0, -1.0]), np.poly1d([0.0])
    for i, c in enumerate(coeffs):
        power = degree - i
        poly += (float(np.real(c)) / (T ** power)) * (p ** power)
    coeffs_z = np.trim_zeros(np.real_if_close(poly.c, tol=1e-9), "f")
    return np.array([0.0]) if coeffs_z.size == 0 else np.array(coeffs_z, dtype=float)

def _poly_mul_power(coeffs, base_poly, power):
    if power <= 0: return np.array(coeffs, dtype=float)
    coeffs_out = np.trim_zeros(np.real_if_close((np.poly1d(coeffs) * (base_poly ** power)).c, tol=1e-9), "f")
    return np.array([0.0]) if coeffs_out.size == 0 else np.array(coeffs_out, dtype=float)

def _poly_subst_backward(coeffs, T):
    degree = len(coeffs) - 1
    z_poly, p, poly = np.poly1d([1.0, 0.0]), np.poly1d([1.0, -1.0]), np.poly1d([0.0])
    for i, c in enumerate(coeffs):
        power = degree - i
        poly += (float(np.real(c)) / (T ** power)) * (p ** power) * (z_poly ** (degree - power))
    coeffs_z = np.trim_zeros(np.real_if_close(poly.c, tol=1e-9), "f")
    return np.array([0.0]) if coeffs_z.size == 0 else np.array(coeffs_z, dtype=float)

def _poly_subst_tustin(coeffs, T):
    degree = len(coeffs) - 1
    p, q, poly = np.poly1d([1.0, -1.0]), np.poly1d([1.0, 1.0]), np.poly1d([0.0])
    for i, c in enumerate(coeffs):
        power = degree - i
        poly += (float(np.real(c)) * (2.0 / T) ** power) * (p ** power) * (q ** (degree - power))
    coeffs_z = np.trim_zeros(np.real_if_close(poly.c, tol=1e-9), "f")
    return np.array([0.0]) if coeffs_z.size == 0 else np.array(coeffs_z, dtype=float)

# ==========================================
# FUNÇÕES DE RESOLUÇÃO E DISCRETIZAÇÃO
# ==========================================

def _discretizar_controlador(tipo, Kp, Ki, Kd):
    st.markdown("---")
    st.header("Discretização do Controlador")

    if not st.checkbox(f"Deseja discretizar o controlador {tipo}?"):
        return

    col1, col2 = st.columns(2)
    with col1:
        T = st.number_input("Período de amostragem T (s)", value=None, format="%.4f", placeholder="Ex: 0.1")
    with col2:
        metodo = st.selectbox(
            "Método de discretização:",
            options=["1) Euler (Forward)", "2) Euler (Backward)", "3) Tustin (Bilinear)"],
            index=2
        )

    if T is None:
        st.warning("Informe o período de amostragem (T) para calcular a discretização.")
        st.stop()

    st.subheader("Passo 1: Definir a aproximação de $s$")

    if metodo.startswith("1"):
        metodo_nome = "Euler (Forward)"
        s_latex, inv_s_latex = r"\frac{z - 1}{T}", r"\frac{T}{z - 1}"
        p_latex = rf"P(z) = K_p = {Kp:.4f}"
        i_latex = rf"I(z) \approx K_i \frac{{T}}{{z - 1}} = {Ki:.4f} \cdot \frac{{{T:.4f}}}{{z - 1}}" if abs(Ki) > 1e-12 else None
        d_latex = rf"D(z) \approx K_d \frac{{z - 1}}{{T}} = \frac{{{Kd:.4f}}}{{{T:.4f}}} (z - 1)" if abs(Kd) > 1e-12 else None

        c2, c1, c0 = Kd / T, Kp - 2.0 * Kd / T, -Kp + Ki * T + Kd / T
        num_coeffs, den_coeffs = [c2, c1, c0], [1.0, -1.0]
        den_comum_latex, nz_latex = "(z - 1)", r"K_p(z - 1) + K_i T + \frac{K_d}{T}(z - 1)^2"

    elif metodo.startswith("2"):
        metodo_nome = "Euler (Backward)"
        s_latex, inv_s_latex = r"\frac{z - 1}{Tz}", r"\frac{Tz}{z - 1}"
        p_latex = rf"P(z) = K_p = {Kp:.4f}"
        i_latex = rf"I(z) \approx K_i \frac{{Tz}}{{z - 1}} = {Ki:.4f} \cdot \frac{{{T:.4f} z}}{{z - 1}}" if abs(Ki) > 1e-12 else None
        d_latex = rf"D(z) \approx K_d \frac{{z - 1}}{{Tz}} = \frac{{{Kd:.4f}}}{{{T:.4f}}} \frac{{z - 1}}{{z}}" if abs(Kd) > 1e-12 else None

        c2, c1, c0 = Kp + Ki * T + Kd / T, -Kp - 2.0 * Kd / T, Kd / T
        num_coeffs, den_coeffs = [c2, c1, c0], [1.0, -1.0, 0.0]
        den_comum_latex, nz_latex = "z(z - 1)", r"K_p z(z - 1) + K_i T z^2 + \frac{K_d}{T}(z - 1)^2"

    else:
        metodo_nome = "Tustin (Bilinear)"
        s_latex, inv_s_latex = r"\frac{2}{T} \frac{z - 1}{z + 1}", r"\frac{T}{2} \frac{z + 1}{z - 1}"
        p_latex = rf"P(z) = K_p = {Kp:.4f}"
        i_latex = rf"I(z) \approx K_i \frac{{T}}{{2}} \frac{{z + 1}}{{z - 1}} = {Ki:.4f} \cdot {T/2.0:.4f} \frac{{z + 1}}{{z - 1}}" if abs(Ki) > 1e-12 else None
        d_latex = rf"D(z) \approx K_d \frac{{2}}{{T}} \frac{{z - 1}}{{z + 1}} = {Kd:.4f} \cdot {2.0/T:.4f} \frac{{z - 1}}{{z + 1}}" if abs(Kd) > 1e-12 else None

        c2, c1, c0 = Kp + Ki * T / 2.0 + 2.0 * Kd / T, Ki * T - 4.0 * Kd / T, -Kp + Ki * T / 2.0 + 2.0 * Kd / T
        num_coeffs, den_coeffs = [c2, c1, c0], [1.0, 0.0, -1.0]
        den_comum_latex, nz_latex = "(z - 1)(z + 1) = z^2 - 1", r"K_p(z^2 - 1) + K_i \frac{T}{2} (z + 1)^2 + K_d \frac{2}{T} (z - 1)^2"

    st.markdown(f"**Método Escolhido:** {metodo_nome}")
    st.latex(rf"s \approx {s_latex} \quad \text{{e}} \quad \frac{{1}}{{s}} \approx {inv_s_latex}")

    st.subheader("Passo 2: Discretizar os termos do controlador")
    st.latex(p_latex)
    if i_latex: st.latex(i_latex)
    if d_latex: st.latex(d_latex)

    st.subheader("Passo 3: Combinar os termos em $N(z)/D(z)$")
    st.markdown("Para encontrar a função de transferência final do controlador discreto, somamos as três ações isoladas e tiramos o mínimo múltiplo comum:")
    st.latex(r"G_c(z) = P(z) + I(z) + D(z) = \frac{N(z)}{D(z)}")
    st.markdown(f"**Denominador comum:** ${den_comum_latex}$")
    st.markdown("**Numerador $N(z)$ expandido:**")
    st.latex(rf"N(z) = {nz_latex}")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**[3.1] Coeficientes do Numerador**")
        st.markdown("$N(z) = c_2 z^2 + c_1 z + c_0$")
        
        if metodo.startswith("1"):
            st.latex(rf"c_2 = \frac{{K_d}}{{T}} = {c2:.4f}")
            st.latex(rf"c_1 = K_p - \frac{{2K_d}}{{T}} = {c1:.4f}")
            st.latex(rf"c_0 = -K_p + K_i T + \frac{{K_d}}{{T}} = {c0:.4f}")
        elif metodo.startswith("2"):
            st.latex(rf"c_2 = K_p + K_i T + \frac{{K_d}}{{T}} = {c2:.4f}")
            st.latex(rf"c_1 = -K_p - \frac{{2K_d}}{{T}} = {c1:.4f}")
            st.latex(rf"c_0 = \frac{{K_d}}{{T}} = {c0:.4f}")
        else:
            st.latex(rf"c_2 = K_p + K_i\frac{{T}}{{2}} + \frac{{2K_d}}{{T}} = {c2:.4f}")
            st.latex(rf"c_1 = K_i T - \frac{{4K_d}}{{T}} = {c1:.4f}")
            st.latex(rf"c_0 = -K_p + K_i\frac{{T}}{{2}} + \frac{{2K_d}}{{T}} = {c0:.4f}")
        
    with col4:
        st.markdown("**[3.2] Coeficientes do Denominador**")
        if metodo.startswith("1"):
            st.markdown("$D(z) = d_1 z + d_0$")
            st.latex(r"d_1 = 1.0000 \quad d_0 = -1.0000")
        elif metodo.startswith("2"):
            st.markdown("$D(z) = d_2 z^2 + d_1 z + d_0$")
            st.latex(r"d_2 = 1.0000 \quad d_1 = -1.0000 \quad d_0 = 0.0000")
        else:
            st.markdown("$D(z) = d_2 z^2 + d_1 z + d_0$")
            st.latex(r"d_2 = 1.0000 \quad d_1 = 0.0000 \quad d_0 = -1.0000")

    st.subheader("Passo 4: Controlador Discretizado")
    num_terms = []
    if abs(c2) > 1e-8: num_terms.append(f"{c2:.4f}z^2")
    if abs(c1) > 1e-8: num_terms.append(f"{'+' if c1 > 0 else '-'} {abs(c1):.4f}z")
    if abs(c0) > 1e-8: num_terms.append(f"{'+' if c0 > 0 else '-'} {abs(c0):.4f}")
        
    num_latex = " ".join(num_terms) if num_terms else "0"
    if num_latex.startswith("+ "): num_latex = num_latex[2:]
    
    den_latex = "z - 1" if metodo.startswith("1") else ("z^2 - z" if metodo.startswith("2") else "z^2 - 1")

    st.latex(rf"G_c(z) = \frac{{{num_latex}}}{{{den_latex}}}")
    Gc_z = ct.tf(num_coeffs, den_coeffs, dt=T)
    st.success(f"**Sistema discretizado com período de amostragem:** $dt = {Gc_z.dt}$ s")

def resolver_pd():
    st.markdown("### RESOLUÇÃO - PROJETO DE CONTROLADOR PD")
    st.markdown("**Insira os coeficientes dos polinômios separados por espaço (ex: 4 16):**")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1: num_g_str = st.text_input("Numerador de G(s)", value="", placeholder="Ex: 4 16")
    with col_g2: den_g_str = st.text_input("Denominador de G(s)", value="", placeholder="Ex: 1 4 4 0")
        
    col_h1, col_h2 = st.columns(2)
    with col_h1: num_h_str = st.text_input("Numerador de H(s)", value="", placeholder="Ex: 1")
    with col_h2: den_h_str = st.text_input("Denominador de H(s)", value="", placeholder="Ex: 1")

    if not (num_g_str and den_g_str and num_h_str and den_h_str):
        st.info("Preencha os numeradores e denominadores do sistema para continuar.")
        st.stop()

    num_g, den_g = [float(x) for x in num_g_str.split()], [float(x) for x in den_g_str.split()]
    num_h, den_h = [float(x) for x in num_h_str.split()], [float(x) for x in den_h_str.split()]

    G_s, H_s = ct.tf(num_g, den_g), ct.tf(num_h, den_h)
    GH = ct.series(G_s, H_s)
    polos, zeros = GH.poles(), GH.zeros()

    st.markdown("**Componentes do Sistema:**")
    st.latex(rf"G(s) = {_tf_to_latex(G_s, 's')}")
    st.latex(rf"H(s) = {_tf_to_latex(H_s, 's')}")
    st.latex(rf"GH(s) = {_tf_to_latex(GH, 's')}")

    if any(np.real(p) > 0 for p in G_s.poles()):
        st.warning("**Atenção:** A planta possui polo no semiplano direito, sendo **INSTÁVEL** em malha aberta.")

    st.markdown("---")
    modo = st.radio(
        "Como deseja informar o polo dominante desejado ($s_d$)?",
        options=["1) A partir de Overshoot (Mp) e Tempo de Acomodação (ts)", "2) A partir do Fator de Amortecimento (ζ) e Frequência Natural (ωn)", "3) Informar s_d diretamente"]
    )

    if modo.startswith("2"):
        col1, col2 = st.columns(2)
        with col1: zeta = st.number_input("Fator de Amortecimento (ζ)", value=None, format="%.4f", placeholder="Ex: 0.7")
        with col2: omega_n = st.number_input("Frequência Natural (ωn) [rad/s]", value=None, format="%.4f", placeholder="Ex: 0.5")

        if zeta is None or omega_n is None:
            st.warning("Preencha os valores de ζ e ωn para prosseguir com o cálculo.")
            st.stop()

        st.info(f"**Requisitos:** Fator de Amortecimento ($\zeta$) = {zeta:.4f} e Frequência Natural ($\omega_n$) = {omega_n:.4f} rad/s.")
        sigma, omega_d, s_d = _calc_sd_from_zeta_omega(zeta, omega_n)
        st.subheader("Passo 1: Determinar a Localização do Polo Dominante Desejado ($s_d$)")
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    elif modo.startswith("3"):
        st.markdown("Informe as coordenadas do polo desejado ($s_d = a + bj$):")
        col1, col2 = st.columns(2)
        with col1: real_part = st.number_input("Parte Real (a)", value=None, format="%.4f")
        with col2: imag_part = st.number_input("Parte Imaginária (+jb)", value=None, format="%.4f")
            
        if real_part is None or imag_part is None:
            st.warning("Preencha a parte real e imaginária para prosseguir.")
            st.stop()
            
        s_d = complex(real_part, imag_part)
        st.subheader("Passo 1: Polo de Malha Fechada Desejado")
        sinal = "+" if imag_part >= 0 else "-"
        st.latex(rf"s_d = {real_part:.4f} {sinal} j{abs(imag_part):.4f}")

    else:
        col1, col2, col3 = st.columns(3)
        with col1: mp_max = st.number_input("Mp max (%)", value=None, format="%.2f", placeholder="Ex: 10")
        with col2: ts_req = st.number_input("ts (s)", value=None, format="%.2f", placeholder="Ex: 4")
        with col3: criterio_ts = st.number_input("Critério ts (%)", value=None, format="%.2f", placeholder="Ex: 5")

        if mp_max is None or ts_req is None or criterio_ts is None:
            st.warning("Preencha os requisitos de Overshoot e Tempo de Acomodação para prosseguir.")
            st.stop()

        zeta, sigma, omega_n, omega_d, s_d = _calc_sd_from_mp_ts(mp_max, ts_req, criterio_ts)
        st.info(f"**Requisitos:** $M_p \le {mp_max}\%$ e $t_s({criterio_ts}\%) < {ts_req}\text{{ s}}$.")
        st.subheader("Passo 1: Traduzir os Requisitos de Desempenho para um Polo Dominante ($s_d$)")
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    sigma, omega_d = -s_d.real, abs(s_d.imag)

    st.subheader("Passo 2: Aplicar a Condição de Ângulo")
    ang_total = _print_contribuicoes_angulo(zeros, polos, s_d)
    phi_c_deg = _normalize_angle_deg(-180.0 - ang_total)
    
    st.markdown("**O ângulo que o zero do controlador PD precisa fornecer ($\phi_c$) é:**")
    st.latex(rf"\phi_c = -180^\circ - ({ang_total:.2f}^\circ) = {phi_c_deg:.2f}^\circ")

    zc = _calc_zc(sigma, omega_d, phi_c_deg)
    st.latex(rf"z_c = \sigma + \frac{{\omega_d}}{{\tan(\phi_c)}} = {zc:.4f}")
    st.success(f"Portanto, o zero do controlador está em **s = {-zc:.4f}**")

    st.subheader("Passo 3: Aplicar a Condição de Módulo")
    st.latex(r"\left| K_c (s_d + z_c) \right| \cdot \left| G(s_d)H(s_d) \right| = 1")

    mags_polos = [abs(s_d - p) for p in polos]
    mags_zeros = [abs(s_d - z) for z in zeros]
    mag_zc = abs(s_d + zc)

    prod_polos = np.prod(mags_polos)
    prod_zeros = np.prod(mags_zeros) if len(mags_zeros) > 0 else 1.0
    K_planta = _leading_gain(GH)
    denominador_total = K_planta * prod_zeros * mag_zc
    Kc = prod_polos / denominador_total

    st.latex(r"K_c = \frac{\prod |s_d - p_i|}{K_{planta} \cdot \prod |s_d - z_i| \cdot |s_d + z_c|}")
    st.latex(rf"K_c = \frac{{{prod_polos:.4f}}}{{{denominador_total:.4f}}} = {Kc:.4f}")

    st.subheader("Passo 4: Determinar os Parâmetros Finais $K_p$ e $K_d$")
    Kd, Kp = Kc, Kc * zc
    st.latex(rf"K_d = K_c = {Kd:.4f} \quad \text{{e}} \quad K_p = K_c \cdot z_c = {Kp:.4f}")

    st.markdown("---")
    st.markdown("### Conclusão")
    st.latex(rf"G_c(s) = {Kp:.4f} + {Kd:.4f}s")
    st.success(f"**Parâmetros Finais do Controlador:** \n\n $K_p = {Kp:.4f}$ \n\n $K_d = {Kd:.4f}$")

    _discretizar_controlador("PD", Kp, 0.0, Kd)

def resolver_pi():
    st.markdown("### RESOLUÇÃO - PROJETO DE CONTROLADOR PI")
    st.markdown("**Insira os coeficientes dos polinômios separados por espaço (ex: 5 25 20):**")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1: num_g_str = st.text_input("Numerador de G(s)", value="", placeholder="Ex: 5 25 20")
    with col_g2: den_g_str = st.text_input("Denominador de G(s)", value="", placeholder="Ex: 1 4 4")
        
    col_h1, col_h2 = st.columns(2)
    with col_h1: num_h_str = st.text_input("Numerador de H(s)", value="", placeholder="Ex: 0.2")
    with col_h2: den_h_str = st.text_input("Denominador de H(s)", value="", placeholder="Ex: 1 1")

    if not (num_g_str and den_g_str and num_h_str and den_h_str):
        st.info("Preencha os numeradores e denominadores do sistema para continuar.")
        st.stop()

    num_g, den_g = [float(x) for x in num_g_str.split()], [float(x) for x in den_g_str.split()]
    num_h, den_h = [float(x) for x in num_h_str.split()], [float(x) for x in den_h_str.split()]

    G_s, H_s = ct.tf(num_g, den_g), ct.tf(num_h, den_h)

    st.markdown("---")
    modo = st.radio(
        "Como deseja informar o polo dominante desejado ($s_d$)?",
        options=["1) A partir de Overshoot (Mp) e Tempo de Acomodação (ts)", "2) A partir do Fator de Amortecimento (ζ) e Frequência Natural (ωn)", "3) Informar s_d diretamente"]
    )

    if modo.startswith("2"):
        col1, col2 = st.columns(2)
        with col1: zeta = st.number_input("Fator de Amortecimento (ζ)", value=None, format="%.4f", placeholder="Ex: 0.7")
        with col2: omega_n = st.number_input("Frequência Natural (ωn) [rad/s]", value=None, format="%.4f", placeholder="Ex: 0.5")

        if zeta is None or omega_n is None:
            st.warning("Preencha os valores de ζ e ωn para prosseguir com o cálculo.")
            st.stop()
        
        st.info(f"**Requisitos:** Fator de Amortecimento ($\zeta$) = {zeta:.4f} e Frequência Natural ($\omega_n$) = {omega_n:.4f} rad/s.")
        sigma, omega_d, s_d = _calc_sd_from_zeta_omega(zeta, omega_n)
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    elif modo.startswith("3"):
        st.markdown("Informe as coordenadas do polo desejado ($s_d = a + bj$):")
        col1, col2 = st.columns(2)
        with col1: real_part = st.number_input("Parte Real (a)", value=None, format="%.4f", placeholder="Ex: -4.0")
        with col2: imag_part = st.number_input("Parte Imaginária (+jb)", value=None, format="%.4f", placeholder="Ex: 4.0")
            
        if real_part is None or imag_part is None:
            st.warning("Preencha a parte real e imaginária para prosseguir.")
            st.stop()
            
        s_d = complex(real_part, imag_part)
        sinal = "+" if imag_part >= 0 else "-"
        st.latex(rf"s_d = {real_part:.4f} {sinal} j{abs(imag_part):.4f}")

    else:
        col1, col2, col3 = st.columns(3)
        with col1: mp_max = st.number_input("Mp max (%)", value=None, format="%.2f", placeholder="Ex: 10")
        with col2: ts_req = st.number_input("ts (s)", value=None, format="%.2f", placeholder="Ex: 4")
        with col3: criterio_ts = st.number_input("Critério ts (%)", value=None, format="%.2f", placeholder="Ex: 5")

        if mp_max is None or ts_req is None or criterio_ts is None:
            st.warning("Preencha os requisitos de Overshoot e Tempo de Acomodação para prosseguir.")
            st.stop()

        zeta, sigma, omega_n, omega_d, s_d = _calc_sd_from_mp_ts(mp_max, ts_req, criterio_ts)
        st.info(f"**Requisitos:** $M_p \le {mp_max}\%$ e $t_s({criterio_ts}\%) < {ts_req}\text{{ s}}$.")
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    st.markdown("---")
    st.subheader("Passo 1: Determinar a Função de Transferência de Malha Aberta")

    cancel = _find_cancellations(G_s.zeros(), H_s.poles())
    if cancel: st.info(f"**Nota:** Ao combinar $G(s)$ e $H(s)$, ocorre um cancelamento polo-zero em $s = {_fmt_num(cancel[0])}$.")

    GH = ct.series(G_s, H_s)
    sistema_base = ct.series(GH, ct.tf([1], [1, 0]))
    polos_base, zeros_base = sistema_base.poles(), sistema_base.zeros()
    
    st.markdown("O controlador PI adiciona a ação integral (um polo na origem) ao sistema. Este é o sistema base $G_{base}(s)$ que será usado para a análise de ângulo e módulo:")
    st.latex(rf"G_{{base}}(s) = \frac{{G(s)H(s)}}{{s}} = {_tf_to_latex(sistema_base, 's')}")

    sigma, omega_d = -s_d.real, abs(s_d.imag)

    st.subheader("Passo 2: Aplicar a Condição de Ângulo")
    ang_total = _print_contribuicoes_angulo(zeros_base, polos_base, s_d)
    phi_c_deg = _normalize_angle_deg(-180.0 - ang_total)
    
    st.markdown("**O ângulo que o zero do controlador PI precisa fornecer ($\phi_c$) é:**")
    st.latex(rf"\phi_c = -180^\circ - ({ang_total:.2f}^\circ) = {phi_c_deg:.2f}^\circ")

    zc = _calc_zc(sigma, omega_d, phi_c_deg)
    st.latex(rf"z_c = \sigma + \frac{{\omega_d}}{{\tan(\phi_c)}} = {zc:.4f}")
    st.success(f"Portanto, o zero do controlador PI está em **s = {-zc:.4f}**")

    if abs(zc) < 1e-9:
        st.warning("**Nota Importante:** Um $z_c \approx 0$ indica o cancelamento do polo na origem. O LGR do sistema base já passava pelo ponto desejado e a ação integral foi anulada (comportando-se apenas como P).")

    st.subheader("Passo 3: Aplicar a Condição de Módulo")
    st.latex(r"\left| K_c (s_d + z_c) \right| \cdot \left| G_{base}(s_d) \right| = 1")

    mags_polos = [abs(s_d - p) for p in polos_base]
    mags_zeros = [abs(s_d - z) for z in zeros_base]
    mag_zc = abs(s_d + zc)

    prod_polos = np.prod(mags_polos)
    prod_zeros = np.prod(mags_zeros) if len(mags_zeros) > 0 else 1.0
    K_sistema = _leading_gain(GH)
    denominador_total = K_sistema * prod_zeros * mag_zc
    Kc = prod_polos / denominador_total

    st.latex(r"K_c = \frac{\prod |s_d - p_i|}{K_{sistema} \cdot \prod |s_d - z_i| \cdot |s_d + z_c|}")
    st.latex(rf"K_c = \frac{{{prod_polos:.4f}}}{{{denominador_total:.4f}}} = {Kc:.4f}")

    st.subheader("Passo 4: Determinar os Parâmetros Finais $K_p$ e $K_i$")
    Kp, Ki = Kc, Kc * zc
    st.latex(rf"K_p = K_c = {Kp:.4f} \quad \text{{e}} \quad K_i = K_c \cdot z_c = {Ki:.4f}")

    st.markdown("---")
    st.markdown("### Conclusão")
    st.latex(rf"G_c(s) = {Kp:.4f} + \frac{{{Ki:.4f}}}{{s}}")
    st.success(f"**Parâmetros Finais do Controlador:** \n\n $K_p = {Kp:.4f}$ \n\n $K_i = {Ki:.4f}$")

    _discretizar_controlador("PI", Kp, Ki, 0.0)

def resolver_pid():
    st.markdown("### RESOLUÇÃO - PROJETO DE CONTROLADOR PID")
    st.markdown("**Insira os coeficientes dos polinômios separados por espaço:**")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1: num_g_str = st.text_input("Numerador de G(s)", value="", placeholder="Ex: 5")
    with col_g2: den_g_str = st.text_input("Denominador de G(s)", value="", placeholder="Ex: 1 12 22 20")
        
    col_h1, col_h2 = st.columns(2)
    with col_h1: num_h_str = st.text_input("Numerador de H(s)", value="", placeholder="Ex: 0.4")
    with col_h2: den_h_str = st.text_input("Denominador de H(s)", value="", placeholder="Ex: 1")

    if not (num_g_str and den_g_str and num_h_str and den_h_str):
        st.info("Preencha os numeradores e denominadores do sistema para continuar.")
        st.stop()

    num_g, den_g = [float(x) for x in num_g_str.split()], [float(x) for x in den_g_str.split()]
    num_h, den_h = [float(x) for x in num_h_str.split()], [float(x) for x in den_h_str.split()]

    G_s, H_s = ct.tf(num_g, den_g), ct.tf(num_h, den_h)

    st.markdown("---")
    modo = st.radio(
        "Como deseja informar o polo dominante desejado ($s_d$)?",
        options=["1) A partir de Overshoot (Mp) e Tempo de Acomodação (ts)", "2) A partir do Fator de Amortecimento (ζ) e Frequência Natural (ωn)", "3) Informar s_d diretamente"]
    )

    if modo.startswith("2"):
        col1, col2 = st.columns(2)
        with col1: zeta = st.number_input("Fator de Amortecimento (ζ)", value=None, format="%.4f", placeholder="Digite ζ")
        with col2: omega_n = st.number_input("Frequência Natural (ωn) [rad/s]", value=None, format="%.4f", placeholder="Digite ωn")

        if zeta is None or omega_n is None:
            st.warning("Preencha os valores de ζ e ωn para prosseguir com o cálculo.")
            st.stop()

        st.info(f"**Requisitos:** Fator de Amortecimento ($\zeta$) = {zeta:.4f} e Frequência Natural ($\omega_n$) = {omega_n:.4f} rad/s.")
        sigma, omega_d, s_d = _calc_sd_from_zeta_omega(zeta, omega_n)
        st.subheader("Passo 1: Determinar a Localização do Polo Dominante Desejado ($s_d$)")
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    elif modo.startswith("3"):
        st.markdown("Informe as coordenadas do polo desejado ($s_d = a + bj$):")
        col1, col2 = st.columns(2)
        with col1: real_part = st.number_input("Parte Real (a)", value=None, format="%.4f", placeholder="Ex: -0.75")
        with col2: imag_part = st.number_input("Parte Imaginária (+jb)", value=None, format="%.4f", placeholder="Ex: 1.03")
            
        if real_part is None or imag_part is None:
            st.warning("Preencha a parte real e imaginária para prosseguir.")
            st.stop()
            
        s_d = complex(real_part, imag_part)
        st.subheader("Passo 1: Polo de Malha Fechada Desejado")
        sinal = "+" if imag_part >= 0 else "-"
        st.latex(rf"s_d = {real_part:.4f} {sinal} j{abs(imag_part):.4f}")

    else:
        col1, col2, col3 = st.columns(3)
        with col1: mp_max = st.number_input("Mp max (%)", value=None, format="%.2f", placeholder="Ex: 20")
        with col2: ts_req = st.number_input("ts (s)", value=None, format="%.2f", placeholder="Ex: 5")
        with col3: criterio_ts = st.number_input("Critério ts (%)", value=None, format="%.2f", placeholder="Ex: 2")

        if mp_max is None or ts_req is None or criterio_ts is None:
            st.warning("Preencha os requisitos de Overshoot e Tempo de Acomodação para prosseguir.")
            st.stop()

        zeta, sigma, omega_n, omega_d, s_d = _calc_sd_from_mp_ts(mp_max, ts_req, criterio_ts)
        st.info(f"**Requisitos:** $M_p \le {mp_max}\%$ e $t_s({criterio_ts}\%) < {ts_req}\text{{ s}}$, com zeros do PID reais e iguais.")
        st.subheader("Passo 1: Traduzir os Requisitos de Desempenho para um Polo Dominante ($s_d$)")
        st.latex(rf"s_d = -\sigma + j\omega_d = -{sigma:.4f} + j{omega_d:.4f}")

    GH = ct.series(G_s, H_s)
    sistema_base = ct.series(GH, ct.tf([1], [1, 0]))
    polos_base, zeros_base = sistema_base.poles(), sistema_base.zeros()

    st.subheader("Passo 2: Aplicar a Condição de Ângulo para encontrar o zero duplo do controlador ($-z_c$)")
    st.markdown("O sistema base é a planta $G(s)H(s)$ associada ao polo do PID na origem (ação integral).")
    st.latex(rf"G_{{base}}(s) = {get_zpk_latex(sistema_base)}")

    ang_total = _print_contribuicoes_angulo(zeros_base, polos_base, s_d)

    phi_total_zeros = -180.0 - ang_total
    phi_c_deg = _normalize_angle_deg(phi_total_zeros / 2.0)

    st.markdown("**Cálculo do Ângulo de Compensação dos Zeros do PID**")
    st.latex(rf"\phi_{{total}} = -180^\circ - ({ang_total:.2f}^\circ) = {phi_total_zeros:.2f}^\circ")
    st.latex(rf"\phi_c = \frac{{\phi_{{total}}}}{{2}} = \frac{{{phi_total_zeros:.2f}^\circ}}{{2}} = {phi_c_deg:.2f}^\circ")

    sigma, omega_d = -s_d.real, abs(s_d.imag)
    zc = _calc_zc(sigma, omega_d, phi_c_deg)
    st.latex(rf"z_c = \sigma + \frac{{\omega_d}}{{\tan(\phi_c)}} = {zc:.4f}")
    st.success(f"Portanto, o controlador tem um zero duplo em **s = {-zc:.4f}**")

    mags_polos = [abs(s_d - p) for p in polos_base]
    mags_zeros = [abs(s_d - z) for z in zeros_base]
    mag_zc = abs(s_d + zc)

    prod_polos = np.prod(mags_polos)
    prod_zeros = np.prod(mags_zeros) if len(mags_zeros) > 0 else 1.0
    K_sistema = _leading_gain(GH)
    denominador_total = K_sistema * prod_zeros * (mag_zc ** 2)
    Kc = prod_polos / denominador_total

    st.subheader("Passo 3: Aplicar a Condição de Módulo")
    st.latex(r"\left| K_c \frac{(s_d+z_c)^2}{s_d} G(s_d)H(s_d) \right|_{s = s_d} = 1")
    st.latex(r"K_c = \frac{\prod |s_d - p_i|}{K_{sistema} \cdot \prod |s_d - z_i| \cdot |s_d + z_c|^2}")
    st.latex(rf"K_c = \frac{{{prod_polos:.4f}}}{{{denominador_total:.4f}}} = {Kc:.4f}")
    st.success(f"Ganho do Controlador PID encontrado: **Kc = {Kc:.4f}**")

    Kd, Kp, Ki = Kc, 2 * Kc * zc, Kc * (zc ** 2)

    st.subheader("Passo 4: Determinar os Parâmetros Finais $K_p$, $K_i$ e $K_d$")
    st.latex(r"G_c(s) = \frac{K_c(s+z_c)^2}{s} = \frac{K_d s^2 + K_p s + K_i}{s}")
    st.latex(rf"K_d = {Kd:.4f} \quad K_p = {Kp:.4f} \quad K_i = {Ki:.4f}")

    st.markdown("---")
    st.markdown("### Conclusão")
    st.latex(rf"G_c(s) = {Kp:.4f} + \frac{{{Ki:.4f}}}{{s}} + {Kd:.4f}s")
    st.success(f"**Parâmetros Finais do Controlador:** \n\n $K_p = {Kp:.4f}$ \n\n $K_i = {Ki:.4f}$ \n\n $K_d = {Kd:.4f}$")

    _discretizar_controlador("PID", Kp, Ki, Kd)

def resolver_gc_generico():
    st.markdown("### RESOLUÇÃO - CONTROLADOR GENÉRICO $G_c(s) = \\frac{a}{s+b}$")
    st.markdown("**Insira os coeficientes dos polinômios separados por espaço:**")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1: num_g_str = st.text_input("Numerador de G(s)", value="", placeholder="Ex: 2 2")
    with col_g2: den_g_str = st.text_input("Denominador de G(s)", value="", placeholder="Ex: 1 2 2")
        
    col_h1, col_h2 = st.columns(2)
    with col_h1: num_h_str = st.text_input("Numerador de H(s)", value="", placeholder="Ex: 1 3")
    with col_h2: den_h_str = st.text_input("Denominador de H(s)", value="", placeholder="Ex: 1 5")

    if not (num_g_str and den_g_str and num_h_str and den_h_str):
        st.info("Preencha os numeradores e denominadores do sistema para continuar.")
        st.stop()

    num_g, den_g = [float(x) for x in num_g_str.split()], [float(x) for x in den_g_str.split()]
    num_h, den_h = [float(x) for x in num_h_str.split()], [float(x) for x in den_h_str.split()]

    G_s, H_s = ct.tf(num_g, den_g), ct.tf(num_h, den_h)

    st.markdown("---")
    modo = st.radio(
        "Como deseja informar o polo dominante desejado ($s_d$)?",
        options=["1) A partir de Overshoot (Mp) e Tempo de Acomodação (ts)", "2) A partir do Fator de Amortecimento (ζ) e Frequência Natural (ωn)", "3) Informar s_d diretamente"]
    )

    if modo.startswith("2"):
        col1, col2 = st.columns(2)
        with col1: zeta = st.number_input("Fator de Amortecimento (ζ)", value=None, format="%.4f")
        with col2: omega_n = st.number_input("Frequência Natural (ωn) [rad/s]", value=None, format="%.4f")

        if zeta is None or omega_n is None:
            st.warning("Preencha os valores de ζ e ωn para prosseguir com o cálculo.")
            st.stop()

        st.info(f"**Requisitos:** Fator de Amortecimento ($\zeta$) = {zeta:.4f} e Frequência Natural ($\omega_n$) = {omega_n:.4f} rad/s.")
        sigma, omega_d, s_d = _calc_sd_from_zeta_omega(zeta, omega_n)
        
    elif modo.startswith("3"):
        st.markdown("Informe as coordenadas do polo desejado ($s_d = a + bj$):")
        col1, col2 = st.columns(2)
        with col1: real_part = st.number_input("Parte Real (a)", value=None, format="%.4f")
        with col2: imag_part = st.number_input("Parte Imaginária (+jb)", value=None, format="%.4f")
            
        if real_part is None or imag_part is None:
            st.warning("Preencha a parte real e imaginária para prosseguir.")
            st.stop()
            
        s_d = complex(real_part, imag_part)

    else:
        col1, col2, col3 = st.columns(3)
        with col1: mp_max = st.number_input("Mp max (%)", value=None, format="%.2f")
        with col2: ts_req = st.number_input("ts (s)", value=None, format="%.2f")
        with col3: criterio_ts = st.number_input("Critério ts (%)", value=None, format="%.2f")

        if mp_max is None or ts_req is None or criterio_ts is None:
            st.warning("Preencha os requisitos de Overshoot e Tempo de Acomodação para prosseguir.")
            st.stop()

        zeta, sigma, omega_n, omega_d, s_d = _calc_sd_from_mp_ts(mp_max, ts_req, criterio_ts)
        st.info(f"**Requisitos:** $M_p \le {mp_max}\%$ e $t_s({criterio_ts}\%) < {ts_req}\text{{ s}}$.")

    st.markdown("---")
    st.markdown("**Componentes do Sistema:**")
    st.latex(rf"G(s) = {_tf_to_latex(G_s, 's')}")
    st.latex(rf"H(s) = {_tf_to_latex(H_s, 's')}")
    
    sinal_sd = "+" if s_d.imag >= 0 else "-"
    st.markdown(f"**Estrutura do controlador:** $G_c(s) = \\frac{{a}}{{s+b}}$")
    st.markdown(f"**Polo desejado:** $s_d = {s_d.real:.4f} {sinal_sd} j{abs(s_d.imag):.4f}$")

    st.subheader("Passo 1: Apresentação do Método de Resolução")
    st.latex(r"1 + G_c(s_d)G(s_d)H(s_d) = 0 \implies b + a \cdot [G(s_d)H(s_d)] = -s_d")

    st.subheader("Passo 2: Calcular o valor do sistema $G(s)H(s)$ no ponto $s_d$")
    valor_GH_em_sd = ct.evalfr(G_s, s_d) * ct.evalfr(H_s, s_d)

    def format_complex_latex(c):
        if abs(c.imag) < 1e-12: return f"{c.real:.4f}"
        sinal = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sinal} j{abs(c.imag):.4f}"

    st.latex(rf"G(s_d)H(s_d) = {format_complex_latex(valor_GH_em_sd)}")

    st.subheader("Passo 3: Resolver o Sistema de Equações para $a$ e $b$")
    gh_real, gh_imag = valor_GH_em_sd.real, valor_GH_em_sd.imag
    sd_real, sd_imag = s_d.real, s_d.imag

    st.latex(rf"\text{{Eq. Real: }} \quad b + a \cdot ({gh_real:.4f}) = -({sd_real:.4f})")
    st.latex(rf"\text{{Eq. Imaginária: }} \quad a \cdot ({gh_imag:.4f}) = -({sd_imag:.4f})")

    if abs(gh_imag) < 1e-12:
        st.error("**Erro:** A parte imaginária de $G(s_d)H(s_d)$ é zero, não sendo possível determinar '$a$'.")
        st.stop()

    a = -sd_imag / gh_imag
    st.latex(rf"a = \frac{{-\text{{Im}}(s_d)}}{{\text{{Im}}(G(s_d)H(s_d))}} = {a:.4f}")

    b = -sd_real - (a * gh_real)
    st.latex(rf"b = -\text{{Re}}(s_d) - a \cdot \text{{Re}}(G(s_d)H(s_d)) = {b:.4f}")

    Gc_s = ct.tf([a], [1, b])
    st.success(f"**Conclusão - Controlador Projetado:** \n\n $G_c(s) = \\frac{{{a:.4f}}}{{s + ({b:.4f})}}$")

    st.markdown("---")
    st.header("Discretização do Sistema Completo $L(s)$")
    
    if not st.checkbox("Deseja discretizar o sistema completo L(s) = Gc(s)G(s)H(s)?"):
        return

    col_t, col_metodo = st.columns(2)
    with col_t: T = st.number_input("Período de amostragem T (s)", value=None, format="%.4f", placeholder="Ex: 1.0")
    with col_metodo:
        metodo = st.selectbox("Método de discretização:", options=["1) Euler (Forward)", "2) Euler (Backward)", "3) Tustin (Bilinear)"])

    if T is None: st.stop()

    L_s = ct.series(Gc_s, G_s, H_s)
    num_s, den_s = L_s.num[0][0], L_s.den[0][0]
    
    num_degree, den_degree = len(num_s) - 1, len(den_s) - 1
    grau_max = max(num_degree, den_degree)
    z_poly, q_poly = np.poly1d([1.0, 0.0]), np.poly1d([1.0, 1.0])

    if metodo.startswith("2"):
        num_z, den_z = _poly_subst_backward(num_s, T), _poly_subst_backward(den_s, T)
        diff = den_degree - num_degree
        if diff > 0: num_z = _poly_mul_power(num_z, z_poly, diff)
        elif diff < 0: den_z = _poly_mul_power(den_z, z_poly, -diff)
            
    elif metodo.startswith("3"):
        num_z, den_z = _poly_subst_tustin(num_s, T), _poly_subst_tustin(den_s, T)
        diff = den_degree - num_degree
        if diff > 0: num_z = _poly_mul_power(num_z, q_poly, diff)
        elif diff < 0: den_z = _poly_mul_power(den_z, q_poly, -diff)
            
    else:
        num_z, den_z = _poly_subst_euler(num_s, T), _poly_subst_euler(den_s, T)

    st.subheader("Sistema discretizado $L(z)$")
    try:
        st.latex(rf"L(z) = \frac{{{_poly_to_str(num_z, 'z')}}}{{{_poly_to_str(den_z, 'z')}}}")
    except: pass
        
    L_z = ct.tf(num_z, den_z, dt=T)
    st.success(f"**Amostragem concluída:** $dt = {L_z.dt}$ s")


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

opcoes_menu = {
    "1) PD": "Controlador PD (Proporcional-Derivativo)",
    "2) PI": "Controlador PI (Proporcional-Integral)",
    "3) PID": "Controlador PID (Proporcional-Integral-Derivativo)",
    "4) Gc(s) genérico": "Controlador Genérico Gc(s) = a/(s+b)"
}

st.sidebar.title("Menu de Controladores")
st.sidebar.markdown("Selecione o tipo de controlador que deseja projetar no menu abaixo:")
opcao = st.sidebar.radio("Escolha uma opção:", list(opcoes_menu.keys()))

st.sidebar.markdown("---")
if st.sidebar.button("Limpar Todos os Dados", use_container_width=True):
    st.session_state.clear()
    st.rerun()

controlador_nome = opcoes_menu[opcao]

st.markdown(
    f"""
    <div class="hero" style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
      <div class="hero-kicker" style="color: #666; font-size: 14px; text-transform: uppercase;">Projeto de Controladores</div>
      <h1 style="margin-top: 5px; color: #1f77b4;">{html.escape(controlador_nome)}</h1>
      <p style="color: #444;">Interativo, detalhado e com foco na visualização das etapas de projeto e discretização.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("**Instruções de entrada:** Use ponto (`.`) para decimais e separe coeficientes dos polinômios por espaço.")
st.markdown("---")

try:
    if opcao.startswith("1"): resolver_pd()
    elif opcao.startswith("2"): resolver_pi()
    elif opcao.startswith("3"): resolver_pid()
    elif opcao.startswith("4"): resolver_gc_generico()
except Exception as exc:
    st.error(f"**Erro na execução dos cálculos:** {exc}")