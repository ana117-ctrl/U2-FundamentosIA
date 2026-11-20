#!/usr/bin/env python3
"""
Intérprete de funciones matemáticas con árbol de expresión.

- Tokeniza y parsea la expresión (descenso recursivo, sin eval()).
- Construye el árbol de expresión (AST) y lo dibuja.
- Evalúa el árbol sobre un arreglo de NumPy y grafica la función.

Uso:
    python arbol_expresion.py                       # ventana interactiva
    python arbol_expresion.py "sin(x)/x" -10 10     # expresión y rango iniciales
    python arbol_expresion.py "x^2-4" -5 5 --guardar salida.png

Requisitos: pip install numpy matplotlib
"""
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# ----------------------------------------------------------------------------
# 1. Funciones y constantes soportadas
# ----------------------------------------------------------------------------
FUNCIONES = {
    "sin": np.sin, "sen": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "ln": np.log, "log": np.log10, "log10": np.log10,
    "log2": np.log2, "sqrt": np.sqrt, "abs": np.abs,
    "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
}
CONSTANTES = {"pi": np.pi, "e": np.e}


# ----------------------------------------------------------------------------
# 2. Nodos del árbol
# ----------------------------------------------------------------------------
class Nodo:
    tipo = "nodo"
    hijos = ()

    def etiqueta(self):
        raise NotImplementedError

    def evaluar(self, entorno):
        raise NotImplementedError


class Numero(Nodo):
    tipo = "numero"

    def __init__(self, valor):
        self.valor = float(valor)

    def etiqueta(self):
        return f"{self.valor:g}"

    def evaluar(self, entorno):
        return self.valor


class Constante(Nodo):
    tipo = "constante"

    def __init__(self, nombre):
        self.nombre = nombre

    def etiqueta(self):
        return self.nombre

    def evaluar(self, entorno):
        return CONSTANTES[self.nombre]


class Variable(Nodo):
    tipo = "variable"

    def __init__(self, nombre):
        self.nombre = nombre

    def etiqueta(self):
        return self.nombre

    def evaluar(self, entorno):
        return entorno[self.nombre]


class Unario(Nodo):
    tipo = "operador"

    def __init__(self, op, hijo):
        self.op, self.hijos = op, (hijo,)

    def etiqueta(self):
        return f"{self.op}(un)"

    def evaluar(self, entorno):
        v = self.hijos[0].evaluar(entorno)
        return -v if self.op == "-" else v


class Binario(Nodo):
    tipo = "operador"
    OPS = {
        "+": np.add, "-": np.subtract, "*": np.multiply,
        "/": np.divide, "^": np.power,
    }

    def __init__(self, op, izq, der):
        self.op, self.hijos = op, (izq, der)

    def etiqueta(self):
        return self.op

    def evaluar(self, entorno):
        a = self.hijos[0].evaluar(entorno)
        b = self.hijos[1].evaluar(entorno)
        return self.OPS[self.op](a, b)


class Llamada(Nodo):
    tipo = "funcion"

    def __init__(self, nombre, argumento):
        self.nombre, self.hijos = nombre, (argumento,)

    def etiqueta(self):
        return f"{self.nombre}()"

    def evaluar(self, entorno):
        return FUNCIONES[self.nombre](self.hijos[0].evaluar(entorno))


# ----------------------------------------------------------------------------
# 3. Tokenizador
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(
    r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)|([A-Za-z_]\w*)|(\*\*|[-+*/^()]))"
)


def tokenizar(texto):
    texto = texto.strip()
    tokens, pos = [], 0
    while pos < len(texto):
        m = TOKEN_RE.match(texto, pos)
        if not m:
            raise SyntaxError(f"Carácter no válido en la posición {pos}: {texto[pos]!r}")
        num, nombre, op = m.groups()
        if num is not None:
            tokens.append(("NUM", num))
        elif nombre is not None:
            tokens.append(("NOM", nombre))
        else:
            tokens.append(("OP", "^" if op == "**" else op))
        pos = m.end()
    return tokens


# ----------------------------------------------------------------------------
# 4. Parser (descenso recursivo)
#    expr    := termino (('+'|'-') termino)*
#    termino := unario (('*'|'/')? unario)*      (multiplicación implícita: 2x, 2(x+1))
#    unario  := ('-'|'+') unario | potencia
#    potencia:= primario ('^' unario)?           (asociativa a la derecha)
#    primario:= NUM | NOM | NOM '(' expr ')' | '(' expr ')'
# ----------------------------------------------------------------------------
class Parser:
    def __init__(self, texto):
        self.tokens = tokenizar(texto)
        self.i = 0

    def ver(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def tomar(self):
        t = self.ver()
        self.i += 1
        return t

    def esperar(self, valor):
        t = self.tomar()
        if t[1] != valor:
            raise SyntaxError(f"Se esperaba {valor!r}")

    def parsear(self):
        if not self.tokens:
            raise SyntaxError("Expresión vacía")
        nodo = self.expr()
        if self.i < len(self.tokens):
            raise SyntaxError(f"Símbolo inesperado: {self.ver()[1]!r}")
        return nodo

    def expr(self):
        nodo = self.termino()
        while self.ver() in (("OP", "+"), ("OP", "-")):
            op = self.tomar()[1]
            nodo = Binario(op, nodo, self.termino())
        return nodo

    def inicia_primario(self):
        k, v = self.ver()
        return k in ("NUM", "NOM") or (k == "OP" and v == "(")

    def termino(self):
        nodo = self.unario()
        while True:
            if self.ver() in (("OP", "*"), ("OP", "/")):
                op = self.tomar()[1]
                nodo = Binario(op, nodo, self.unario())
            elif self.inicia_primario():          # multiplicación implícita
                nodo = Binario("*", nodo, self.unario())
            else:
                return nodo

    def unario(self):
        if self.ver() in (("OP", "-"), ("OP", "+")):
            op = self.tomar()[1]
            return Unario(op, self.unario())
        return self.potencia()

    def potencia(self):
        base = self.primario()
        if self.ver() == ("OP", "^"):
            self.tomar()
            return Binario("^", base, self.unario())
        return base

    def primario(self):
        k, v = self.tomar()
        if k == "NUM":
            return Numero(v)
        if k == "NOM":
            if self.ver() == ("OP", "(") and v in FUNCIONES:
                self.tomar()
                arg = self.expr()
                self.esperar(")")
                return Llamada(v, arg)
            if v in FUNCIONES:
                raise SyntaxError(f"La función {v!r} necesita paréntesis")
            if v in CONSTANTES:
                return Constante(v)
            return Variable(v)
        if (k, v) == ("OP", "("):
            nodo = self.expr()
            self.esperar(")")
            return nodo
        raise SyntaxError("Expresión incompleta" if k is None else f"Símbolo inesperado: {v!r}")


def variables_de(nodo):
    if isinstance(nodo, Variable):
        return {nodo.nombre}
    return set().union(*(variables_de(h) for h in nodo.hijos)) if nodo.hijos else set()


# ----------------------------------------------------------------------------
# 5. Visualización del árbol
# ----------------------------------------------------------------------------
COLORES = {
    "operador": "#f4a261", "funcion": "#90be6d", "variable": "#4cc9f0",
    "numero": "#e9c46a", "constante": "#cdb4db",
}


def asignar_posiciones(nodo, profundidad=0, contador=None, pos=None):
    """x: hojas en orden; padres centrados sobre sus hijos. y: -profundidad."""
    if contador is None:
        contador, pos = [0], {}
    if not nodo.hijos:
        x = contador[0]
        contador[0] += 1
    else:
        xs = [asignar_posiciones(h, profundidad + 1, contador, pos)[0][0] for h in nodo.hijos]
        x = sum(xs) / len(xs)
    pos[id(nodo)] = (x, -profundidad)
    return pos[id(nodo)], pos


def dibujar_arbol(ax, raiz):
    ax.clear()
    _, pos = asignar_posiciones(raiz)

    def aristas(n):
        for h in n.hijos:
            (x1, y1), (x2, y2) = pos[id(n)], pos[id(h)]
            ax.plot([x1, x2], [y1, y2], color="#555", lw=1.3, zorder=1)
            aristas(h)

    def nodos(n):
        x, y = pos[id(n)]
        ax.text(x, y, n.etiqueta(), ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=2,
                bbox=dict(boxstyle="round,pad=0.4", fc=COLORES[n.tipo], ec="#333"))
        for h in n.hijos:
            nodos(h)

    aristas(raiz)
    nodos(raiz)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.7, 0.7)
    ax.axis("off")
    ax.set_title("Árbol de expresión")


# ----------------------------------------------------------------------------
# 6. Gráfica de la función
# ----------------------------------------------------------------------------
def graficar_funcion(ax, raiz, texto, xmin, xmax, var):
    ax.clear()
    x = np.linspace(xmin, xmax, 2000)
    with np.errstate(all="ignore"):
        y = np.broadcast_to(raiz.evaluar({var: x}), x.shape).astype(float)
    y[~np.isfinite(y)] = np.nan
    y[np.abs(y) > 1e6] = np.nan          # corta asíntotas (tan, 1/x, ...)

    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.plot(x, y, lw=2, color="#1d6fd8")
    finitos = y[np.isfinite(y)]
    if finitos.size:
        lo, hi = np.percentile(finitos, [1, 99])
        margen = (hi - lo) * 0.25 or 1.0
        ax.set_ylim(lo - margen, hi + margen)
    ax.set_xlim(xmin, xmax)
    ax.grid(alpha=0.3)
    ax.set_xlabel(var)
    ax.set_ylabel(f"f({var})")
    ax.set_title(f"f({var}) = {texto}")


# ----------------------------------------------------------------------------
# 7. Aplicación
# ----------------------------------------------------------------------------
def procesar(texto, rango):
    """Devuelve (raiz, variable, xmin, xmax) o lanza una excepción con mensaje."""
    raiz = Parser(texto).parsear()
    libres = variables_de(raiz)
    if len(libres) > 1:
        raise ValueError(f"Sólo se admite una variable; encontradas: {sorted(libres)}")
    var = next(iter(libres), "x")
    try:
        xmin, xmax = (float(v) for v in re.split(r"[,; ]+", rango.strip()))
    except ValueError:
        raise ValueError("Rango inválido. Usa el formato: -10, 10")
    if xmin >= xmax:
        raise ValueError("El rango debe cumplir min < max")
    return raiz, var, xmin, xmax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("expresion", nargs="?", default="sin(x)/x + x^2/10")
    ap.add_argument("xmin", nargs="?", type=float, default=-10)
    ap.add_argument("xmax", nargs="?", type=float, default=10)
    ap.add_argument("--guardar", metavar="ARCHIVO", help="guarda la figura y termina")
    args = ap.parse_args()

    if args.guardar:
        import matplotlib
        matplotlib.use("Agg")

    fig = plt.figure(figsize=(14, 7))
    ax_arbol = fig.add_axes([0.04, 0.22, 0.46, 0.68])
    ax_func = fig.add_axes([0.57, 0.22, 0.40, 0.68])
    ax_msg = fig.add_axes([0.04, 0.01, 0.93, 0.05])
    ax_msg.axis("off")
    msg = ax_msg.text(0, 0.5, "", color="crimson", fontsize=10, va="center")

    caja_f = TextBox(fig.add_axes([0.10, 0.10, 0.50, 0.06]), "f(x) = ", initial=args.expresion)
    caja_r = TextBox(fig.add_axes([0.72, 0.10, 0.20, 0.06]), "rango ", initial=f"{args.xmin:g}, {args.xmax:g}")

    def actualizar(_=None):
        try:
            raiz, var, xmin, xmax = procesar(caja_f.text, caja_r.text)
            dibujar_arbol(ax_arbol, raiz)
            graficar_funcion(ax_func, raiz, caja_f.text, xmin, xmax, var)
            msg.set_text("")
        except Exception as err:               # errores de sintaxis / rango
            msg.set_text(f"Error: {err}")
        fig.canvas.draw_idle()

    caja_f.on_submit(actualizar)
    caja_r.on_submit(actualizar)
    actualizar()

    if args.guardar:
        fig.savefig(args.guardar, dpi=140)
        print(f"Figura guardada en {args.guardar}")
    else:
        plt.show()


if __name__ == "__main__":
    main()