"""
evaluador.py — Intérprete de expresiones matemáticas basado en árbol de expresión.

Solo depende de NumPy (sin gráficos): se puede importar desde cualquier algoritmo.

Operaciones válidas (diccionario OPERACIONES)
---------------------------------------------
    binarias : +  -  *  /  %  ^          (% = módulo, ^ = potencia)
    unarias  : sen  cos  tan  sqrt  log   (log = base 10; "sin" se acepta como alias de "sen")
    extra    : menos unario (-x), paréntesis, multiplicación implícita (2x, 3sen(x))
    constantes: pi, e      variables: cualquier otro identificador (x, y, t, ...)

Agregar una operación nueva es una línea:
    registrar_operacion("sigmoide", 1, lambda v: 1/(1+np.exp(-v)))
    registrar_operacion("min2", ...)  # (las binarias necesitan nivel y símbolo no alfabético)

Uso rápido
----------
    from evaluador import Evaluador

    f = Evaluador("sen(x)/x + x^2 % 3", altura_maxima=5)
    f(2.0)                        # -> float
    f(np.linspace(-5, 5, 100))    # -> ndarray (vectorizado)
    f.altura                      # niveles del árbol (una hoja sola = 1)
    f.arbol_texto()               # árbol indentado

    for p in f.pasos(2.0):        # paso a paso (orden posterior: hijos antes que el padre)
        print(p.texto)            # p.nodo, p.texto, p.valor

    xs = np.linspace(-5, 5, 50)
    f.exportar_csv(xs=xs)         # crea funcion_<fecha>_<expresion>.csv con columnas x, y
    f.exportar_csv("datos/mi_funcion.csv", xs=xs)   # o en la ruta que indiques
"""
import csv
import os
import re
import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ErrorExpresion", "Evaluador", "Paso", "Operacion", "OPERACIONES",
    "registrar_operacion", "registrar_constante", "operaciones_binarias",
    "operaciones_unarias", "formatear", "Nodo", "Numero", "Constante",
    "Variable", "Unario", "Binario", "Llamada", "Tokenizador", "Parser",
]


class ErrorExpresion(Exception):
    """Error de sintaxis, de evaluación o de altura en una expresión."""


# ----------------------------------------------------------------------------
# Diccionario de operaciones válidas
# ----------------------------------------------------------------------------
class Operacion:
    """simbolo, aridad (1 ó 2), funcion (NumPy) y nivel de precedencia (solo binarias)."""

    def __init__(self, simbolo, aridad, funcion, nivel, descripcion="", alias=()):
        self.simbolo, self.aridad, self.funcion = simbolo, aridad, funcion
        self.nivel, self.descripcion, self.alias = nivel, descripcion, tuple(alias)

    def __repr__(self):
        return f"Operacion({self.simbolo!r}, aridad={self.aridad})"


OPERACIONES = {}      # simbolo -> Operacion
_ALIAS = {}           # alias en minúscula -> simbolo canónico
_PRIORIDAD = {"suma": 1, "producto": 2, "potencia": 4}   # unario "-": 3, átomos y funciones: 5
_PRIORIDAD_UNARIO, _PRIORIDAD_ATOMO = 3, 5


def registrar_operacion(simbolo, aridad, funcion, nivel=None, descripcion="", alias=()):
    """Agrega o reemplaza una operación. Binarias: nivel en {'suma','producto','potencia'}."""
    if aridad == 2:
        if nivel not in _PRIORIDAD:
            raise ValueError(f"Las binarias necesitan nivel en {sorted(_PRIORIDAD)}")
        if simbolo.isalnum():
            raise ValueError("El símbolo de una operación binaria no puede ser alfabético")
    elif aridad == 1:
        nivel = "funcion"
        simbolo = simbolo.lower()
    else:
        raise ValueError("La aridad debe ser 1 ó 2")
    OPERACIONES[simbolo] = Operacion(simbolo, aridad, funcion, nivel, descripcion, alias)
    for a in alias:
        _ALIAS[a.lower()] = simbolo


def operaciones_binarias(nivel=None):
    return {s: o for s, o in OPERACIONES.items()
            if o.aridad == 2 and (nivel is None or o.nivel == nivel)}


def operaciones_unarias():
    return {s: o for s, o in OPERACIONES.items() if o.aridad == 1}


def _nombre_funcion(nombre):
    """Nombre canónico de una función unaria registrada (o None)."""
    bajo = nombre.lower()
    bajo = _ALIAS.get(bajo, bajo)
    op = OPERACIONES.get(bajo)
    return bajo if op is not None and op.aridad == 1 else None


# Tabla inicial: exactamente las operaciones pedidas
registrar_operacion("+", 2, np.add, "suma", "suma")
registrar_operacion("-", 2, np.subtract, "suma", "resta")
registrar_operacion("*", 2, np.multiply, "producto", "multiplicación")
registrar_operacion("/", 2, np.true_divide, "producto", "división")
registrar_operacion("%", 2, np.mod, "producto", "módulo")
registrar_operacion("^", 2, np.power, "potencia", "potencia")
registrar_operacion("sen", 1, np.sin, descripcion="seno", alias=("sin",))
registrar_operacion("cos", 1, np.cos, descripcion="coseno")
registrar_operacion("tan", 1, np.tan, descripcion="tangente")
registrar_operacion("sqrt", 1, np.sqrt, descripcion="raíz cuadrada")
registrar_operacion("log", 1, np.log10, descripcion="logaritmo base 10")

CONSTANTES = {"pi": np.pi, "e": np.e}

# Para armar nombres de archivo legibles a partir de la expresión
_NOMBRE_SIMBOLO = {"+": "_mas_", "-": "_menos_", "*": "_por_", "/": "_entre_",
                   "^": "_pot_", "%": "_mod_"}


def registrar_constante(nombre, valor):
    CONSTANTES[nombre.lower()] = float(valor)


# ----------------------------------------------------------------------------
# Formato
# ----------------------------------------------------------------------------
def formatear(valor):
    valor = float(valor)
    if np.isnan(valor):
        return "indef."
    if np.isinf(valor):
        return "∞" if valor > 0 else "-∞"
    return f"{valor:.4g}"


def _arg(valor):
    s = formatear(valor)
    return f"({s})" if float(valor) < 0 else s


# ----------------------------------------------------------------------------
# Nodos del árbol
# ----------------------------------------------------------------------------
class Nodo:
    tipo = "nodo"
    hijos = ()
    prioridad = _PRIORIDAD_ATOMO

    def etiqueta(self):
        raise NotImplementedError

    def _calcular(self, entorno, traza):
        raise NotImplementedError

    def describir(self, valores_hijos, valor):
        """Texto del paso: qué se calculó y cuánto dio."""
        raise NotImplementedError

    def _texto_expresion(self):
        raise NotImplementedError

    def evaluar(self, entorno, traza=None):
        """Si `traza` es un dict, guarda traza[id(nodo)] = valor de cada nodo."""
        valor = self._calcular(entorno, traza)
        if traza is not None:
            traza[id(self)] = valor
        return valor

    def a_expresion(self, minimo=0):
        """Texto con los paréntesis mínimos; al parsearlo se obtiene el mismo árbol."""
        s = self._texto_expresion()
        return f"({s})" if self.prioridad < minimo else s

    def a_texto(self):
        """Texto con paréntesis en todas las operaciones (muestra cómo se interpretó)."""
        return self._texto_expresion() if not self.hijos else self._texto_total()

    def _texto_total(self):
        raise NotImplementedError

    def recorrer(self):
        """Preorden."""
        yield self
        for h in self.hijos:
            yield from h.recorrer()

    def postorden(self):
        """Orden de evaluación: hijos primero, luego el nodo."""
        for h in self.hijos:
            yield from h.postorden()
        yield self

    def altura(self):
        """Niveles del subárbol (una hoja = 1)."""
        return 1 + max((h.altura() for h in self.hijos), default=0)

    def variables(self):
        return {n.nombre for n in self.recorrer() if isinstance(n, Variable)}

    def __repr__(self):
        return f"{type(self).__name__}({self.etiqueta()!r})"


class Numero(Nodo):
    tipo = "numero"

    def __init__(self, valor):
        self.valor = float(valor)

    def etiqueta(self):
        return f"{self.valor:g}"

    def _texto_expresion(self):
        return self.etiqueta()

    _texto_total = _texto_expresion

    def _calcular(self, entorno, traza):
        return self.valor

    def describir(self, valores_hijos, valor):
        return f"número {self.etiqueta()}"


class Constante(Nodo):
    tipo = "constante"

    def __init__(self, nombre):
        self.nombre = nombre

    def etiqueta(self):
        return self.nombre

    def _texto_expresion(self):
        return self.nombre

    _texto_total = _texto_expresion

    def _calcular(self, entorno, traza):
        return CONSTANTES[self.nombre]

    def describir(self, valores_hijos, valor):
        return f"{self.nombre} = {formatear(valor)}"


class Variable(Nodo):
    tipo = "variable"

    def __init__(self, nombre):
        self.nombre = nombre

    def etiqueta(self):
        return self.nombre

    def _texto_expresion(self):
        return self.nombre

    _texto_total = _texto_expresion

    def _calcular(self, entorno, traza):
        return entorno[self.nombre]

    def describir(self, valores_hijos, valor):
        return f"{self.nombre} = {formatear(valor)}"


class Unario(Nodo):
    """Signo unario: -x o +x."""
    tipo = "operador"
    prioridad = _PRIORIDAD_UNARIO

    def __init__(self, op, hijo):
        self.op, self.hijos = op, (hijo,)

    def etiqueta(self):
        return f"{self.op}(un)"

    def _texto_expresion(self):
        return f"{self.op}{self.hijos[0].a_expresion(_PRIORIDAD_UNARIO)}"

    def _texto_total(self):
        return f"({self.op}{self.hijos[0].a_texto()})"

    def _calcular(self, entorno, traza):
        v = self.hijos[0].evaluar(entorno, traza)
        return -v if self.op == "-" else v

    def describir(self, valores_hijos, valor):
        return f"{self.op}({formatear(valores_hijos[0])}) = {formatear(valor)}"


class Binario(Nodo):
    tipo = "operador"

    def __init__(self, op, izq, der):
        self.op, self.hijos = op, (izq, der)

    @property
    def prioridad(self):
        return _PRIORIDAD[OPERACIONES[self.op].nivel]

    def etiqueta(self):
        return self.op

    def _texto_expresion(self):
        p = self.prioridad
        if self.op == "^":              # asociativa a la derecha
            izq, der, sep = p + 1, _PRIORIDAD_UNARIO, ""
        else:                           # asociativa a la izquierda
            izq, der, sep = p, p + 1, " "
        return (f"{self.hijos[0].a_expresion(izq)}{sep}{self.op}{sep}"
                f"{self.hijos[1].a_expresion(der)}")

    def _texto_total(self):
        return f"({self.hijos[0].a_texto()} {self.op} {self.hijos[1].a_texto()})"

    def _calcular(self, entorno, traza):
        a = self.hijos[0].evaluar(entorno, traza)
        b = self.hijos[1].evaluar(entorno, traza)
        return OPERACIONES[self.op].funcion(a, b)

    def describir(self, valores_hijos, valor):
        a, b = valores_hijos
        return f"{_arg(a)} {self.op} {_arg(b)} = {formatear(valor)}"


class Llamada(Nodo):
    """Operación unaria con nombre: sen, cos, tan, sqrt, log, ..."""
    tipo = "funcion"

    def __init__(self, nombre, argumento):
        self.nombre, self.hijos = nombre, (argumento,)

    def etiqueta(self):
        return f"{self.nombre}()"

    def _texto_expresion(self):
        return f"{self.nombre}({self.hijos[0].a_expresion(0)})"

    def _texto_total(self):
        return f"{self.nombre}({self.hijos[0].a_texto()})"

    def _calcular(self, entorno, traza):
        return OPERACIONES[self.nombre].funcion(self.hijos[0].evaluar(entorno, traza))

    def describir(self, valores_hijos, valor):
        return f"{self.nombre}({formatear(valores_hijos[0])}) = {formatear(valor)}"


# ----------------------------------------------------------------------------
# Tokenizador
# ----------------------------------------------------------------------------
class Tokenizador:
    def __init__(self):
        simbolos = [s for s in operaciones_binarias()]
        simbolos = sorted((re.escape(s) for s in simbolos), key=len, reverse=True)
        ops = "|".join([r"\*\*"] + simbolos + [r"\(", r"\)"])
        self._re = re.compile(
            r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)|([A-Za-z_]\w*)|(" + ops + "))")

    def tokenizar(self, texto):
        texto = texto.strip()
        tokens, pos = [], 0
        while pos < len(texto):
            m = self._re.match(texto, pos)
            if not m:
                pos += len(texto[pos:]) - len(texto[pos:].lstrip())
                raise ErrorExpresion(f"Carácter no válido en la posición {pos}: {texto[pos]!r}")
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
# Parser (descenso recursivo; los símbolos por nivel salen de OPERACIONES)
#   expr    := termino (suma termino)*             suma     = + -
#   termino := unario ((producto)? unario)*        producto = * / %   (o multiplicación implícita)
#   unario  := ('-'|'+') unario | potencia
#   potencia:= primario ('^' unario)?              (asociativa a la derecha)
#   primario:= NUM | NOM | NOM '(' expr ')' | '(' expr ')'
# ----------------------------------------------------------------------------
class Parser:
    def __init__(self, texto):
        self.tokens = Tokenizador().tokenizar(texto)
        self.i = 0
        self._sim_suma = set(operaciones_binarias("suma"))
        self._sim_producto = set(operaciones_binarias("producto"))
        self._sim_potencia = set(operaciones_binarias("potencia"))

    def _ver(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def _tomar(self):
        t = self._ver()
        self.i += 1
        return t

    def _es_op(self, simbolos):
        k, v = self._ver()
        return k == "OP" and v in simbolos

    def _esperar(self, valor):
        if self._tomar()[1] != valor:
            raise ErrorExpresion(f"Se esperaba {valor!r}")

    def parsear(self):
        if not self.tokens:
            raise ErrorExpresion("Expresión vacía")
        nodo = self._expr()
        if self.i < len(self.tokens):
            raise ErrorExpresion(f"Símbolo inesperado: {self._ver()[1]!r}")
        return nodo

    def _expr(self):
        nodo = self._termino()
        while self._es_op(self._sim_suma):
            op = self._tomar()[1]
            nodo = Binario(op, nodo, self._termino())
        return nodo

    def _inicia_primario(self):
        k, v = self._ver()
        return k in ("NUM", "NOM") or (k == "OP" and v == "(")

    def _termino(self):
        nodo = self._unario()
        while True:
            if self._es_op(self._sim_producto):
                op = self._tomar()[1]
                nodo = Binario(op, nodo, self._unario())
            elif self._inicia_primario():          # multiplicación implícita
                nodo = Binario("*", nodo, self._unario())
            else:
                return nodo

    def _unario(self):
        if self._ver() in (("OP", "-"), ("OP", "+")):
            op = self._tomar()[1]
            return Unario(op, self._unario())
        return self._potencia()

    def _potencia(self):
        base = self._primario()
        if self._es_op(self._sim_potencia):
            op = self._tomar()[1]
            return Binario(op, base, self._unario())
        return base

    def _primario(self):
        k, v = self._tomar()
        if k == "NUM":
            return Numero(v)
        if k == "NOM":
            funcion = _nombre_funcion(v)
            if funcion is not None:
                if self._ver() != ("OP", "("):
                    raise ErrorExpresion(f"La operación {v!r} necesita paréntesis: {v}(...)")
                self._tomar()
                arg = self._expr()
                self._esperar(")")
                return Llamada(funcion, arg)
            if v.lower() in CONSTANTES:
                return Constante(v.lower())
            if len(v) > 1 and self._ver() == ("OP", "("):   # nombre(...) que no existe
                validas = ", ".join(OPERACIONES)
                raise ErrorExpresion(f"Operación no válida: {v!r}. Válidas: {validas}")
            return Variable(v)
        if (k, v) == ("OP", "("):
            nodo = self._expr()
            self._esperar(")")
            return nodo
        raise ErrorExpresion("Expresión incompleta" if k is None else f"Símbolo inesperado: {v!r}")


# ----------------------------------------------------------------------------
# Evaluador (la clase que se usa desde fuera)
# ----------------------------------------------------------------------------
@dataclass
class Paso:
    """Un paso de la evaluación: el nodo calculado, su texto y su valor."""
    nodo: Nodo
    texto: str
    valor: float


class Evaluador:
    """
    Compila una expresión a un árbol y la evalúa como una función normal.

    expresion     : str
    variables     : orden de las variables en llamadas posicionales (None = alfabético)
    altura_maxima : si se indica, rechaza árboles con más niveles (ErrorExpresion)
    """

    def __init__(self, expresion, variables=None, altura_maxima=None):
        self.expresion = expresion.strip()
        self.arbol = Parser(self.expresion).parsear()
        usadas = self.arbol.variables()
        if variables is None:
            self.variables = tuple(sorted(usadas))
        else:
            self.variables = tuple(variables)
            desconocidas = usadas - set(self.variables)
            if desconocidas:
                raise ErrorExpresion(f"Variables no declaradas: {sorted(desconocidas)}")
        self.altura_maxima = altura_maxima
        if altura_maxima is not None and self.altura > altura_maxima:
            raise ErrorExpresion(
                f"El árbol tiene altura {self.altura} y el máximo permitido es {altura_maxima}")

    @classmethod
    def desde_arbol(cls, arbol, variables=None, altura_maxima=None):
        """Crea un Evaluador a partir de un árbol ya construido (p. ej. del generador)."""
        return cls(arbol.a_expresion(), variables=variables, altura_maxima=altura_maxima)

    @property
    def altura(self):
        return self.arbol.altura()

    # --- evaluación -------------------------------------------------------
    def _entorno(self, args, kwargs):
        if len(args) > len(self.variables):
            raise ErrorExpresion(
                f"Se esperaban {len(self.variables)} valor(es) {self.variables}, llegaron {len(args)}")
        entorno = dict(zip(self.variables, args))
        entorno.update(kwargs)
        extra = set(entorno) - set(self.variables)
        if extra:
            raise ErrorExpresion(f"Variables desconocidas: {sorted(extra)}")
        faltan = [v for v in self.variables if v not in entorno]
        if faltan:
            raise ErrorExpresion(f"Faltan valores para: {faltan}")
        return {k: np.asarray(v, dtype=float) for k, v in entorno.items()}

    def evaluar(self, *args, **kwargs):
        """Escalares -> float. Arreglos -> ndarray (broadcast entre variables)."""
        entorno = self._entorno(args, kwargs)
        with np.errstate(all="ignore"):
            res = np.asarray(self.arbol.evaluar(entorno), dtype=float)
        forma = np.broadcast(*entorno.values()).shape if entorno else ()
        res = np.broadcast_to(res, forma)
        return float(res) if forma == () else np.array(res)

    __call__ = evaluar

    def evaluar_traza(self, *args, **kwargs):
        """(valor, traza) en UN punto escalar; traza[id(nodo)] = valor intermedio."""
        entorno = self._entorno(args, kwargs)
        traza = {}
        with np.errstate(all="ignore"):
            valor = self.arbol.evaluar(entorno, traza)
        return float(valor), {k: float(v) for k, v in traza.items()}

    def pasos(self, *args, **kwargs):
        """Lista de `Paso` (uno por nodo, en el orden en que se calcula) para un punto."""
        _, traza = self.evaluar_traza(*args, **kwargs)
        lista = []
        for nodo in self.arbol.postorden():
            valor = traza[id(nodo)]
            hijos = [traza[id(h)] for h in nodo.hijos]
            lista.append(Paso(nodo, nodo.describir(hijos, valor), valor))
        return lista

    def muestrear(self, xmin, xmax, n=200):
        xs = np.linspace(xmin, xmax, n)
        return xs, self.evaluar(xs)

    def exportar_csv(self, ruta=None, xs=None, ys=None, carpeta=".", delimitador=","):
        """
        Guarda los puntos (x, f(x)) en un archivo .csv y devuelve la ruta creada.

        ruta   : archivo de salida. Si es None se crea
                 <carpeta>/funcion_<AAAAMMDD_HHMMSS>_<expresion>.csv (nunca pisa otro).
        xs     : valores de x a exportar (obligatorio).
        ys     : valores ya calculados; si se omiten se evalúa la función en xs.
        Formato: encabezado "x,y"; los valores indefinidos salen como nan / inf / -inf.
        Solo para funciones de una variable.
        """
        if len(self.variables) > 1:
            raise ErrorExpresion("exportar_csv solo admite funciones de una variable")
        if xs is None:
            raise ErrorExpresion("Indica los valores de x a exportar (xs=...)")
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        if ys is None:
            ys = self(xs) if self.variables else np.full(xs.shape, self())
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        if xs.shape != ys.shape:
            raise ErrorExpresion("xs e ys deben tener la misma cantidad de valores")

        if ruta is None:
            texto = "".join(_NOMBRE_SIMBOLO.get(c, c) for c in self.expresion)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", texto).strip("_")[:50] or "expresion"
            base = os.path.join(carpeta, f"funcion_{time.strftime('%Y%m%d_%H%M%S')}_{slug}")
            ruta, k = base + ".csv", 2
            while os.path.exists(ruta):             # varias funciones en el mismo segundo
                ruta, k = f"{base}_{k}.csv", k + 1
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)

        var = self.variables[0] if self.variables else "x"
        with open(ruta, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo, delimiter=delimitador)
            escritor.writerow([var, "y"])
            for x, y in zip(xs, ys):
                escritor.writerow([float(x), float(y)])
        return ruta

    # --- inspección -------------------------------------------------------
    @property
    def texto_infijo(self):
        """Expresión con paréntesis explícitos: muestra cómo se interpretó."""
        return self.arbol.a_texto()

    def arbol_texto(self):
        lineas = []

        def rec(nodo, nivel):
            lineas.append("  " * nivel + nodo.etiqueta())
            for h in nodo.hijos:
                rec(h, nivel + 1)

        rec(self.arbol, 0)
        return "\n".join(lineas)

    def __repr__(self):
        return f"Evaluador({self.expresion!r}, variables={self.variables}, altura={self.altura})"


if __name__ == "__main__":  # demo rápida
    f = Evaluador("-x^2 + 3sen(2x) % 5")
    print(f, "->", f.texto_infijo)
    print(f.arbol_texto())
    print("f(1.5) =", f(1.5))
    for i, p in enumerate(f.pasos(1.5), 1):
        print(f"{i:>2}. {p.texto}")
    print("Operaciones válidas:", list(OPERACIONES))