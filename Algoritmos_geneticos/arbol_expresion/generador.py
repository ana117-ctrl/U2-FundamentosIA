"""
generador.py — Generador aleatorio de expresiones (árboles) con altura máxima.

Usa el diccionario OPERACIONES de evaluador.py, así que cualquier operación que
se registre allí queda disponible automáticamente. Sirve para poblaciones
iniciales de algoritmos genéticos / programación genética.

    from generador import GeneradorExpresiones

    gen = GeneradorExpresiones(altura_maxima=4, semilla=7)
    gen.generar()                         # -> "sen(x) * 3 + x^2 % 5"  (texto)
    gen.generar_arbol()                   # -> Nodo (raíz del árbol)
    f = gen.generar_evaluador(-10, 10)    # -> Evaluador con valores bien definidos
    poblacion = gen.generar_poblacion(50, -10, 10)

Métodos de construcción (clásicos de programación genética):
    "completo"    : todas las ramas llegan exactamente a la altura máxima
    "crecimiento" : las ramas pueden terminar antes (alturas variadas)
    None          : elige uno al azar en cada expresión
"""
import random

import numpy as np

from evaluador import (OPERACIONES, Binario, ErrorExpresion, Evaluador, Llamada,
                       Numero, Variable)

__all__ = ["GeneradorExpresiones", "METODOS"]

METODOS = ("completo", "crecimiento")


class GeneradorExpresiones:
    """
    altura_maxima : niveles máximos del árbol (una hoja sola = 1)
    operaciones   : símbolos permitidos (por defecto, todas las de OPERACIONES)
    variable      : nombre de la variable (x)
    rango_numeros : (mín, máx) de las constantes numéricas generadas
    enteros       : constantes enteras (True) o con un decimal (False)
    prob_variable : probabilidad de que una hoja sea la variable (si no, un número)
    prob_terminal : probabilidad de cortar una rama antes (método "crecimiento")
    semilla       : para resultados reproducibles
    """

    def __init__(self, altura_maxima=4, operaciones=None, variable="x",
                 rango_numeros=(1, 9), enteros=True, prob_variable=0.6,
                 prob_terminal=0.3, semilla=None):
        self.altura_maxima = altura_maxima
        self.operaciones = list(operaciones) if operaciones else list(OPERACIONES)
        desconocidas = [o for o in self.operaciones if o not in OPERACIONES]
        if desconocidas:
            raise ErrorExpresion(f"Operaciones no registradas: {desconocidas}")
        self.variable = variable
        self.rango_numeros = rango_numeros
        self.enteros = enteros
        self.prob_variable = prob_variable
        self.prob_terminal = prob_terminal
        self.rnd = random.Random(semilla)

    # --- altura máxima (validada) -----------------------------------------
    @property
    def altura_maxima(self):
        return self._altura_maxima

    @altura_maxima.setter
    def altura_maxima(self, valor):
        if int(valor) < 1:
            raise ErrorExpresion("La altura máxima debe ser al menos 1")
        self._altura_maxima = int(valor)

    # --- piezas de construcción -------------------------------------------
    def _terminal(self):
        if self.rnd.random() < self.prob_variable:
            return Variable(self.variable)
        lo, hi = self.rango_numeros
        valor = self.rnd.randint(int(lo), int(hi)) if self.enteros \
            else round(self.rnd.uniform(lo, hi), 1)
        return Numero(valor)

    def _nodo(self, restante, metodo, es_raiz):
        corta = (metodo == "crecimiento" and not es_raiz
                 and self.rnd.random() < self.prob_terminal)
        if restante <= 1 or corta:
            return self._terminal()
        op = OPERACIONES[self.rnd.choice(self.operaciones)]
        hijos = [self._nodo(restante - 1, metodo, False) for _ in range(op.aridad)]
        return Binario(op.simbolo, *hijos) if op.aridad == 2 else Llamada(op.simbolo, *hijos)

    def _insertar_variable(self, raiz):
        """Si el árbol quedó sin variable, cambia una hoja numérica por la variable."""
        if not raiz.hijos:
            return Variable(self.variable)
        hojas = [(n, i) for n in raiz.recorrer() for i, h in enumerate(n.hijos) if not h.hijos]
        padre, i = self.rnd.choice(hojas)
        hijos = list(padre.hijos)
        hijos[i] = Variable(self.variable)
        padre.hijos = tuple(hijos)
        return raiz

    # --- API pública -------------------------------------------------------
    def generar_arbol(self, metodo=None, altura=None):
        """Árbol aleatorio con altura <= altura_maxima. Siempre contiene la variable."""
        altura = self.altura_maxima if altura is None else min(int(altura), self.altura_maxima)
        metodo = metodo or self.rnd.choice(METODOS)
        if metodo not in METODOS:
            raise ErrorExpresion(f"Método desconocido: {metodo!r} (usa {METODOS})")
        raiz = self._nodo(max(1, altura), metodo, True)
        if not raiz.variables():
            raiz = self._insertar_variable(raiz)
        return raiz

    def generar(self, metodo=None, altura=None):
        """Expresión aleatoria como texto (parseable por Evaluador)."""
        return self.generar_arbol(metodo, altura).a_expresion()

    @staticmethod
    def _calidad(ys, max_abs):
        finitos = ys[np.isfinite(ys)]
        if finitos.size == 0:
            return 0.0, False
        proporcion = finitos.size / ys.size
        util = np.abs(finitos).max() <= max_abs and np.ptp(finitos) > 1e-9
        return proporcion, bool(util)

    def generar_evaluador(self, xmin=-10, xmax=10, intentos=400, min_validos=0.6,
                          max_abs=1e6, muestras=200, metodo=None, altura_minima=None):
        """
        Genera hasta `intentos` expresiones y devuelve la primera "buena":
          - altura entre `altura_minima` (por defecto altura_maxima - 1) y altura_maxima
          - definida (sin nan/inf) en al menos `min_validos` del rango
          - acotada por `max_abs` y no constante
        Si ninguna cumple todo, devuelve la que más se acerque.
        """
        minima = max(1, self.altura_maxima - 1) if altura_minima is None else altura_minima
        xs = np.linspace(xmin, xmax, muestras)
        mejor, mejor_p, ultimo = None, -1.0, None
        for _ in range(intentos):
            ev = Evaluador(self.generar(metodo), variables=(self.variable,),
                           altura_maxima=self.altura_maxima)
            ultimo = ev
            if ev.altura < minima:
                continue
            proporcion, util = self._calidad(ev(xs), max_abs)
            if util and proporcion >= min_validos:
                return ev
            if util and proporcion > mejor_p:
                mejor, mejor_p = ev, proporcion
        return mejor if mejor is not None else ultimo

    def generar_poblacion(self, n, xmin=-10, xmax=10, **kwargs):
        """Lista de `n` Evaluadores aleatorios (población inicial)."""
        return [self.generar_evaluador(xmin, xmax, **kwargs) for _ in range(n)]


if __name__ == "__main__":  # demo rápida
    gen = GeneradorExpresiones(altura_maxima=4, semilla=1)
    for _ in range(6):
        ev = gen.generar_evaluador(-10, 10)
        print(f"altura {ev.altura}:  f(x) = {ev.expresion}")
