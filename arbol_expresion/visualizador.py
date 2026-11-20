"""
visualizador.py — Interfaz gráfica para `evaluador.py`, `generador.py` y `metricas.py`.

Ecuación por defecto:   y = 6.87 + 11*cos(7.23*x^3)   en  x ∈ [-10, 10],  Δx = 0.001
(20 001 puntos).

  Izquierda : árbol de expresión; solo se resalta el nodo que se está evaluando.
  Centro    : gráfica; cada punto se marca cuando termina de evaluarse.
  Derecha   : pasos de la evaluación (arriba) y errores MSE / RMSE / NMSE / R² (abajo).

Errores: se comparan los valores que calcula el árbol (ŷ) contra la ecuación objetivo
(y), calculada aparte con NumPy. Si cambias la expresión de la caja f(x), el error
mide qué tan lejos está de la ecuación objetivo.

CSV (al terminar todos los puntos de una función, automático):
    <carpeta_csv>/funcion_<fecha>_<expresion>.csv            columnas x, y
    <carpeta_csv>/funcion_<fecha>_<expresion>_errores.csv    MSE, RMSE, NMSE, R²

Con 20 001 puntos, el deslizador de velocidad llega a 20 000 pasos/s y «Todo»
evalúa los puntos restantes de una vez.

Uso:
    python visualizador.py                                  # la ecuación de arriba
    python visualizador.py "6.87 + 11*cos(7.2*x^3)"         # otra candidata (verás el error)
    python visualizador.py --aleatoria --altura 5           # candidata aleatoria
    python visualizador.py --paso 0.01 --rango -5 5
    python visualizador.py --guardar fig.png                # sin ventana (calcula todo)
    python visualizador.py --guardar fig.png --pasos 25     # a los 25 pasos

Desde código:
    from visualizador import AplicacionArbol
    AplicacionArbol().mostrar()

Requiere: numpy, matplotlib, evaluador.py, generador.py y metricas.py (misma carpeta).
"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

from evaluador import ErrorExpresion, Evaluador, formatear
from generador import GeneradorExpresiones
from metricas import calcular_metricas, exportar_metricas_csv

EXPRESION_OBJETIVO = "6.87 + 11*cos(7.23*x^3)"


def objetivo_numpy(x):
    """y = 6.87 + 11*cos(7.23*x^3) calculada directo con NumPy (independiente del árbol)."""
    x = np.asarray(x, dtype=float)
    return 6.87 + 11.0 * np.cos(7.23 * x ** 3)


def limpiar(y, limite=1e12):
    """inf/nan/valores absurdos -> nan (no se grafican)."""
    y = np.array(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    y[np.abs(y) > limite] = np.nan
    return y


# ----------------------------------------------------------------------------
# Vista del árbol (solo resalta el nodo actual)
# ----------------------------------------------------------------------------
class VistaArbol:
    COLORES = {
        "operador": "#f4a261", "funcion": "#90be6d", "variable": "#4cc9f0",
        "numero": "#e9c46a", "constante": "#cdb4db",
    }
    RESALTE_RELLENO, RESALTE_BORDE = "#ffd60a", "#d00000"

    def __init__(self, ax):
        self.ax = ax
        self._textos = {}       # id(nodo) -> Text
        self._tipos = {}        # id(nodo) -> tipo
        self._resaltado = None

    @staticmethod
    def _posiciones(raiz):
        """x: hojas en orden y padres centrados sobre sus hijos; y: -profundidad."""
        pos, contador = {}, [0]

        def rec(nodo, prof):
            if not nodo.hijos:
                x = contador[0]
                contador[0] += 1
            else:
                xs = [rec(h, prof + 1) for h in nodo.hijos]
                x = sum(xs) / len(xs)
            pos[id(nodo)] = (x, -prof)
            return x

        rec(raiz, 0)
        return pos

    def dibujar(self, raiz):
        ax = self.ax
        ax.clear()
        self._textos, self._tipos, self._resaltado = {}, {}, None
        pos = self._posiciones(raiz)
        hojas = sum(1 for n in raiz.recorrer() if not n.hijos)
        fs = 11 if hojas <= 6 else 9 if hojas <= 12 else 7.5
        pad = 0.4 if hojas <= 12 else 0.25

        for nodo in raiz.recorrer():
            for h in nodo.hijos:
                (x1, y1), (x2, y2) = pos[id(nodo)], pos[id(h)]
                ax.plot([x1, x2], [y1, y2], color="#555", lw=1.3, zorder=1)
        for nodo in raiz.recorrer():
            x, y = pos[id(nodo)]
            self._tipos[id(nodo)] = nodo.tipo
            self._textos[id(nodo)] = ax.text(
                x, y, nodo.etiqueta(), ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=2,
                bbox=dict(boxstyle=f"round,pad={pad}", fc=self.COLORES[nodo.tipo],
                          ec="#333", lw=1))

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.7, 0.7)
        ax.axis("off")
        ax.set_title(f"Árbol de expresión  (altura {raiz.altura()})", fontsize=10)

    def _pintar(self, nid, resaltado):
        parche = self._textos[nid].get_bbox_patch()
        if resaltado:
            parche.set_facecolor(self.RESALTE_RELLENO)
            parche.set_edgecolor(self.RESALTE_BORDE)
            parche.set_linewidth(3)
        else:
            parche.set_facecolor(self.COLORES[self._tipos[nid]])
            parche.set_edgecolor("#333")
            parche.set_linewidth(1)

    def resaltar(self, nodo):
        if self._resaltado is not None:
            self._pintar(self._resaltado, False)
        self._resaltado = id(nodo)
        self._pintar(self._resaltado, True)

    def quitar_resaltado(self):
        if self._resaltado is not None:
            self._pintar(self._resaltado, False)
            self._resaltado = None


# ----------------------------------------------------------------------------
# Paneles de texto (derecha)
# ----------------------------------------------------------------------------
def _estilo_panel(ax, titulo):
    ax.set_facecolor("#f8f9fa")
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_color("#bbb")
    ax.set_title(titulo, fontsize=10)


class PanelPasos:
    FS = 10
    LINEA = "─" * 34

    def __init__(self, ax):
        self.ax = ax
        _estilo_panel(ax, "Evaluación paso a paso")
        self.texto = ax.text(0.03, 0.98, "", va="top", ha="left", family="monospace",
                             fontsize=self.FS, transform=ax.transAxes)

    def _capacidad(self):
        fig = self.ax.figure
        alto = self.ax.get_position().height * fig.get_figheight() * 72
        return max(8, int(alto / (self.FS * 1.5)))

    def limpiar(self, mensaje=""):
        self.texto.set_text(mensaje)

    def mostrar(self, var, x, i_punto, n_puntos, pasos, n_mostrados):
        lineas = [f"Punto {i_punto}/{n_puntos}   {var} = {formatear(x)}", self.LINEA]
        cuerpo = [f"{'►' if k == n_mostrados - 1 else ' '}{k + 1:>2}. {pasos[k].texto}"
                  for k in range(n_mostrados)]
        pie = []
        if n_mostrados == len(pasos):
            pie = [self.LINEA, f"y = {formatear(pasos[-1].valor)}"]
        cupo = self._capacidad() - len(lineas) - len(pie) - 1
        if len(cuerpo) > cupo:
            ocultos = len(cuerpo) - cupo + 1
            cuerpo = [f"  … ({ocultos} pasos anteriores)"] + cuerpo[ocultos:]
        self.texto.set_text("\n".join(lineas + cuerpo + pie))


class PanelErrores:
    FS = 10

    def __init__(self, ax):
        self.ax = ax
        _estilo_panel(ax, "Error vs. objetivo")
        self.texto = ax.text(0.04, 0.92, "", va="top", ha="left", family="monospace",
                             fontsize=self.FS, transform=ax.transAxes)

    def limpiar(self):
        self.ax.set_title("Error vs. objetivo", fontsize=10)
        self.texto.set_text("(se calcula al evaluar puntos)")

    @staticmethod
    def _nmse(v):
        return "-∞ dB (error 0)" if v == float("-inf") else f"{formatear(v)} dB"

    def mostrar(self, m, n_total):
        self.ax.set_title(f"Error vs. objetivo  ({m.n_validos}/{n_total} puntos)",
                          fontsize=10)
        self.texto.set_text(
            f"MSE  = {formatear(m.mse)}\n"
            f"RMSE = {formatear(m.rmse)}\n"
            f"NMSE = {self._nmse(m.nmse_db)}\n"
            f"R²   = {m.r2:.8f}")


# ----------------------------------------------------------------------------
# Vista de la gráfica (preparada para decenas de miles de puntos)
# ----------------------------------------------------------------------------
class VistaGrafica:
    MAX_MARCADORES = 400      # con más puntos solo se dibuja la línea

    def __init__(self, ax):
        self.ax = ax
        self.xs = np.array([])
        self.ys = np.array([])
        self.k = 0            # puntos ya calculados
        self.span = 1.0
        self.curva_completa = None

    def preparar(self, evaluador, xs):
        """Limpia el eje y fija los límites con una evaluación completa (oculta)."""
        ax = self.ax
        ax.clear()
        self.xs = xs
        self.ys = np.full(len(xs), np.nan)
        self.k = 0
        self.grilla_propia = len(xs) >= 1500
        self.xd = xs if self.grilla_propia else np.linspace(xs[0], xs[-1], 1500)
        yd = limpiar(evaluador(*([self.xd] if evaluador.variables else [])))
        if yd.ndim == 0:                                   # expresión constante
            yd = np.full_like(self.xd, float(yd))
        self.yd = yd

        finitos = yd[np.isfinite(yd)]
        if finitos.size:
            lo, hi = np.percentile(finitos, [1, 99])
            if hi - lo < 1e-12:
                lo, hi = lo - 1, hi + 1
            margen = (hi - lo) * 0.25
            ax.set_ylim(lo - margen, hi + margen)
            self.span = hi - lo
        else:
            ax.set_ylim(-1, 1)
            self.span = 2.0

        var = evaluador.variables[0] if evaluador.variables else "x"
        ax.set_xlim(xs[0], xs[-1])
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel(var)
        ax.set_ylabel("y")
        titulo = evaluador.expresion if len(evaluador.expresion) <= 40 \
            else evaluador.expresion[:38] + "…"
        ax.set_title(f"y = {titulo}", fontsize=10)

        self.linea, = ax.plot([], [], lw=1.2 if self.grilla_propia else 2,
                              color="#1d6fd8", zorder=3)
        self.puntos, = ax.plot([], [], "o", ms=5, color="#1d6fd8", zorder=4)
        self.guia, = ax.plot([], [], ":", color="#c1121f", lw=1.3, zorder=2)
        self.actual, = ax.plot([], [], "o", ms=9, mfc="#c1121f", mec="white", mew=1.5, zorder=6)
        self.info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                            family="monospace", fontsize=10, zorder=7,
                            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
        self.curva_completa = None

    # --- datos ---------------------------------------------------------------
    def registrar(self, i, y):
        self.ys[i] = y
        self.k = max(self.k, i + 1)

    def registrar_lote(self, i0, ys):
        ys = np.asarray(ys, dtype=float)
        self.ys[i0:i0 + len(ys)] = ys
        self.k = i0 + len(ys)

    def _cortar(self, y):
        """Rompe el trazo en saltos enormes (asíntotas)."""
        y = np.array(y, dtype=float)
        with np.errstate(invalid="ignore"):
            salto = np.abs(np.diff(y)) > self.span * 1.05
        y[1:][salto] = np.nan
        return y

    # --- dibujo (se llama una vez por cuadro, no por punto) -------------------
    def refrescar(self, var="x"):
        k = self.k
        if k == 0:
            for a in (self.linea, self.puntos, self.guia, self.actual):
                a.set_data([], [])
            self.info.set_text("")
            return
        x, y = self.xs[:k], limpiar(self.ys[:k])
        self.linea.set_data(x, self._cortar(y))
        if len(self.xs) <= self.MAX_MARCADORES:
            ok = np.isfinite(y)
            self.puntos.set_data(x[ok], y[ok])
        xa, ya = self.xs[k - 1], self.ys[k - 1]
        if np.isfinite(ya) and abs(ya) <= 1e12:
            self.actual.set_data([xa], [ya])
            self.guia.set_data([xa, xa], [0, ya])
            self.info.set_text(f"{var} = {formatear(xa)}\ny = {formatear(ya)}")
        else:
            self.actual.set_data([], [])
            self.guia.set_data([], [])
            self.info.set_text(f"{var} = {formatear(xa)}\ny = indefinido")

    def mostrar_curva(self):
        """Curva densa de fondo al terminar (solo si los puntos eran pocos)."""
        if not self.grilla_propia and self.curva_completa is None:
            self.curva_completa, = self.ax.plot(
                self.xd, self._cortar(self.yd), lw=1.5, color="#9db4d6", zorder=1)


# ----------------------------------------------------------------------------
# Aplicación (controles + animación)
# ----------------------------------------------------------------------------
class AplicacionArbol:
    ALTURA_TOPE = 8
    MAX_PUNTOS = 500_000
    RATE_MIN, RATE_MAX = 0.5, 20000.0      # pasos (nodos) por segundo
    TICK_MS = 40                           # cuadro de animación cuando va rápido

    # Tamaño de cada zona: [izquierda, abajo, ancho, alto] como fracción de la ventana.
    # Para una gráfica más grande/chica, ajusta "ancho" y "alto" de "grafica" y
    # compensa con "arbol" y los paneles (que no se encimen).
    DISPOSICION = {
        "arbol":   [0.01, 0.30, 0.30, 0.62],
        "grafica": [0.345, 0.33, 0.395, 0.60],
        "pasos":   [0.765, 0.47, 0.225, 0.45],
        "errores": [0.765, 0.30, 0.225, 0.12],
    }

    def __init__(self, expresion=None, xmin=-10.0, xmax=10.0, paso=0.001,
                 altura_maxima=6, velocidad=8.0, semilla=None, reproducir=True,
                 carpeta_csv="salidas_csv", exportar_csv=True, objetivo=None,
                 aleatoria=False):
        self.carpeta_csv, self.exportar_csv = carpeta_csv, exportar_csv
        self.csv_ruta, self.csv_errores, self.csv_error = None, None, None
        self.objetivo = self._crear_objetivo(objetivo)
        self.generador = GeneradorExpresiones(altura_maxima=altura_maxima, semilla=semilla)
        self.ev = None
        self.xs = np.array([])
        self.y_ref = np.array([])
        self.pasos, self.i_paso, self.i_punto = [], 0, 0
        self.x_actual, self.punto_actual = 0.0, 0
        self.reproduciendo = False
        self.lote = 1
        if aleatoria:
            expresion = self.generador.generar_evaluador(xmin, xmax).expresion
        elif expresion is None:
            expresion = EXPRESION_OBJETIVO
        self._crear_figura(expresion, xmin, xmax, paso, altura_maxima, velocidad)
        self.iniciar(reproducir=reproducir)

    @staticmethod
    def _crear_objetivo(objetivo):
        """None -> la ecuación por defecto en NumPy; str -> expresión; callable -> tal cual."""
        if objetivo is None:
            return objetivo_numpy
        if isinstance(objetivo, str):
            ev = Evaluador(objetivo)
            return lambda x: ev(*([np.asarray(x, dtype=float)] if ev.variables else []))
        return objetivo

    # --- velocidad: deslizador logarítmico ---------------------------------
    @classmethod
    def _rate(cls, v):
        return cls.RATE_MIN * (cls.RATE_MAX / cls.RATE_MIN) ** (v / 100.0)

    @classmethod
    def _valor_slider(cls, rate):
        return 100.0 * np.log(rate / cls.RATE_MIN) / np.log(cls.RATE_MAX / cls.RATE_MIN)

    def _aplicar_velocidad(self, rate):
        if rate <= 25:
            self.lote, intervalo = 1, int(1000 / rate)
        else:                                   # muchos pasos por cuadro
            self.lote, intervalo = max(1, int(round(rate * self.TICK_MS / 1000))), self.TICK_MS
        self.timer.interval = intervalo
        self.slider.valtext.set_text(f"{rate:.4g}/s")

    # --- interfaz ---------------------------------------------------------
    def _crear_figura(self, expresion, xmin, xmax, paso, altura, velocidad):
        self.fig = plt.figure(figsize=(17, 9))
        try:
            self.fig.canvas.manager.set_window_title("Árbol de expresión")
        except Exception:
            pass
        add = self.fig.add_axes
        d = self.DISPOSICION
        self.vista_arbol = VistaArbol(add(d["arbol"]))
        self.vista_grafica = VistaGrafica(add(d["grafica"]))
        self.panel = PanelPasos(add(d["pasos"]))
        self.panel_err = PanelErrores(add(d["errores"]))

        self.caja_f = TextBox(add([0.06, 0.21, 0.43, 0.05]), "y = ", initial=expresion)
        self.caja_rango = TextBox(add([0.58, 0.21, 0.11, 0.05]), "rango ",
                                  initial=f"{xmin:g}, {xmax:g}")
        self.caja_paso = TextBox(add([0.77, 0.21, 0.06, 0.05]), "paso Δx ", initial=f"{paso:g}")
        self.caja_altura = TextBox(add([0.92, 0.21, 0.05, 0.05]), "altura máx. ",
                                   initial=str(altura))

        def boton(x, texto, color):
            return Button(add([x, 0.11, 0.095, 0.055]), texto, color=color, hovercolor="#dddddd")

        self.btn_inicio = boton(0.06, "Inicio", "#b7e4c7")
        self.btn_pausa = boton(0.165, "Pausa", "#ffe8a3")
        self.btn_paso = boton(0.27, "Paso", "#cfe2ff")
        self.btn_punto = boton(0.375, "Sig. punto", "#cfe2ff")
        self.btn_todo = boton(0.48, "Fin", "#e2d4f5")
        self.btn_azar = boton(0.585, "Aleatoria", "#ffc8dd")
        self.slider = Slider(add([0.77, 0.125, 0.17, 0.03]), "velocidad ", 0, 100,
                             valinit=self._valor_slider(velocidad))
        self.estado = self.fig.text(0.06, 0.04, "", fontsize=10)

        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self._tick)
        self._aplicar_velocidad(velocidad)

        for caja in (self.caja_f, self.caja_rango, self.caja_paso, self.caja_altura):
            caja.on_submit(lambda _t: self.iniciar())      # Enter = Inicio
        self.btn_inicio.on_clicked(lambda _e: self.iniciar())
        self.btn_pausa.on_clicked(lambda _e: self.alternar_pausa())
        self.btn_paso.on_clicked(lambda _e: self.paso())
        self.btn_punto.on_clicked(lambda _e: self.siguiente_punto())
        self.btn_todo.on_clicked(lambda _e: self.mostrar_todo())
        self.btn_azar.on_clicked(lambda _e: self.aleatoria())
        self.slider.on_changed(self._cambiar_velocidad)

    def _mensaje(self, texto, error=False):
        self.estado.set_color("crimson" if error else "#222")
        self.estado.set_text(texto)

    # --- entradas ---------------------------------------------------------
    def _leer_numericos(self):
        try:
            xmin, xmax = (float(v) for v in re.split(r"[,; ]+", self.caja_rango.text.strip()))
        except ValueError:
            raise ErrorExpresion("Rango inválido. Usa el formato: -10, 10")
        if xmin >= xmax:
            raise ErrorExpresion("El rango debe cumplir min < max")
        try:
            paso = float(self.caja_paso.text)
            altura = int(float(self.caja_altura.text))
        except ValueError:
            raise ErrorExpresion("«paso Δx» debe ser un número y «altura máx.» un entero")
        if paso <= 0:
            raise ErrorExpresion("El paso Δx debe ser mayor que 0")
        if not 1 <= altura <= self.ALTURA_TOPE:
            raise ErrorExpresion(f"La altura máxima debe estar entre 1 y {self.ALTURA_TOPE}")
        n = int(round((xmax - xmin) / paso)) + 1
        if n < 2:
            raise ErrorExpresion("El paso es demasiado grande para el rango")
        if n > self.MAX_PUNTOS:
            raise ErrorExpresion(f"Demasiados puntos ({n}); el máximo es {self.MAX_PUNTOS}")
        return xmin, xmax, paso, altura

    def _poner_texto(self, texto):
        """Cambia el texto de la caja sin disparar 'Inicio'."""
        self.caja_f.eventson = False
        self.caja_f.set_val(texto)
        self.caja_f.eventson = True

    # --- control de la animación -------------------------------------------
    def iniciar(self, reproducir=True):
        """Lee las cajas, valida (incluida la altura máxima) y arranca desde cero."""
        self.timer.stop()
        self.reproduciendo = False
        try:
            xmin, xmax, paso, altura = self._leer_numericos()
            ev = Evaluador(self.caja_f.text, altura_maxima=altura)
            if len(ev.variables) > 1:
                raise ErrorExpresion(
                    f"Solo se grafica con una variable; encontradas: {list(ev.variables)}")
            n = int(round((xmax - xmin) / paso)) + 1
            xs = np.round(np.linspace(xmin, xmax, n), 12)
            y_ref = np.asarray(self.objetivo(xs), dtype=float)
        except ErrorExpresion as err:
            self._mensaje(f"Error: {err}", error=True)
            self.fig.canvas.draw_idle()
            return
        self.ev, self.xs, self.y_ref = ev, xs, y_ref
        self.pasos, self.i_paso, self.i_punto = [], 0, 0
        self.csv_ruta = self.csv_errores = self.csv_error = None
        self.vista_arbol.dibujar(ev.arbol)
        self.vista_grafica.preparar(ev, xs)
        self.panel.limpiar("Pulsa «Paso» o «Pausa/Seguir»\npara evaluar el primer punto.")
        self.panel_err.limpiar()
        self._mensaje(f"{n} puntos  ·  paso Δx = {paso:g}  ·  punto 0/{n}")
        self.reproduciendo = reproducir
        self.btn_pausa.label.set_text("Pausa" if reproducir else "Seguir")
        if reproducir:
            self.timer.start()
        self.fig.canvas.draw_idle()

    def aleatoria(self):
        """Genera una candidata aleatoria (respetando la altura máxima) y la inicia."""
        try:
            xmin, xmax, _paso, altura = self._leer_numericos()
        except ErrorExpresion as err:
            self._mensaje(f"Error: {err}", error=True)
            self.fig.canvas.draw_idle()
            return
        self.generador.altura_maxima = altura
        self._poner_texto(self.generador.generar_evaluador(xmin, xmax).expresion)
        self.iniciar()

    def pausar(self):
        self.reproduciendo = False
        self.timer.stop()
        self.btn_pausa.label.set_text("Seguir")

    def alternar_pausa(self):
        if self.ev is None:
            return
        if self.terminado:
            self.iniciar()
        elif self.reproduciendo:
            self.pausar()
            self.fig.canvas.draw_idle()
        else:
            self.reproduciendo = True
            self.btn_pausa.label.set_text("Pausa")
            self.timer.start()
            self.fig.canvas.draw_idle()

    def _cambiar_velocidad(self, v):
        self._aplicar_velocidad(self._rate(v))
        if self.reproduciendo:
            self.timer.stop()
            self.timer.start()

    def _tick(self):
        if not self.reproduciendo:
            return
        for k in range(self.lote):
            self._avanzar(visual=(k == self.lote - 1))
            if self.terminado:
                break
        if self.terminado and self.pasos:
            self._pintar_paso()
            self._finalizar()
        self._refrescar()
        self.fig.canvas.draw_idle()

    # --- un paso = un nodo -------------------------------------------------
    @property
    def terminado(self):
        return self.ev is not None and self.i_punto >= len(self.xs)

    def _var(self):
        return self.ev.variables[0] if self.ev.variables else "x"

    def _cargar_punto(self, idx=None):
        idx = self.i_punto if idx is None else idx
        self.x_actual = float(self.xs[idx])
        self.punto_actual = idx + 1
        self.pasos = self.ev.pasos(*([self.x_actual] if self.ev.variables else []))
        self.i_paso = 0

    def _pintar_paso(self):
        paso = self.pasos[self.i_paso - 1]
        self.vista_arbol.resaltar(paso.nodo)
        self.panel.mostrar(self._var(), self.x_actual, self.punto_actual, len(self.xs),
                           self.pasos, self.i_paso)
        self._mensaje(f"Punto {self.punto_actual}/{len(self.xs)} · "
                      f"paso {self.i_paso}/{len(self.pasos)}")

    def _refrescar(self):
        """Actualiza gráfica y errores (una vez por cuadro)."""
        self.vista_grafica.refrescar(self._var() if self.ev else "x")
        k = self.vista_grafica.k
        if k:
            m = calcular_metricas(self.y_ref[:k], self.vista_grafica.ys[:k])
            self.panel_err.mostrar(m, len(self.xs))

    def _avanzar(self, visual=True):
        """Evalúa el siguiente nodo; al terminar el último marca el punto en la gráfica."""
        if self.ev is None or self.terminado:
            return
        if self.i_paso >= len(self.pasos):
            self._cargar_punto()
        paso = self.pasos[self.i_paso]
        self.i_paso += 1
        if self.i_paso == len(self.pasos):
            self.vista_grafica.registrar(self.i_punto, paso.valor)
            self.i_punto += 1
        if visual:
            self._pintar_paso()
        if self.terminado:
            self._finalizar()

    def _exportar(self):
        """Guarda los puntos (x, y) y las métricas en .csv, una sola vez por función."""
        if not self.exportar_csv or self.csv_ruta is not None or self.csv_error is not None:
            return
        try:
            k = self.i_punto
            ys = self.vista_grafica.ys[:k]
            self.csv_ruta = self.ev.exportar_csv(xs=self.xs[:k], ys=ys, carpeta=self.carpeta_csv)
            m = calcular_metricas(self.y_ref[:k], ys)
            self.csv_errores = exportar_metricas_csv(self.csv_ruta[:-4] + "_errores.csv", m)
            print(f"\ny = {self.ev.expresion}   ({k} puntos)\n{m.texto()}")
            print(f"CSV de puntos : {self.csv_ruta}\nCSV de errores: {self.csv_errores}")
        except (OSError, ErrorExpresion) as err:
            self.csv_error = str(err)

    def _finalizar(self):
        self.reproduciendo = False
        self.timer.stop()
        self.vista_grafica.mostrar_curva()
        self.btn_pausa.label.set_text("Reiniciar")
        self._exportar()
        texto = f"Listo: {len(self.xs)} puntos evaluados"
        if self.csv_ruta:
            texto += f"  ·  CSV: {self.csv_ruta}"
        elif self.csv_error:
            self._mensaje(f"{texto}  ·  No se pudo guardar el CSV: {self.csv_error}", error=True)
            return
        self._mensaje(texto)

    def paso(self):
        if self.ev is None:
            return
        if self.reproduciendo:
            self.pausar()
        self._avanzar()
        self._refrescar()
        self.fig.canvas.draw_idle()

    def siguiente_punto(self):
        """Completa los pasos del punto en curso."""
        if self.ev is None or self.terminado:
            return
        self.pausar()
        p = self.i_punto
        while not self.terminado and self.i_punto == p:
            self._avanzar(visual=False)
        if self.pasos:
            self._pintar_paso()
        if self.terminado:
            self._finalizar()
        self._refrescar()
        self.fig.canvas.draw_idle()

    def mostrar_todo(self):
        """Evalúa de golpe (vectorizado) los puntos que faltan y termina."""
        if self.ev is None:
            return
        self.pausar()
        if not self.terminado:
            i0 = self.i_punto
            resto = self.xs[i0:]
            ys = self.ev(resto) if self.ev.variables else np.full(resto.shape, self.ev())
            self.vista_grafica.registrar_lote(i0, ys)
            self.i_punto = len(self.xs)
            self._cargar_punto(len(self.xs) - 1)      # el panel muestra el último punto
            self.i_paso = len(self.pasos)
        if self.pasos:
            self._pintar_paso()
        self._finalizar()
        self._refrescar()
        self.fig.canvas.draw_idle()

    def mostrar(self):
        plt.show()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Árbol de expresión paso a paso, gráfica, CSV y errores (MSE/RMSE/NMSE/R²)")
    ap.add_argument("expresion", nargs="?", default=None,
                    help=f"por defecto: {EXPRESION_OBJETIVO}")
    ap.add_argument("--aleatoria", action="store_true", help="empezar con una expresión aleatoria")
    ap.add_argument("--rango", nargs=2, type=float, default=[-10, 10], metavar=("MIN", "MAX"))
    ap.add_argument("--paso", type=float, default=0.001, help="Δx entre puntos (por defecto 0.001)")
    ap.add_argument("--altura", type=int, default=6, help="altura máxima del árbol")
    ap.add_argument("--objetivo", default=None, metavar="EXPR",
                    help="ecuación de referencia para el error (por defecto: la ecuación objetivo)")
    ap.add_argument("--semilla", type=int, default=None, help="semilla del generador aleatorio")
    ap.add_argument("--velocidad", type=float, default=8.0, help="pasos por segundo (0.5–20000)")
    ap.add_argument("--carpeta-csv", default="salidas_csv", metavar="CARPETA",
                    help="donde se guardan los .csv (por defecto: salidas_csv)")
    ap.add_argument("--sin-csv", action="store_true", help="no generar archivos .csv")
    ap.add_argument("--guardar", metavar="ARCHIVO", help="guarda la figura y termina")
    ap.add_argument("--pasos", type=int, help="con --guardar: cuántos pasos (nodos) ejecutar")
    args = ap.parse_args()

    if args.guardar:
        plt.switch_backend("Agg")
    app = AplicacionArbol(args.expresion, args.rango[0], args.rango[1], args.paso,
                          args.altura, args.velocidad, args.semilla,
                          reproducir=not args.guardar, carpeta_csv=args.carpeta_csv,
                          exportar_csv=not args.sin_csv, objetivo=args.objetivo,
                          aleatoria=args.aleatoria)
    if args.guardar:
        if args.pasos is None:
            app.mostrar_todo()
        else:
            for _ in range(args.pasos):
                app._avanzar()
            app._refrescar()
        app.fig.savefig(args.guardar, dpi=110)
        print(f"Figura guardada en {args.guardar}")
    else:
        app.mostrar()


if __name__ == "__main__":
    main()