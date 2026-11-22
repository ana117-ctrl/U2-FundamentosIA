"""
Programación evolutiva (algoritmo genético binario) - conversión de C++ a Python.

Solo usa la librería estándar de Python (math, random, bisect, tkinter).
No necesita DEAP, numpy ni matplotlib.

Qué hace el código original en C++ (funciones.h / funciones.cpp):
  * Individuo = cadena de bits (genotipo) -> se decodifica a un número real x (fenotipo).
  * Longitud de la cadena: lcrom = ceil( log2( 1 + (x_max - x_min) / TOL ) ).
  * Población inicial aleatoria.
  * Selección por RULETA (puntuación acumulada) con ELITISMO (2 % de la población).
  * Reproducción: se eligen individuos con probabilidad prob_cruce y se cruzan por pares
    con cruce de UN punto; los hijos sustituyen a los padres (respetando la élite).
  * Mutación: cada bit se invierte con probabilidad prob_mut (la élite no muta).
  * Tres versiones:
      v1.01  -> MAXIMIZAR  f(x) = 20 + e - 20*exp(-0.2|x|) - exp(cos(2*pi*x)),  x en [0, 32]
      v2.0   -> MINIMIZAR  f(x) = -|x * sin(sqrt|x|)|  (se transforma la aptitud: cmax - f)
      varias -> MINIMIZAR  f(x) = suma_i  -x_i * sin(sqrt|x_i|)   (varias variables)

Esta versión añade una interfaz gráfica (tkinter) que dibuja CADA generación.
Ejecutar:  python3 algoritmo_genetico.py
"""

import math
import random
import bisect
import tkinter as tk
from tkinter import ttk, messagebox


# ----------------------------------------------------------------------
# Funciones objetivo (reciben una lista con los fenotipos de cada variable)
# ----------------------------------------------------------------------
def f_ackley_1d(xs):
    x = xs[0]
    return 20 + math.e - 20 * math.exp(-0.2 * abs(x)) - math.exp(math.cos(2 * math.pi * x))


def f_schwefel_abs_1d(xs):
    x = xs[0]
    return -abs(x * math.sin(math.sqrt(abs(x))))


def f_schwefel_n(xs):
    return sum(-x * math.sin(math.sqrt(abs(x))) for x in xs)


PROBLEMAS = {
    "v1.01  Maximizar (Ackley 1D)": dict(
        f=f_ackley_1d, maximizar=True, xmin=0, xmax=32, tol=0.001, nvar=1, multi=False),
    "v2.0   Minimizar (-|x·sin√|x||)": dict(
        f=f_schwefel_abs_1d, maximizar=False, xmin=-512, xmax=512, tol=0.001, nvar=1, multi=False),
    "Varias variables - Minimizar (Schwefel)": dict(
        f=f_schwefel_n, maximizar=False, xmin=-500, xmax=500, tol=0.001, nvar=2, multi=True),
}


# ----------------------------------------------------------------------
# Individuo  (typedef struct TIndividuo)
# ----------------------------------------------------------------------
class Individuo:
    def __init__(self):
        self.genes = []          # lista de cadenas de bits (una por variable)
        self.x = []              # fenotipos
        self.aptitud_bruta = 0.0 # valor de la función objetivo
        self.aptitud = 0.0       # aptitud transformada (siempre se MAXIMIZA)
        self.puntuacion = 0.0    # fitness / suma de fitness
        self.punt_acu = 0.0      # puntuación acumulada (ruleta)
        self.elite = False

    def copia(self):
        c = Individuo()
        c.genes = [g[:] for g in self.genes]
        c.x = self.x[:]
        c.aptitud_bruta = self.aptitud_bruta
        c.aptitud = self.aptitud
        c.puntuacion = self.puntuacion
        c.punt_acu = self.punt_acu
        c.elite = self.elite
        return c


# ----------------------------------------------------------------------
# Algoritmo genético (las funciones conservan los nombres del C++)
# ----------------------------------------------------------------------
class AlgoritmoGenetico:
    def __init__(self, f, maximizar, nvar, x_min, x_max, tol,
                 tam_pob, prob_cruce, prob_mut, pct_elite=2.0):
        self.f = f
        self.maximizar = maximizar
        self.nvar = nvar
        self.x_min = x_min
        self.x_max = x_max
        self.tam_pob = tam_pob
        self.prob_cruce = prob_cruce
        self.prob_mut = prob_mut
        self.pct_elite = pct_elite

        # lcrom = ceil(log(1 + (x_max - x_min)/TOL) / log(2))
        self.lcrom = max(1, math.ceil(math.log(1 + (x_max - x_min) / tol) / math.log(2)))

        self.generacion = 0
        self.pos_mejor = 0
        self.suma_aptitud = 0.0
        self.hist_mejor, self.hist_media, self.hist_peor = [], [], []

        self.pob = self.poblacion_inicial()
        self.revisar_adap_mini()
        self.evaluacion()

    # --- utilidades -----------------------------------------------------
    def mejor(self, a, b):
        """True si el valor bruto 'a' es mejor que 'b' según maximice o minimice."""
        return a > b if self.maximizar else a < b

    def bin_dec(self, genes):
        d, pot = 0, 1
        for bit in genes:
            d += pot * bit
            pot *= 2
        return d

    def decod(self, genes):
        x = self.bin_dec(genes) / (2 ** self.lcrom - 1)
        return self.x_min + (self.x_max - self.x_min) * x

    def adaptacion(self, ind):
        """Decodifica el individuo y evalúa la función objetivo."""
        ind.x = [self.decod(g) for g in ind.genes]
        ind.aptitud_bruta = self.f(ind.x)
        return ind.aptitud_bruta

    # --- población inicial ---------------------------------------------
    def genera_indiv(self):
        ind = Individuo()
        ind.genes = [[random.randint(0, 1) for _ in range(self.lcrom)]
                     for _ in range(self.nvar)]
        self.adaptacion(ind)
        return ind

    def poblacion_inicial(self):
        return [self.genera_indiv() for _ in range(self.tam_pob)]

    # --- transformación de la aptitud ----------------------------------
    def revisar_adap_mini(self):
        """Si se maximiza: aptitud = bruta. Si se minimiza: aptitud = cmax - bruta
        (así el mejor individuo, el de menor valor, tiene la mayor aptitud)."""
        if self.maximizar:
            for ind in self.pob:
                ind.aptitud = ind.aptitud_bruta
            return
        cmax = max(ind.aptitud_bruta for ind in self.pob)
        cmax += abs(cmax) / 100.0 or 1e-9     # cmax + cmax/100 (válido también si cmax < 0)
        for ind in self.pob:
            ind.aptitud = cmax - ind.aptitud_bruta

    # --- estadísticas ---------------------------------------------------
    def evaluacion(self):
        n = len(self.pob)
        self.suma_aptitud = sum(ind.aptitud for ind in self.pob)
        self.pos_mejor = 0
        for i, ind in enumerate(self.pob):
            if ind.aptitud > self.pob[self.pos_mejor].aptitud:
                self.pos_mejor = i
        acu = 0.0
        for ind in self.pob:
            ind.puntuacion = (ind.aptitud / self.suma_aptitud
                              if self.suma_aptitud > 0 else 1.0 / n)
            ind.punt_acu = acu + ind.puntuacion
            acu += ind.puntuacion

        brutas = [ind.aptitud_bruta for ind in self.pob]
        mejor = self.pob[self.pos_mejor].aptitud_bruta
        peor = min(brutas) if self.maximizar else max(brutas)
        self.hist_mejor.append(mejor)
        self.hist_peor.append(peor)
        self.hist_media.append(sum(brutas) / n)

    # --- selección: ruleta + elitismo ----------------------------------
    def seleccion(self):
        n = self.tam_pob
        tam_elite = int(n * self.pct_elite / 100.0)
        orden = sorted(range(n), key=lambda i: self.pob[i].aptitud, reverse=True)
        elite_idx = set(orden[:tam_elite])
        acu = [ind.punt_acu for ind in self.pob]

        nueva = []
        for i in range(n):
            if i in elite_idx:                      # ELITISMO: la élite se conserva
                sel = i
            else:                                   # ruleta
                r = random.random()
                sel = bisect.bisect_left(acu, r)
                if sel >= n:
                    sel = n - 1
            c = self.pob[sel].copia()
            c.elite = i in elite_idx
            nueva.append(c)
        self.pob = nueva

    # --- reproducción: cruce de un punto -------------------------------
    def cruce(self, padre1, padre2, puntos):
        hijo1, hijo2 = Individuo(), Individuo()
        for v in range(self.nvar):
            c = puntos[v]
            hijo1.genes.append(padre1.genes[v][:c] + padre2.genes[v][c:])
            hijo2.genes.append(padre2.genes[v][:c] + padre1.genes[v][c:])
        self.adaptacion(hijo1)
        self.adaptacion(hijo2)
        return hijo1, hijo2

    def reproduccion(self):
        sel_cruce = [i for i in range(self.tam_pob) if random.random() < self.prob_cruce]
        if len(sel_cruce) % 2 == 1:        # el número de seleccionados debe ser par
            sel_cruce.pop()

        for k in range(0, len(sel_cruce), 2):
            a, b = sel_cruce[k], sel_cruce[k + 1]
            pa, pb = self.pob[a], self.pob[b]
            if self.lcrom > 1:
                puntos = [random.randint(1, self.lcrom - 1) for _ in range(self.nvar)]
            else:
                puntos = [0] * self.nvar
            h1, h2 = self.cruce(pa, pb, puntos)

            # los hijos sustituyen a sus padres, respetando la élite
            if self.mejor(h1.aptitud_bruta, pa.aptitud_bruta) or not pa.elite:
                h1.elite = pa.elite
                self.pob[a] = h1
            if self.mejor(h2.aptitud_bruta, pb.aptitud_bruta) or not pb.elite:
                h2.elite = pb.elite
                self.pob[b] = h2

    # --- mutación -------------------------------------------------------
    def mutacion(self):
        for ind in self.pob:
            if ind.elite:                  # ELITISMO: la élite no muta
                continue
            mutado = False
            for g in ind.genes:
                for j in range(len(g)):
                    if random.random() < self.prob_mut:
                        g[j] = 1 - g[j]
                        mutado = True
            if mutado:
                self.adaptacion(ind)

    # --- una generación completa ---------------------------------------
    def siguiente_generacion(self):
        self.seleccion()
        self.reproduccion()
        self.mutacion()
        self.revisar_adap_mini()
        self.evaluacion()
        self.generacion += 1

    @property
    def mejor_individuo(self):
        return self.pob[self.pos_mejor]


# ----------------------------------------------------------------------
# Interfaz gráfica (tkinter): dibuja cada generación
# ----------------------------------------------------------------------
class App(tk.Tk):
    CAMPOS = [
        ("xmin", "Límite inferior (x_min)"),
        ("xmax", "Límite superior (x_max)"),
        ("tol", "Tolerancia (TOL)"),
        ("nvar", "N.º de variables"),
        ("tam_pob", "Tamaño de población"),
        ("num_gen", "N.º de generaciones"),
        ("p_cruce", "Prob. de cruce"),
        ("p_mut", "Prob. de mutación (por bit)"),
        ("elite", "Élite (%)"),
    ]
    DEFECTOS = dict(tam_pob="100", num_gen="100", p_cruce="0.6", p_mut="0.02", elite="2")

    def __init__(self):
        super().__init__()
        self.title("Algoritmo genético - evolución por generación")
        self.geometry("1200x720")
        self.minsize(900, 560)

        self.ag = None
        self.num_gen = 0
        self.problema = None
        self.curva = None
        self.running = False
        self.job = None
        self.vars = {}

        self._crear_panel()
        self._crear_canvas()
        self.cargar_preset()
        self.after(200, self.reiniciar)

    # --- construcción de la interfaz -----------------------------------
    def _crear_panel(self):
        panel = ttk.Frame(self, padding=10)
        panel.grid(row=0, column=0, sticky="ns")

        ttk.Label(panel, text="Problema").grid(row=0, column=0, sticky="w")
        self.cb = ttk.Combobox(panel, values=list(PROBLEMAS), state="readonly", width=34)
        self.cb.current(0)
        self.cb.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 8))
        self.cb.bind("<<ComboboxSelected>>", lambda e: (self.cargar_preset(), self.reiniciar()))

        fila = 2
        for clave, texto in self.CAMPOS:
            ttk.Label(panel, text=texto).grid(row=fila, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=self.DEFECTOS.get(clave, ""))
            e = ttk.Entry(panel, textvariable=v, width=10)
            e.grid(row=fila, column=1, sticky="e", pady=2)
            self.vars[clave] = v
            if clave == "nvar":
                self.entry_nvar = e
            fila += 1

        ttk.Label(panel, text="Pausa entre generaciones (ms)").grid(
            row=fila, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.velocidad = tk.Scale(panel, from_=0, to=1000, orient="horizontal", length=250)
        self.velocidad.set(150)
        self.velocidad.grid(row=fila + 1, column=0, columnspan=2, sticky="we")

        botones = ttk.Frame(panel)
        botones.grid(row=fila + 2, column=0, columnspan=2, pady=12, sticky="we")
        ttk.Button(botones, text="▶ Iniciar", command=self.iniciar).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(botones, text="⏸ Pausa", command=self.pausa).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(botones, text="⏭ Paso", command=self.paso).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(botones, text="↺ Reiniciar", command=self.reiniciar).grid(row=1, column=1, padx=2, pady=2)

        self.estado = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.estado, wraplength=260, justify="left").grid(
            row=fila + 3, column=0, columnspan=2, sticky="w")

        leyenda = ("Leyenda (gráfica superior):\n"
                   "● rojo = mejor individuo\n"
                   "● verde = élite\n"
                   "● azul = resto de la población")
        ttk.Label(panel, text=leyenda, justify="left").grid(
            row=fila + 4, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _crear_canvas(self):
        der = ttk.Frame(self)
        der.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        der.columnconfigure(0, weight=1)
        der.rowconfigure(0, weight=3)
        der.rowconfigure(1, weight=2)

        self.cv_pob = tk.Canvas(der, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.cv_pob.grid(row=0, column=0, sticky="nsew", padx=6, pady=(10, 4))
        self.cv_conv = tk.Canvas(der, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.cv_conv.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 10))
        self.cv_pob.bind("<Configure>", lambda e: self.dibujar())
        self.cv_conv.bind("<Configure>", lambda e: self.dibujar())

    # --- parámetros -----------------------------------------------------
    def cargar_preset(self):
        p = PROBLEMAS[self.cb.get()]
        self.vars["xmin"].set(str(p["xmin"]))
        self.vars["xmax"].set(str(p["xmax"]))
        self.vars["tol"].set(str(p["tol"]))
        self.vars["nvar"].set(str(p["nvar"]))
        self.entry_nvar.configure(state="normal" if p["multi"] else "disabled")

    def crear_ag(self):
        try:
            p = PROBLEMAS[self.cb.get()]
            xmin = float(self.vars["xmin"].get())
            xmax = float(self.vars["xmax"].get())
            tol = float(self.vars["tol"].get())
            nvar = int(self.vars["nvar"].get()) if p["multi"] else 1
            tam_pob = int(self.vars["tam_pob"].get())
            num_gen = int(self.vars["num_gen"].get())
            p_cruce = float(self.vars["p_cruce"].get())
            p_mut = float(self.vars["p_mut"].get())
            elite = float(self.vars["elite"].get())
            if xmax <= xmin or tol <= 0 or nvar < 1 or tam_pob < 2 or num_gen < 1:
                raise ValueError
            if not (0 <= p_cruce <= 1 and 0 <= p_mut <= 1 and 0 <= elite <= 100):
                raise ValueError
        except ValueError:
            messagebox.showerror("Parámetros inválidos", "Revisa los valores numéricos ingresados.")
            return False

        self.problema = p
        self.num_gen = num_gen
        self.ag = AlgoritmoGenetico(p["f"], p["maximizar"], nvar, xmin, xmax, tol,
                                    tam_pob, p_cruce, p_mut, elite)
        # curva de la función (solo para 1 variable)
        self.curva = None
        if nvar == 1:
            m = 600
            xs = [xmin + (xmax - xmin) * k / (m - 1) for k in range(m)]
            self.curva = [(x, p["f"]([x])) for x in xs]
        return True

    # --- control de ejecución ------------------------------------------
    def iniciar(self):
        if self.running:
            return
        if self.ag is None or self.ag.generacion >= self.num_gen:
            if not self.crear_ag():
                return
            self.dibujar()
        self.running = True
        self.tick()

    def tick(self):
        if not self.running:
            return
        if self.ag.generacion >= self.num_gen:
            self.terminar()
            return
        self.ag.siguiente_generacion()
        self.dibujar()
        if self.ag.generacion >= self.num_gen:
            self.terminar()
            return
        self.job = self.after(int(self.velocidad.get()), self.tick)

    def pausa(self):
        self.running = False
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None

    def paso(self):
        self.pausa()
        if self.ag is None or self.ag.generacion >= self.num_gen:
            if not self.crear_ag():
                return
        if self.ag.generacion < self.num_gen:
            self.ag.siguiente_generacion()
            self.dibujar()
            if self.ag.generacion >= self.num_gen:
                self.terminar()

    def reiniciar(self):
        self.pausa()
        if self.crear_ag():
            self.dibujar()

    def terminar(self):
        self.pausa()
        m = self.ag.mejor_individuo
        xs = ", ".join(f"{x:.5f}" for x in m.x)
        print(f"Aptitud del mejor ({self.ag.pos_mejor}) = {m.aptitud_bruta:.6f} en {xs}")
        self.dibujar(final=True)

    # --- dibujo ---------------------------------------------------------
    def _marco(self, cv, titulo, x0, x1, y0, y1, lx="", ly=""):
        cv.delete("all")
        w = max(cv.winfo_width(), 220)
        h = max(cv.winfo_height(), 160)
        ml, mr, mt, mb = 70, 20, 30, 42
        if x1 == x0:
            x1 = x0 + 1
        if y1 == y0:
            y0, y1 = y0 - 1, y1 + 1
        else:
            d = (y1 - y0) * 0.05
            y0, y1 = y0 - d, y1 + d
        px0, px1, py0, py1 = ml, w - mr, h - mb, mt
        fx = lambda v: px0 + (v - x0) / (x1 - x0) * (px1 - px0)
        fy = lambda v: py0 + (v - y0) / (y1 - y0) * (py1 - py0)
        small = ("TkDefaultFont", 8)

        cv.create_text(w / 2, 14, text=titulo, font=("TkDefaultFont", 10, "bold"))
        for k in range(6):
            vx = x0 + (x1 - x0) * k / 5
            X = fx(vx)
            cv.create_line(X, py0, X, py1, fill="#ececec")
            cv.create_text(X, py0 + 12, text=f"{vx:.4g}", font=small)
            vy = y0 + (y1 - y0) * k / 5
            Y = fy(vy)
            cv.create_line(px0, Y, px1, Y, fill="#ececec")
            cv.create_text(px0 - 6, Y, text=f"{vy:.4g}", font=small, anchor="e")
        cv.create_rectangle(px0, py1, px1, py0, outline="#888")
        cv.create_text((px0 + px1) / 2, h - 10, text=lx, font=small)
        cv.create_text(12, (py0 + py1) / 2, text=ly, font=small, angle=90)
        return fx, fy

    def _punto(self, cv, X, Y, r, color):
        cv.create_oval(X - r, Y - r, X + r, Y + r, fill=color, outline="")

    def dibujar(self, final=False):
        if self.ag is None:
            return
        ag = self.ag
        sufijo = "  (FINALIZADO)" if final else ""
        titulo = f"Generación {ag.generacion} / {self.num_gen}{sufijo}"

        # ---- población ---------------------------------------------------
        cv = self.cv_pob
        if ag.nvar == 1 and self.curva:
            ys = [c[1] for c in self.curva]
            fx, fy = self._marco(cv, titulo + "  -  población sobre f(x)",
                                 ag.x_min, ag.x_max, min(ys), max(ys), "x", "f(x)")
            pts = []
            for x, y in self.curva:
                pts += [fx(x), fy(y)]
            cv.create_line(*pts, fill="#999", width=1)
            coord = lambda ind: (ind.x[0], ind.aptitud_bruta)
        else:
            fx, fy = self._marco(cv, titulo + "  -  población (x1 vs x2)",
                                 ag.x_min, ag.x_max, ag.x_min, ag.x_max, "x1", "x2")
            if ag.nvar == 1:
                coord = lambda ind: (ind.x[0], 0)
            else:
                coord = lambda ind: (ind.x[0], ind.x[1])

        for i, ind in enumerate(ag.pob):
            if i == ag.pos_mejor:
                continue
            x, y = coord(ind)
            self._punto(cv, fx(x), fy(y), 3, "#2e8b57" if ind.elite else "#3b78d8")
        x, y = coord(ag.mejor_individuo)
        self._punto(cv, fx(x), fy(y), 6, "#d62728")

        # ---- convergencia -----------------------------------------------
        cv = self.cv_conv
        hm, hd, hp = ag.hist_mejor, ag.hist_media, ag.hist_peor
        todos = hm + hd + hp
        fx, fy = self._marco(cv, "Convergencia (valor de la función objetivo)",
                             0, max(self.num_gen, 1), min(todos), max(todos),
                             "generación", "f")
        for serie, color in ((hp, "#aaaaaa"), (hd, "#3b78d8"), (hm, "#d62728")):
            if len(serie) >= 2:
                pts = []
                for g, v in enumerate(serie):
                    pts += [fx(g), fy(v)]
                cv.create_line(*pts, fill=color, width=2)
            else:
                self._punto(cv, fx(0), fy(serie[0]), 3, color)
        w = cv.winfo_width()
        for k, (txt, col) in enumerate((("mejor", "#d62728"), ("media", "#3b78d8"), ("peor", "#aaaaaa"))):
            cv.create_line(w - 150 + k * 48, 22, w - 130 + k * 48, 22, fill=col, width=3)
            cv.create_text(w - 126 + k * 48, 22, text=txt, anchor="w", font=("TkDefaultFont", 8))

        # ---- texto de estado --------------------------------------------
        m = ag.mejor_individuo
        xs = ", ".join(f"{x:.4f}" for x in m.x[:4]) + (" ..." if len(m.x) > 4 else "")
        self.estado.set(f"Generación: {ag.generacion}/{self.num_gen}\n"
                        f"Mejor ({ag.pos_mejor}) = {m.aptitud_bruta:.6f}\n"
                        f"en x = {xs}\n"
                        f"Bits por variable (lcrom): {ag.lcrom}")


if __name__ == "__main__":
    App().mainloop()