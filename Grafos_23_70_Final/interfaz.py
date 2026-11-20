"""Interfaz de escritorio con Tkinter: grafo, procedimiento y matriz."""
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from grafo import BASE, cargar_grafo
from algoritmos import ALGORITMOS
from algoritmos.comun import numero

BG = "#0b1220"
PANEL = "#131f30"
CANVAS = "#0e1929"
TEXT = "#e6edf7"
MUTED = "#a5b5c8"
GREEN = "#56e4b3"
BLUE = "#64b9ff"
YELLOW = "#ffc857"
PURPLE = "#bb9aff"
RED = "#ff9a91"

AYUDA = {
    "Dijkstra": "Selecciona la menor distancia pendiente y relaja sus vecinos. Los nodos cerrados tienen distancia definitiva. Requiere pesos no negativos.",
    "Bellman–Ford": "Recorre todos los arcos hasta |V|−1 veces. Se detiene si una pasada no cambia nada y revisa ciclos negativos alcanzables. Respeta las flechas y permite pesos negativos.",
    "Warshall–Floyd": "Cada etapa usa un nodo k como intermedio para todos los pares (i,j). D(i,j) = min(D(i,j), D(i,k)+D(k,j)). Abre «Matriz» para ver las 30 × 30 distancias.",
    "A*": "Prioriza f(n)=g(n)+h(n). La heurística es distancia geométrica escalada por el menor peso/longitud. Es consistente y no sobreestima; la posición visual por sí sola no es un costo.",
}


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Grafos · Ruta 23 → 70 | Dijkstra, Bellman–Ford, Warshall–Floyd y A*")
        ancho = min(1380, self.winfo_screenwidth() - 40)
        alto = min(880, self.winfo_screenheight() - 75)
        self.geometry(f"{ancho}x{alto}")
        self.minsize(1024, 650)
        self.configure(bg=BG)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self.algoritmo = tk.StringVar(value="Dijkstra")
        self.origen_var = tk.StringVar(value="23")
        self.destino_var = tk.StringVar(value="70")
        self.velocidad = tk.DoubleVar(value=0.35)
        self.posicion = tk.DoubleVar(value=0)
        self.auto_id = None
        self.ruta_id = None
        self.indice = 0
        self.pasos = []
        self.escala = 1.0
        self.aspecto_x = 1.0
        self.desplazamiento = (0.0, 0.0)
        self._arrastre = None
        self._ajustar_pendiente = None
        self._cambiando_barra = False
        self.matriz_ventana = None
        self.matriz_tabla = None
        self.ruta_visible = None
        self._estilos()
        self._construir()
        self.cargar()
        self.bind("<space>", self._tecla_reproducir)
        self.bind("<Right>", lambda e: self._tecla_paso(e, 1))
        self.bind("<Left>", lambda e: self._tecla_paso(e, -1))
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    def _estilos(self):
        estilo = ttk.Style(self)
        estilo.theme_use("clam")
        estilo.configure(".", font=("DejaVu Sans", 10), background=PANEL, foreground=TEXT)
        estilo.configure("TFrame", background=BG)
        estilo.configure("Card.TFrame", background=PANEL)
        estilo.configure("TLabel", background=BG, foreground=TEXT)
        estilo.configure("Muted.TLabel", foreground=MUTED)
        estilo.configure("Card.TLabel", background=PANEL)
        estilo.configure("TButton", padding=(10, 7), background="#26384f", foreground=TEXT, borderwidth=0)
        estilo.map("TButton", background=[("active", "#36506e"), ("disabled", "#1b283b")],
                   foreground=[("disabled", "#66778b")])
        estilo.configure("Accent.TButton", background=GREEN, foreground="#08291f")
        estilo.map("Accent.TButton", background=[("active", "#89f2ce")])
        estilo.configure("TCombobox", fieldbackground="#23354b", background="#23354b", foreground=TEXT,
                         arrowcolor=TEXT, padding=6)
        estilo.map("TCombobox", fieldbackground=[("readonly", "#23354b")], foreground=[("readonly", TEXT)])
        estilo.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                         rowheight=25, borderwidth=0)
        estilo.configure("Treeview.Heading", background="#26384f", foreground=TEXT, padding=5)
        estilo.map("Treeview", background=[("selected", "#32547c")])
        estilo.configure("TNotebook", background=BG, borderwidth=0)
        estilo.configure("TNotebook.Tab", background="#22334a", foreground=TEXT, padding=(12, 7))
        estilo.map("TNotebook.Tab", background=[("selected", "#34506b")])
        estilo.configure("Horizontal.TScale", background=BG, troughcolor="#26364b")
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _construir(self):
        cabecera = ttk.Frame(self, padding=(18, 12, 18, 5))
        cabecera.grid(row=0, column=0, sticky="ew")
        ttk.Label(cabecera, text="GRAFOS  /  RUTAS MÍNIMAS", foreground=GREEN,
                  font=("DejaVu Sans", 10, "bold")).pack(anchor="w")
        ttk.Label(cabecera, text="Explora el camino, paso a paso", font=("DejaVu Sans", 20, "bold")).pack(anchor="w", pady=(4, 0))

        ajustes = ttk.Frame(self, padding=(18, 5, 18, 7))
        ajustes.grid(row=1, column=0, sticky="ew")
        ttk.Label(ajustes, text="Algoritmo").pack(side="left", padx=(0, 8))
        selector = ttk.Combobox(ajustes, textvariable=self.algoritmo, values=list(ALGORITMOS), state="readonly", width=19)
        selector.pack(side="left")
        selector.bind("<<ComboboxSelected>>", self.cargar)
        ttk.Label(ajustes, text="Origen").pack(side="left", padx=(15, 6))
        self.origen_combo = ttk.Combobox(ajustes, textvariable=self.origen_var, width=5, state="readonly")
        self.origen_combo.pack(side="left")
        ttk.Label(ajustes, text="Destino").pack(side="left", padx=(12, 6))
        self.destino_combo = ttk.Combobox(ajustes, textvariable=self.destino_var, width=5, state="readonly")
        self.destino_combo.pack(side="left")
        self.origen_combo.bind("<<ComboboxSelected>>", self.cargar)
        self.destino_combo.bind("<<ComboboxSelected>>", self.cargar)
        ttk.Button(ajustes, text="Imagen original", command=self.ver_imagen).pack(side="right", padx=(6, 0))
        ttk.Button(ajustes, text="Aristas", command=self.ver_aristas).pack(side="right", padx=(6, 0))
        self.boton_matriz = ttk.Button(ajustes, text="Matriz", command=self.ver_matriz)
        self.boton_matriz.pack(side="right", padx=(6, 0))

        controles = ttk.Frame(self, padding=(18, 3, 18, 7))
        controles.grid(row=2, column=0, sticky="ew")
        self.boton_play = ttk.Button(controles, text="▶ Reproducir", style="Accent.TButton", command=self.reproducir)
        self.boton_play.pack(side="left", padx=(0, 5))
        ttk.Button(controles, text="◀ Anterior", command=lambda: self.avanzar(-1)).pack(side="left", padx=3)
        ttk.Button(controles, text="Paso ▶", command=lambda: self.avanzar(1)).pack(side="left", padx=3)
        ttk.Button(controles, text="Reiniciar", command=lambda: self.ir(0)).pack(side="left", padx=3)
        ttk.Button(controles, text="Ver resultado", command=self.ver_resultado).pack(side="left", padx=3)
        ttk.Label(controles, text="Pausa entre pasos", style="Muted.TLabel").pack(side="left", padx=(16, 5))
        ttk.Scale(controles, from_=0.04, to=1.5, variable=self.velocidad, length=110).pack(side="left")
        self.rapidez_label = ttk.Label(controles, text="0.35 s", width=7, style="Muted.TLabel")
        self.rapidez_label.pack(side="left", padx=(5, 0))
        self.velocidad.trace_add("write", lambda *_: self.rapidez_label.configure(text=f"{self.velocidad.get():.2f} s"))

        cuerpo = ttk.Panedwindow(self, orient="horizontal")
        cuerpo.grid(row=3, column=0, sticky="nsew", padx=18, pady=(2, 5))
        izquierda = ttk.Frame(cuerpo, style="Card.TFrame")
        derecha = ttk.Frame(cuerpo, style="Card.TFrame", padding=12, width=370)
        cuerpo.add(izquierda, weight=7)
        cuerpo.add(derecha, weight=3)
        izquierda.columnconfigure(0, weight=1)
        izquierda.rowconfigure(1, weight=1)
        derecha.columnconfigure(0, weight=1)
        derecha.rowconfigure(2, weight=3)
        derecha.rowconfigure(5, weight=4)
        self.grafo_label = ttk.Label(izquierda, style="Card.TLabel", padding=(12, 10))
        self.grafo_label.grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(izquierda, bg=CANVAS, highlightthickness=0, width=850, height=485)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._redimensionar)
        self.canvas.bind("<ButtonPress-1>", self._iniciar_arrastre)
        self.canvas.bind("<B1-Motion>", self._mover)
        self.canvas.bind("<ButtonRelease-1>", self._terminar_arrastre)
        self.canvas.bind("<Button-4>", lambda e: self.zoom(1.15, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self.zoom(1 / 1.15, e.x, e.y))
        self.canvas.bind("<MouseWheel>", lambda e: self.zoom(1.15 if e.delta > 0 else 1 / 1.15, e.x, e.y))
        inferior = ttk.Frame(izquierda, style="Card.TFrame", padding=7)
        inferior.grid(row=2, column=0, sticky="ew")
        ttk.Button(inferior, text="−", width=2, command=lambda: self.zoom(0.85)).pack(side="left", padx=2)
        ttk.Button(inferior, text="+", width=2, command=lambda: self.zoom(1.15)).pack(side="left", padx=2)
        ttk.Button(inferior, text="Ajustar", command=self.ajustar).pack(side="left", padx=(3, 10))
        ttk.Label(inferior, text="Rueda: zoom · Arrastra: mover · Clic: nodo", style="Card.TLabel",
                  foreground=MUTED, font=("DejaVu Sans", 8)).pack(side="left")
        leyenda = tk.Frame(izquierda, bg=PANEL)
        leyenda.grid(row=3, column=0, sticky="ew", padx=9, pady=(0, 8))
        for txt, color in [("Inicio", PURPLE), ("Meta", RED), ("Actual / k", YELLOW),
                           ("Pendiente", BLUE), ("Cerrado", "#668196"), ("Ruta", GREEN)]:
            tk.Label(leyenda, text="● " + txt, bg=PANEL, fg=color, font=("DejaVu Sans", 8)).pack(side="left", padx=4)

        ttk.Label(derecha, text="PROCEDIMIENTO", style="Card.TLabel", foreground=GREEN,
                  font=("DejaVu Sans", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.paso_label = ttk.Label(derecha, text="", style="Card.TLabel", wraplength=340,
                                    font=("DejaVu Sans", 12, "bold"))
        self.paso_label.grid(row=1, column=0, sticky="ew", pady=(7, 6))
        caja_detalle = ttk.Frame(derecha, style="Card.TFrame")
        caja_detalle.grid(row=2, column=0, sticky="nsew")
        self.detalle = tk.Text(caja_detalle, bg=CANVAS, fg=TEXT, insertbackground=TEXT, wrap="word",
                               relief="flat", padx=10, pady=8, font=("DejaVu Sans", 9), height=7, width=37)
        scroll_detalle = ttk.Scrollbar(caja_detalle, command=self.detalle.yview)
        self.detalle.configure(yscrollcommand=scroll_detalle.set)
        self.detalle.pack(side="left", fill="both", expand=True)
        scroll_detalle.pack(side="right", fill="y")
        self.metricas = ttk.Label(derecha, style="Card.TLabel", foreground=MUTED, font=("DejaVu Sans", 8))
        self.metricas.grid(row=3, column=0, sticky="w", pady=(7, 6))
        self.tabla_label = ttk.Label(derecha, text="Distancias desde el origen", style="Card.TLabel", font=("DejaVu Sans", 9, "bold"))
        self.tabla_label.grid(row=4, column=0, sticky="w", pady=(0, 5))
        caja_tabla = ttk.Frame(derecha, style="Card.TFrame")
        caja_tabla.grid(row=5, column=0, sticky="nsew")
        cols = ("nodo", "d", "pred", "estado")
        self.tabla = ttk.Treeview(caja_tabla, columns=cols, show="headings", height=6)
        for col, txt, width in zip(cols, ("Nodo", "d / g", "Previo", "Estado"), (48, 62, 58, 100)):
            self.tabla.heading(col, text=txt)
            self.tabla.column(col, width=width, minwidth=40, anchor="center")
        scroll = ttk.Scrollbar(caja_tabla, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tabla.bind("<<TreeviewSelect>>", self._seleccionar_fila)
        self.tabla.tag_configure("actual", foreground=YELLOW)
        self.tabla.tag_configure("ruta", foreground=GREEN)
        self.tabla.tag_configure("cerrado", foreground="#8ea3b8")
        self.ruta_label = ttk.Label(derecha, text="", style="Card.TLabel", foreground=GREEN, wraplength=340,
                                    font=("DejaVu Sans", 11, "bold"))
        self.ruta_label.grid(row=6, column=0, sticky="ew", pady=(10, 5))
        ttk.Button(derecha, text="Guardar procedimiento (.txt)", command=self.exportar).grid(row=7, column=0, sticky="ew")
        derecha.bind("<Configure>", lambda e: [w.configure(wraplength=max(230, e.width-28)) for w in (self.paso_label, self.ruta_label)])

        pie = ttk.Frame(self, padding=(18, 0, 18, 9))
        pie.grid(row=4, column=0, sticky="ew")
        self.progreso_label = ttk.Label(pie, text="", width=16, style="Muted.TLabel")
        self.progreso_label.pack(side="left")
        self.barra = ttk.Scale(pie, from_=0, to=1, variable=self.posicion, command=self._barra_mover)
        self.barra.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Button(pie, text="Cómo funciona", command=self.ayuda).pack(side="right")
        self.info = ttk.Label(self, text="", style="Muted.TLabel", padding=(18, 0, 18, 7), font=("DejaVu Sans", 8))
        self.info.grid(row=5, column=0, sticky="ew")

    def cargar(self, event=None):
        self.pausar()
        self.cancelar_ruta()
        nombre = self.algoritmo.get()
        self.grafo = cargar_grafo("bellman_ford" if nombre == "Bellman–Ford" else "positivo")
        for combo, variable, defecto in ((self.origen_combo, self.origen_var, 23), (self.destino_combo, self.destino_var, 70)):
            combo.configure(values=self.grafo.nodos)
            if int(variable.get()) not in self.grafo.posiciones:
                variable.set(str(defecto))
        self.origen, self.destino = int(self.origen_var.get()), int(self.destino_var.get())
        try:
            self.pasos = list(ALGORITMOS[nombre](self.grafo, self.origen, self.destino))
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("No se pudo calcular", str(exc), parent=self)
            return
        self.indice = 0
        self.barra.configure(to=max(1, len(self.pasos) - 1))
        self.grafo_label.configure(text=f"{self.grafo.nombre}  ·  {len(self.grafo.nodos)} nodos / {len(self.grafo.aristas)} aristas")
        self.boton_matriz.configure(state="normal" if nombre == "Warshall–Floyd" else "disabled")
        if nombre != "Warshall–Floyd" and self.matriz_ventana is not None and self.matriz_ventana.winfo_exists():
            self.matriz_ventana.destroy()
            self.matriz_ventana = None
        self.info.configure(text=("Flechas y pesos negativos conservados. Nodo inferior: 666." if self.grafo.dirigido else
                                  "Grafo no dirigido: cada línea puede recorrerse en ambos sentidos. Nodo inferior: 66."))
        self.after_idle(self.ajustar)
        self.mostrar()

    def _redimensionar(self, event):
        if self._ajustar_pendiente is not None:
            self.after_cancel(self._ajustar_pendiente)
        self._ajustar_pendiente = self.after(90, self.ajustar)

    def ajustar(self):
        self._ajustar_pendiente = None
        ancho, alto = self.canvas.winfo_width(), self.canvas.winfo_height()
        if ancho < 20 or alto < 20:
            return
        xs, ys = zip(*self.grafo.posiciones.values())
        margen = 38
        self.escala = min((ancho - 2*margen) / (max(xs)-min(xs)), (alto - 2*margen) / (max(ys)-min(ys)))
        # Aprovecha pantallas anchas sin cambiar la topología ni las coordenadas
        # originales utilizadas por la heurística; los círculos permanecen redondos.
        self.aspecto_x = min(1.6, (ancho-2*margen)/(max(xs)-min(xs))/self.escala)
        self.desplazamiento = ((ancho-(max(xs)+min(xs))*self.escala*self.aspecto_x)/2,
                               (alto-(max(ys)+min(ys))*self.escala)/2)
        self.dibujar()

    def punto(self, xy):
        return (xy[0]*self.escala*self.aspecto_x+self.desplazamiento[0], xy[1]*self.escala+self.desplazamiento[1])

    def zoom(self, factor, x=None, y=None):
        x = self.canvas.winfo_width()/2 if x is None else x
        y = self.canvas.winfo_height()/2 if y is None else y
        escala_nueva = max(0.12, min(4.0, self.escala*factor))
        factor = escala_nueva/self.escala
        dx, dy = self.desplazamiento
        self.desplazamiento = (x-(x-dx)*factor, y-(y-dy)*factor)
        self.escala = escala_nueva
        self.dibujar()

    def _iniciar_arrastre(self, e):
        self._arrastre = (e.x, e.y, self.desplazamiento)

    def _mover(self, e):
        if self._arrastre:
            x, y, (dx, dy) = self._arrastre
            self.desplazamiento = (dx+e.x-x, dy+e.y-y)
            self.dibujar()

    def _terminar_arrastre(self, e):
        if self._arrastre and abs(e.x-self._arrastre[0])+abs(e.y-self._arrastre[1]) < 5:
            for nodo, xy in self.grafo.posiciones.items():
                x, y = self.punto(xy)
                if math.hypot(e.x-x, e.y-y) < max(12, 22*self.escala):
                    self.tabla.selection_set(str(nodo))
                    self.tabla.see(str(nodo))
                    self._nodo_info(nodo)
                    break
        self._arrastre = None

    def _geometria_arista(self, a, radio):
        x1, y1 = self.punto(self.grafo.posiciones[a.origen])
        x2, y2 = self.punto(self.grafo.posiciones[a.destino])
        largo = max(1, math.hypot(x2-x1, y2-y1))
        curva = a.curva*self.escala
        cx, cy = ((x1+x2)/2-(y2-y1)/largo*curva, (y1+y2)/2+(x2-x1)/largo*curva)
        def bezier(t):
            return ((1-t)**2*x1+2*t*(1-t)*cx+t*t*x2,
                    (1-t)**2*y1+2*t*(1-t)*cy+t*t*y2)
        margen = min(0.40, (radio+2)/largo)
        puntos = [bezier(margen+(1-2*margen)*i/24) for i in range(25)]
        etiqueta = bezier(a.etiqueta_t)
        return puntos, etiqueta

    def dibujar(self):
        if not self.pasos:
            return
        c, p = self.canvas, self.pasos[self.indice]
        c.delete("all")
        radio = max(11, min(28, 23*self.escala))
        font_nodo = max(9, min(14, round(18*self.escala)))
        font_peso = max(8, min(12, round(15*self.escala)))
        ruta_ids = p.aristas_ruta if self.ruta_visible is None else p.aristas_ruta[:self.ruta_visible]
        ruta = set(ruta_ids)
        etiquetas = []
        # Primero aristas neutras; encima, las que se examinan y la ruta final.
        orden = sorted(self.grafo.aristas, key=lambda a: (a.id in ruta, a.id == p.arista))
        for a in orden:
            color, grosor = "#52677d", 1.2
            if a.id == p.arista:
                color, grosor = YELLOW, 3
            if a.id in ruta:
                color, grosor = GREEN, 4
            puntos, (lx, ly) = self._geometria_arista(a, radio)
            opciones = {"fill": color, "width": grosor, "joinstyle": "round"}
            if self.grafo.dirigido:
                opciones.update(arrow=tk.LAST, arrowshape=(9, 11, 4))
            c.create_line(*[v for xy in puntos for v in xy], **opciones)
            etiquetas.append((a, lx, ly, color))
        for a, x, y, color in etiquetas:
            fg = GREEN if a.id in ruta else YELLOW if a.id == p.arista else RED if a.peso < 0 else "#c3d1df"
            t = c.create_text(x, y, text=numero(a.peso), fill=fg, font=("DejaVu Sans", font_peso, "bold"))
            box = c.bbox(t)
            rect = c.create_rectangle(box[0]-3, box[1]-1, box[2]+3, box[3]+1, fill=CANVAS, outline="")
            c.tag_lower(rect, t)
        nodos_ruta = set(p.ruta if self.ruta_visible is None else p.ruta[:self.ruta_visible+1])
        for n, xy in self.grafo.posiciones.items():
            x, y = self.punto(xy)
            color, contorno, fg = "#20344b", "#637b93", TEXT
            if n in p.frontera:
                color, contorno = "#174367", BLUE
            if n in p.cerrados:
                color, contorno = "#304358", "#8296ab"
            if n in nodos_ruta:
                color, contorno, fg = "#164d40", GREEN, "#d7fff1"
            if n == p.actual:
                color, contorno, fg = "#624713", YELLOW, "#fff3cc"
            if n == self.origen:
                contorno = PURPLE
            if n == self.destino:
                contorno = RED
            c.create_oval(x-radio, y-radio, x+radio, y+radio, fill=color, outline=contorno, width=2)
            c.create_text(x, y, text=str(n), fill=fg, font=("DejaVu Sans", font_nodo, "bold"))

    def mostrar(self):
        p = self.pasos[self.indice]
        self.paso_label.configure(text=p.titulo)
        self.detalle.configure(state="normal")
        self.detalle.delete("1.0", "end")
        self.detalle.insert("1.0", p.detalle)
        self.detalle.configure(state="disabled")
        self.metricas.configure(text=f"Comparaciones: {p.comparaciones}   ·   Mejoras: {p.relajaciones}")
        self.progreso_label.configure(text=f"Paso {self.indice+1} / {len(self.pasos)}")
        self._cambiando_barra = True
        self.posicion.set(self.indice)
        self._cambiando_barra = False
        if p.final:
            self.ruta_label.configure(text=(" → ".join(map(str, p.ruta)) + f"\nCosto mínimo: {numero(p.costo)}") if p.ruta
                                      else "Sin mínimo finito" if p.error else "No hay ruta al destino")
        else:
            self.ruta_label.configure(text=f"Distancia provisional a {self.destino}: {numero(p.distancias[self.destino])}")
        self.tabla_label.configure(text=f"Distancias desde {self.origen}" + (" · clic: ver h y f" if p.heuristica else ""))
        for n in self.grafo.nodos:
            pred = p.predecesores.get(n)
            total_final = p.final and self.algoritmo.get() in ("Bellman–Ford", "Warshall–Floyd")
            estado = ("−∞" if p.distancias[n] == -math.inf else "∞" if p.distancias[n] == math.inf else
                      "Fijo" if n in p.cerrados else "Final" if total_final else "Provisional")
            tags = ("ruta",) if n in p.ruta else ("actual",) if n == p.actual else ("cerrado",) if n in p.cerrados else ()
            valores = (n, numero(p.distancias[n]), pred[0] if pred else "—", estado)
            if self.tabla.exists(str(n)):
                self.tabla.item(str(n), values=valores, tags=tags)
            else:
                self.tabla.insert("", "end", iid=str(n), values=valores, tags=tags)
        for iid in self.tabla.get_children():
            if int(iid) not in self.grafo.posiciones:
                self.tabla.delete(iid)
        if p.actual is not None:
            self.tabla.see(str(p.actual))
        self.dibujar()
        self._actualizar_matriz()

    def pausar(self):
        if self.auto_id is not None:
            self.after_cancel(self.auto_id)
            self.auto_id = None
        if hasattr(self, "boton_play"):
            self.boton_play.configure(text="▶ Reproducir")

    def cancelar_ruta(self):
        if self.ruta_id is not None:
            self.after_cancel(self.ruta_id)
            self.ruta_id = None
        self.ruta_visible = None

    def ir(self, indice):
        self.pausar()
        self.cancelar_ruta()
        self.indice = max(0, min(len(self.pasos)-1, int(indice)))
        self.mostrar()

    def avanzar(self, delta):
        self.ir(self.indice + delta)

    def reproducir(self):
        if self.auto_id is not None:
            self.pausar()
            return
        self.cancelar_ruta()
        if self.indice == len(self.pasos)-1:
            self.indice = 0
            self.mostrar()
        self.boton_play.configure(text="Ⅱ Pausar")
        self.auto_id = self.after(max(40, int(self.velocidad.get()*1000)), self._tick)

    def _tick(self):
        self.auto_id = None
        if self.indice < len(self.pasos)-1:
            self.indice += 1
            self.mostrar()
        if self.indice < len(self.pasos)-1:
            self.auto_id = self.after(max(40, int(self.velocidad.get()*1000)), self._tick)
        else:
            self.pausar()
            self.animar_ruta()

    def ver_resultado(self):
        self.ir(len(self.pasos)-1)
        self.animar_ruta()

    def animar_ruta(self):
        self.cancelar_ruta()
        if not self.pasos[self.indice].ruta:
            return
        self.ruta_visible = 0
        self.dibujar()
        self.ruta_id = self.after(200, self._trazar_segmento)

    def _trazar_segmento(self):
        self.ruta_id = None
        self.ruta_visible += 1
        self.dibujar()
        if self.ruta_visible < len(self.pasos[self.indice].aristas_ruta):
            self.ruta_id = self.after(230, self._trazar_segmento)

    def _barra_mover(self, valor):
        if not self._cambiando_barra and self.pasos:
            self.ir(round(float(valor)))

    def _tecla_reproducir(self, event):
        if not isinstance(event.widget, (ttk.Combobox, tk.Text)):
            self.reproducir()
            return "break"

    def _tecla_paso(self, event, delta):
        if not isinstance(event.widget, (ttk.Combobox, tk.Text)):
            self.avanzar(delta)
            return "break"

    def _seleccionar_fila(self, event):
        seleccion = self.tabla.selection()
        if seleccion:
            self._nodo_info(int(seleccion[0]))

    def _nodo_info(self, n):
        p = self.pasos[self.indice]
        txt = f"Nodo {n}: d / g = {numero(p.distancias[n])}"
        if p.heuristica:
            txt += f"   ·   h = {numero(p.heuristica[n])}   ·   f = {numero(p.distancias[n]+p.heuristica[n])}"
        if n in p.predecesores:
            txt += f"   ·   Predecesor: {p.predecesores[n][0]}"
        self.info.configure(text=txt)

    def _ventana_texto(self, titulo, texto):
        win = tk.Toplevel(self)
        win.title(titulo)
        win.geometry("780x560")
        win.configure(bg=BG)
        txt = tk.Text(win, bg=CANVAS, fg=TEXT, wrap="word", padx=18, pady=18, font=("DejaVu Sans", 10))
        scroll = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", texto)
        txt.configure(state="disabled")
        return win

    def ayuda(self):
        self._ventana_texto("Cómo funciona", AYUDA[self.algoritmo.get()] + "\n\nCONTROLES\n"
            "• Reproducir / Pausar: anima las operaciones.\n• Anterior y Paso: navega una operación a la vez.\n"
            "• Barra inferior: salta a cualquier etapa.\n• Ver resultado: traza la ruta final por segmentos.\n"
            "• Rueda y arrastre: amplía y mueve el grafo.\n• Clic en un nodo: inspecciona su distancia; A* añade h y f.\n"
            "• Matriz: muestra todas las distancias de la etapa de Warshall–Floyd.\n"
            "• Aristas e Imagen original: compara los datos transcritos con la imagen.\n"
            "• Guardar procedimiento: exporta todas las operaciones a un TXT.\n\n"
            "LECTURA DE LOS DATOS\n" + "\n".join("• "+x for x in self.grafo.notas) +
            "\n\nEn Dijkstra y A*, llegar al destino permite terminar: los demás nodos pendientes pueden conservar distancias provisionales.\n"
            "En Warshall–Floyd cada paso reúne las 900 comparaciones de una etapa. El panel enumera todas sus mejoras; usa su barra de desplazamiento.\n"
            "Un costo total positivo en Bellman–Ford no elimina los pesos negativos del grafo.")

    def ver_imagen(self):
        win = tk.Toplevel(self)
        win.title("Imagen original · " + self.grafo.nombre)
        win.geometry("1000x670")
        canvas = tk.Canvas(win, bg=BG, highlightthickness=0)
        sx = ttk.Scrollbar(win, orient="horizontal", command=canvas.xview)
        sy = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        try:
            win.imagen = tk.PhotoImage(file=str(BASE / "referencias" / self.grafo.imagen))
        except tk.TclError as exc:
            win.destroy()
            messagebox.showerror("Imagen no disponible", str(exc), parent=self)
            return
        canvas.create_image(0, 0, image=win.imagen, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
        canvas.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))

    def ver_aristas(self):
        simbolo = "→" if self.grafo.dirigido else "↔"
        self._ventana_texto("Aristas transcritas", self.grafo.nombre + "\n\n" +
            "\n".join(self.grafo.notas) + "\n\nID      ORIGEN      DESTINO      PESO\n" +
            "\n".join(f"{a.id}      {a.origen} {simbolo} {a.destino}      {numero(a.peso)}" for a in self.grafo.aristas))

    def ver_matriz(self):
        if not self.pasos[self.indice].matriz:
            return
        if self.matriz_ventana is not None and self.matriz_ventana.winfo_exists():
            self.matriz_ventana.lift()
            return
        win = self.matriz_ventana = tk.Toplevel(self)
        win.title("Matriz de distancias · Warshall–Floyd")
        win.geometry("1060x650")
        win.configure(bg=BG)
        self.matriz_titulo = ttk.Label(win, padding=12)
        self.matriz_titulo.pack(fill="x")
        marco = ttk.Frame(win)
        marco.pack(fill="both", expand=True, padx=10)
        marco.rowconfigure(0, weight=1)
        marco.columnconfigure(0, weight=1)
        columnas = [str(n) for n in self.grafo.nodos]
        tree = self.matriz_tabla = ttk.Treeview(marco, columns=columnas, show="tree headings")
        tree.heading("#0", text="i / j")
        tree.column("#0", width=54, minwidth=54, stretch=False)
        for n in columnas:
            tree.heading(n, text=n)
            tree.column(n, width=52, minwidth=52, anchor="center", stretch=False)
        sy = ttk.Scrollbar(marco, command=tree.yview)
        sx = ttk.Scrollbar(marco, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        tree.tag_configure("origen", foreground=GREEN)
        ttk.Button(win, text="Exportar esta matriz (.csv)", command=self.exportar_matriz).pack(pady=10)
        self._actualizar_matriz()

    def _actualizar_matriz(self):
        if self.matriz_ventana is None or not self.matriz_ventana.winfo_exists():
            return
        p = self.pasos[self.indice]
        if not p.matriz:
            return
        self.matriz_titulo.configure(text=p.titulo + f"  ·  La fila {self.origen} está en verde. Desplaza horizontalmente para ver todas las columnas.")
        for n, fila in zip(self.grafo.nodos, p.matriz):
            valores = [numero(x) for x in fila]
            tags = ("origen",) if n == self.origen else ()
            if self.matriz_tabla.exists(str(n)):
                self.matriz_tabla.item(str(n), values=valores, tags=tags)
            else:
                self.matriz_tabla.insert("", "end", iid=str(n), text=n, values=valores, tags=tags)

    def exportar_matriz(self):
        ruta = filedialog.asksaveasfilename(parent=self.matriz_ventana, defaultextension=".csv",
                                          initialfile=f"matriz_paso_{self.indice+1}.csv", filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        try:
            with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["i / j"]+self.grafo.nodos)
                for n, fila in zip(self.grafo.nodos, self.pasos[self.indice].matriz):
                    writer.writerow([n]+[numero(x) for x in fila])
        except OSError as exc:
            messagebox.showerror("No se pudo guardar", str(exc), parent=self)

    def exportar(self):
        ruta = filedialog.asksaveasfilename(parent=self, defaultextension=".txt",
            initialfile=f"procedimiento_{self.algoritmo.get().replace('*','_estrella')}_{self.origen}_{self.destino}.txt",
            filetypes=[("Texto", "*.txt")])
        if not ruta:
            return
        lineas = [self.algoritmo.get(), self.grafo.nombre, f"Origen: {self.origen}; destino: {self.destino}",
                  "", AYUDA[self.algoritmo.get()], ""]
        for i, p in enumerate(self.pasos, 1):
            lineas.extend([f"PASO {i}: {p.titulo}", p.detalle,
                           f"Comparaciones: {p.comparaciones}; mejoras: {p.relajaciones}", ""])
        try:
            Path(ruta).write_text("\n".join(lineas), encoding="utf-8")
            messagebox.showinfo("Guardado", "Se guardó el procedimiento completo.", parent=self)
        except OSError as exc:
            messagebox.showerror("No se pudo guardar", str(exc), parent=self)

    def cerrar(self):
        self.pausar()
        self.cancelar_ruta()
        if self._ajustar_pendiente is not None:
            self.after_cancel(self._ajustar_pendiente)
        self.destroy()
