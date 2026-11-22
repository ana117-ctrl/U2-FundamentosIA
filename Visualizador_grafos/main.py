#!/usr/bin/env python3
"""Inicio: python3 main.py. Comprobación sin ventana: --consola / --verificar."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualizador de cuatro algoritmos de caminos mínimos.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--consola", action="store_true", help="imprime las cuatro rutas sin abrir una ventana")
    grupo.add_argument("--verificar", action="store_true", help="ejecuta las pruebas automatizadas sin interfaz")
    args = parser.parse_args()
    if args.verificar:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "pruebas"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.consola:
        from grafo import cargar_grafo
        from algoritmos import ALGORITMOS
        from algoritmos.comun import numero
        for nombre, funcion in ALGORITMOS.items():
            g = cargar_grafo("bellman_ford" if nombre == "Bellman–Ford" else "positivo")
            fin = list(funcion(g, 23, 70))[-1]
            print(f"{nombre}: {' → '.join(map(str, fin.ruta))}; costo = {numero(fin.costo)}")
        return 0
    try:
        import tkinter as tk
        from interfaz import Aplicacion
    except ImportError:
        print("Falta Tkinter. En Zorin OS instala: sudo apt install python3-tk", file=sys.stderr)
        return 1
    try:
        app = Aplicacion()
        app.mainloop()
    except tk.TclError as exc:
        print(f"No se pudo abrir la interfaz: {exc}\n"
              "Ejecuta el programa desde una terminal de tu sesión gráfica de Zorin.\n"
              "Para probar solo los cálculos: python3 main.py --consola", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
