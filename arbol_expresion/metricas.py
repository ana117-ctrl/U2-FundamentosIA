"""
metricas.py — Métricas de error entre una referencia (y_i) y una estimación (ŷ_i).

Convención: `y` = valores de referencia (los "reales" / de la ecuación objetivo),
            `y_est` = valores estimados (los que calcula el Evaluador).

    MSE  = (1/n) Σ (y_i − ŷ_i)²
    RMSE = √MSE
    NMSE = 10·log10( Σ (|Y_n| − |X_n|)² / Σ |X_n|² )    [dB]   X = referencia, Y = estimación
    R²   = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²              ȳ = media de la referencia

Los pares donde alguno de los dos valores no es finito (nan/inf) se ignoran y se
reporta cuántos se usaron. NMSE = -inf dB significa error exactamente cero.

    from metricas import calcular_metricas
    m = calcular_metricas(y_real, y_estimado)
    print(m.texto())          # m.mse, m.rmse, m.nmse_db, m.r2
"""
import csv
import os
from dataclasses import dataclass

import numpy as np

__all__ = ["mse", "rmse", "nmse", "r2", "Metricas", "calcular_metricas",
           "exportar_metricas_csv"]


def _pares(y, y_est):
    y = np.asarray(y, dtype=float).ravel()
    e = np.asarray(y_est, dtype=float).ravel()
    if y.shape != e.shape:
        raise ValueError("y y y_est deben tener la misma cantidad de valores")
    ok = np.isfinite(y) & np.isfinite(e)
    return y[ok], e[ok]


def mse(y, y_est):
    y, e = _pares(y, y_est)
    return float(np.mean((y - e) ** 2)) if y.size else float("nan")


def rmse(y, y_est):
    return float(np.sqrt(mse(y, y_est)))


def nmse(y, y_est):
    """NMSE en dB, tal como en la fórmula: compara magnitudes |Y_n| contra |X_n|."""
    x, yy = _pares(y, y_est)            # x = referencia, yy = estimación
    if x.size == 0:
        return float("nan")
    num = float(np.sum((np.abs(yy) - np.abs(x)) ** 2))
    den = float(np.sum(np.abs(x) ** 2))
    if den == 0:
        return float("nan")
    if num == 0:
        return float("-inf")
    return float(10.0 * np.log10(num / den))


def r2(y, y_est):
    y, e = _pares(y, y_est)
    if y.size == 0:
        return float("nan")
    ss_res = float(np.sum((y - e) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass
class Metricas:
    mse: float
    rmse: float
    nmse_db: float
    r2: float
    n_validos: int
    n_total: int

    def como_dict(self):
        return {"MSE": self.mse, "RMSE": self.rmse, "NMSE_dB": self.nmse_db,
                "R2": self.r2, "puntos_validos": self.n_validos,
                "puntos_totales": self.n_total}

    def texto(self):
        return (f"MSE  = {self.mse:.6g}\nRMSE = {self.rmse:.6g}\n"
                f"NMSE = {self.nmse_db:.6g} dB\nR²   = {self.r2:.8f}\n"
                f"({self.n_validos} de {self.n_total} puntos válidos)")


def calcular_metricas(y, y_est):
    y_arr = np.asarray(y, dtype=float).ravel()
    ref, est = _pares(y, y_est)
    return Metricas(mse(y, y_est), rmse(y, y_est), nmse(y, y_est), r2(y, y_est),
                    int(ref.size), int(y_arr.size))


def exportar_metricas_csv(ruta, metricas, delimitador=","):
    """Guarda las métricas en un .csv (columnas: metrica, valor)."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo, delimiter=delimitador)
        w.writerow(["metrica", "valor"])
        for nombre, valor in metricas.como_dict().items():
            w.writerow([nombre, valor])
    return ruta
