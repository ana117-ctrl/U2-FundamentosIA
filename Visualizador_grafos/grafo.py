"""Lectura de los dos grafos. No necesita paquetes externos."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Arista:
    id: str
    origen: int
    destino: int
    peso: float
    curva: float = 0
    etiqueta_t: float = 0.5


@dataclass
class Grafo:
    nombre: str
    dirigido: bool
    posiciones: dict
    aristas: list
    imagen: str = ""
    notas: tuple = ()

    def __post_init__(self):
        ids = set()
        self.adyacencia = {n: [] for n in self.posiciones}
        self.por_id = {}
        for a in self.aristas:
            if a.id in ids:
                raise ValueError(f"Arista repetida: {a.id}")
            if a.origen not in self.posiciones or a.destino not in self.posiciones:
                raise ValueError(f"La arista {a.id} referencia un nodo inexistente.")
            if not math.isfinite(a.peso):
                raise ValueError("Todos los pesos deben ser números finitos.")
            ids.add(a.id)
            self.por_id[a.id] = a
            self.adyacencia[a.origen].append((a.destino, a.peso, a.id))
            if not self.dirigido and a.origen != a.destino:
                self.adyacencia[a.destino].append((a.origen, a.peso, a.id))
        for vecinos in self.adyacencia.values():
            vecinos.sort(key=lambda x: (x[0], x[1], x[2]))

    @property
    def nodos(self):
        return sorted(self.posiciones)

    def arcos(self):
        """En un grafo no dirigido, cada arista aporta sus dos sentidos."""
        for a in self.aristas:
            yield a.origen, a.destino, a.peso, a.id
            if not self.dirigido and a.origen != a.destino:
                yield a.destino, a.origen, a.peso, a.id

    def validar_extremos(self, origen, destino):
        if origen not in self.posiciones or destino not in self.posiciones:
            raise ValueError("El origen y el destino deben existir en el grafo.")


def cargar_grafo(tipo="positivo"):
    if tipo not in ("positivo", "bellman_ford"):
        raise ValueError("Tipo de grafo desconocido.")
    data = json.loads((BASE / "datos" / f"{tipo}.json").read_text(encoding="utf-8"))
    nodos = {n["id"]: (n["x"], n["y"]) for n in data["nodos"]}
    if len(nodos) != len(data["nodos"]):
        raise ValueError("Hay identificadores de nodo repetidos.")
    return Grafo(data["nombre"], data["dirigido"], nodos,
                 [Arista(**a) for a in data["aristas"]],
                 data.get("imagen", ""), tuple(data.get("notas", [])))
