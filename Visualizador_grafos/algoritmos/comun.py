"""Instantáneas inmutables por convención para reproducir y retroceder pasos."""
from dataclasses import dataclass, field
from math import inf


def numero(x):
    if x == inf:
        return "∞"
    if x == -inf:
        return "−∞"
    return str(int(x)) if float(x).is_integer() else f"{x:.3f}"


@dataclass
class Paso:
    titulo: str
    detalle: str
    distancias: dict = field(default_factory=dict)
    predecesores: dict = field(default_factory=dict)
    actual: int | None = None
    arista: str | None = None
    cerrados: tuple = ()
    frontera: tuple = ()
    heuristica: dict = field(default_factory=dict)
    matriz: tuple = ()
    intermedio: int | None = None
    cambios: tuple = ()
    comparaciones: int = 0
    relajaciones: int = 0
    pasada: int = 0
    ruta: tuple = ()
    aristas_ruta: tuple = ()
    costo: float = inf
    final: bool = False
    error: bool = False


def reconstruir(pred, origen, destino):
    """pred[v] = (nodo previo, id de arista); conserva aristas paralelas."""
    if origen == destino:
        return (origen,), ()
    nodos, aristas, vistos = [destino], [], set()
    actual = destino
    while actual != origen:
        if actual in vistos or actual not in pred:
            return (), ()
        vistos.add(actual)
        anterior, arista = pred[actual]
        aristas.append(arista)
        nodos.append(anterior)
        actual = anterior
    return tuple(reversed(nodos)), tuple(reversed(aristas))


def resultado(dist, pred, origen, destino, **kwargs):
    ruta, aristas = reconstruir(pred, origen, destino)
    if dist[destino] == inf:
        ruta, aristas = (), ()
    detalle = ("Ruta: " + " → ".join(map(str, ruta)) +
               f"\nCosto mínimo = {numero(dist[destino])}.") if ruta else (
                   f"No existe una ruta de {origen} a {destino} en este grafo.")
    return Paso("Resultado final" if ruta else "Destino inalcanzable", detalle,
                dist.copy(), pred.copy(), ruta=ruta, aristas_ruta=aristas,
                costo=dist[destino], final=True, **kwargs)
