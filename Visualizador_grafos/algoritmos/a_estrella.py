"""A* con heurística geométrica consistente calculada a partir de las aristas."""
import heapq
from math import dist as euclidea, inf
from .comun import Paso, numero, resultado


def calcular_heuristica(grafo, destino):
    # Para todo arco (u,v), α·distancia_geométrica(u,v) <= w(u,v).
    # Por desigualdad triangular h(u) <= w(u,v)+h(v): es consistente.
    razones = [a.peso / euclidea(grafo.posiciones[a.origen], grafo.posiciones[a.destino])
               for a in grafo.aristas
               if euclidea(grafo.posiciones[a.origen], grafo.posiciones[a.destino]) > 0]
    escala = min(razones, default=0)
    h = {n: escala * euclidea(xy, grafo.posiciones[destino]) for n, xy in grafo.posiciones.items()}
    return h, escala


def a_estrella(grafo, origen=23, destino=70):
    grafo.validar_extremos(origen, destino)
    if any(a.peso < 0 for a in grafo.aristas):
        raise ValueError("Esta implementación de A* requiere pesos no negativos.")
    h, escala = calcular_heuristica(grafo, destino)
    dist = {n: inf for n in grafo.nodos}
    dist[origen] = 0
    pred, cerrados = {}, set()
    cola = [(h[origen], 0, origen)]
    comparaciones = relajaciones = 0

    def paso(titulo, detalle, **extra):
        return Paso(titulo, detalle, dist.copy(), pred.copy(),
                    cerrados=tuple(sorted(cerrados)),
                    frontera=tuple(n for n in grafo.nodos if dist[n] < inf and n not in cerrados),
                    heuristica=h.copy(), comparaciones=comparaciones, relajaciones=relajaciones, **extra)

    yield paso("Inicialización de A*", "f(n) = g(n) + h(n).\n"
               f"h(n) = α · distancia geométrica(n,{destino}); α = {escala:.7f}.\n"
               "α es el mínimo peso/longitud entre aristas. Así, h nunca sobreestima el costo real.")
    while cola:
        f, g, u = heapq.heappop(cola)
        if g != dist[u] or u in cerrados:
            continue
        cerrados.add(u)
        yield paso(f"Expandir nodo {u}", f"Menor f pendiente: {numero(g)} + {numero(h[u])} = {numero(f)}.", actual=u)
        if u == destino:
            break
        for v, peso, eid in grafo.adyacencia[u]:
            comparaciones += 1
            antes, candidato = dist[v], dist[u] + peso
            mejora = candidato < antes
            if mejora:
                dist[v] = candidato
                pred[v] = (u, eid)
                cerrados.discard(v)
                heapq.heappush(cola, (candidato + h[v], candidato, v))
                relajaciones += 1
            yield paso(f"Evaluar {u} → {v}",
                       f"g candidato = {numero(dist[u])} + ({numero(peso)}) = {numero(candidato)}.\n"
                       f"g anterior de {v} = {numero(antes)}; h({v}) = {numero(h[v])}.\n" +
                       (f"Se actualiza: f({v}) = {numero(dist[v])} + {numero(h[v])} = {numero(dist[v]+h[v])}." if mejora
                        else "No mejora; se conserva g."), actual=u, arista=eid)
    yield resultado(dist, pred, origen, destino, cerrados=tuple(cerrados), heuristica=h,
                    comparaciones=comparaciones, relajaciones=relajaciones)
