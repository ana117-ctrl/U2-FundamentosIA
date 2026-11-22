# """Dijkstra con cola de prioridad; requiere pesos no negativos."""
# import heapq
# from math import inf
# from .comun import Paso, numero, resultado


# def dijkstra(grafo, origen=23, destino=70):
#     grafo.validar_extremos(origen, destino)
#     if any(a.peso < 0 for a in grafo.aristas):
#         raise ValueError("Dijkstra requiere pesos no negativos.")
#     dist = {n: inf for n in grafo.nodos}
#     dist[origen] = 0
#     pred, cerrados, cola = {}, set(), [(0, origen)]
#     comparaciones = relajaciones = 0

#     def paso(titulo, detalle, **extra):
#         return Paso(titulo, detalle, dist.copy(), pred.copy(),
#                     cerrados=tuple(sorted(cerrados)),
#                     frontera=tuple(n for n in grafo.nodos if dist[n] < inf and n not in cerrados),
#                     comparaciones=comparaciones, relajaciones=relajaciones, **extra)

#     yield paso("Inicialización", f"d[{origen}] = 0; las demás distancias son ∞.\n"
#                "Se extrae siempre el nodo pendiente con menor distancia acumulada.")
#     while cola:
#         costo, u = heapq.heappop(cola)
#         if costo != dist[u] or u in cerrados:
#             continue
#         cerrados.add(u)
#         yield paso(f"Fijar nodo {u}", f"d[{u}] = {numero(costo)} es mínima y definitiva.", actual=u)
#         if u == destino:
#             break
#         for v, peso, eid in grafo.adyacencia[u]:
#             if v in cerrados:
#                 continue
#             comparaciones += 1
#             antes, candidato = dist[v], dist[u] + peso
#             mejora = candidato < antes
#             if mejora:
#                 dist[v] = candidato
#                 pred[v] = (u, eid)
#                 heapq.heappush(cola, (candidato, v))
#                 relajaciones += 1
#             yield paso(f"Examinar {u} → {v}",
#                        f"d[{u}] + w({u},{v}) = {numero(dist[u])} + ({numero(peso)}) = {numero(candidato)}.\n"
#                        f"Comparar con d[{v}] = {numero(antes)}.\n" +
#                        (f"Mejora: d[{v}] ← {numero(candidato)}; predecesor[{v}] ← {u}." if mejora
#                         else "No mejora; se conserva la distancia."),
#                        actual=u, arista=eid)
#     yield resultado(dist, pred, origen, destino, cerrados=tuple(cerrados),
#                     comparaciones=comparaciones, relajaciones=relajaciones)

"""Dijkstra paso a paso: los mismos 63 pasos que las diapositivas.

Estructura de los pasos (con los 30 nodos alcanzables y origen 23):
  1        grafo original
  2        título «Algoritmo Dijkstra»
  3 y 4    visita del origen: cálculos de sus vecinos y etiquetas resultantes
  5 a 62   por cada nodo visitado, dos pasos: «Para el nodo X: ...» y etiquetas actualizadas
  63       camino más corto reconstruido con nodoAntecesor

Etiqueta de cada nodo: [AcumuladoPeso, nodoAntecesor].
Regla de desempate: si dos nodos tienen el mismo acumulado, se visita primero el de menor número.
Se visitan todos los nodos del grafo, aunque el destino ya tenga su valor definitivo.
"""
from math import inf
from .comun import Paso, numero, reconstruir, resultado

REGLAS = (
    "1. El origen empieza con AcumuladoPeso = 0 y nodoAntecesor = null; los demás no tienen valor.\n"
    "2. Se elige el nodo no visitado con el menor AcumuladoPeso y se marca como visitado (verde).\n"
    "3. Para cada vecino no visitado: AcumuladoPeso = acumulado del nodo visitado + peso del arco.\n"
    "4. Si el resultado es menor que el valor que ya tenía el vecino, se actualiza y su nodoAntecesor "
    "pasa a ser el nodo visitado; si no, se conserva.\n"
    "5. Se repite hasta visitar todos los nodos. El camino se reconstruye con nodoAntecesor desde el destino."
)


def dijkstra(grafo, origen=23, destino=70):
    grafo.validar_extremos(origen, destino)
    if any(peso < 0 for _, _, peso, _ in grafo.arcos()):
        raise ValueError("Dijkstra requiere pesos no negativos.")
    nodos = grafo.nodos
    dist = {n: inf for n in nodos}
    pred = {}
    cerrados = []
    comparaciones = relajaciones = numero_paso = 0

    def paso(titulo, detalle, **extra):
        nonlocal numero_paso
        numero_paso += 1
        return Paso(f"Paso {numero_paso} · {titulo}", detalle, dist.copy(), pred.copy(),
                    cerrados=tuple(cerrados),
                    frontera=tuple(n for n in nodos if dist[n] < inf and n not in cerrados),
                    comparaciones=comparaciones, relajaciones=relajaciones, **extra)

    def etiqueta(n):
        return f"[{numero(dist[n])}, {pred[n][0] if n in pred else 'null'}]"

    yield paso("Grafo original", f"Grafo no dirigido con pesos positivos.\nOrigen: {origen}; destino: {destino}.")
    yield paso("Algoritmo Dijkstra", REGLAS)
    dist[origen] = 0

    while True:
        pendientes = [n for n in nodos if n not in cerrados and dist[n] < inf]
        if not pendientes:
            break
        u = min(pendientes, key=lambda n: (dist[n], n))   # menor acumulado; empate: menor número
        cerrados.append(u)

        # Vecinos no visitados. La lista viene ordenada por (vecino, peso): con aristas paralelas
        # la primera es la de menor peso.
        vecinos = {}
        for v, peso, eid in grafo.adyacencia[u]:
            if v in cerrados:
                continue
            if v not in vecinos:
                vecinos[v] = [peso, eid, 1]
            else:
                vecinos[v][2] += 1

        bloques, cambios = [], {}
        for v, (peso, eid, cuantas) in vecinos.items():
            comparaciones += 1
            nuevo = dist[u] + peso
            lineas = [f"Para el nodo {v}:"]
            if nuevo < dist[v]:
                antes = f" (antes {numero(dist[v])})" if dist[v] < inf else ""
                lineas += [f"AcumuladoPeso = {numero(dist[u])} + {numero(peso)} = {numero(nuevo)}{antes}",
                           f"nodoAntecesor = {u}"]
                cambios[v] = (nuevo, u, eid)
            else:
                lineas += [f"AcumuladoPeso = {numero(dist[u])} + {numero(peso)} = {numero(nuevo)}",
                           f"Se conserva {etiqueta(v)} porque {numero(dist[v])} ≤ {numero(nuevo)}"]
            if cuantas > 1:
                lineas.append(f"(hay {cuantas} aristas: se usa la de menor peso)")
            bloques.append("\n".join(lineas))

        cabecera = (f"AcumuladoPeso = {numero(dist[u])}\n"
                    f"nodoAntecesor = {pred[u][0] if u in pred else 'null'}")
        cuerpo = ("\n\n".join(bloques) if bloques else
                  f"Todos los vecinos del nodo {u} ya están en verde: no hay nada que calcular.")
        aristas_vecinos = tuple(eid for _, eid, _ in vecinos.values())
        yield paso(f"Visitar el nodo {u}", cabecera + "\n\n" + cuerpo,
                   actual=u, aristas_ruta=aristas_vecinos)

        for v, (nuevo, previo, eid) in cambios.items():
            dist[v] = nuevo
            pred[v] = (previo, eid)
            relajaciones += 1
        if cambios:
            resumen = "Etiquetas actualizadas:\n" + "\n".join(f"Nodo {v}: {etiqueta(v)}" for v in cambios)
        else:
            resumen = "Ninguna etiqueta cambia."
        yield paso(f"Etiquetas tras visitar el nodo {u}", cabecera + "\n\n" + resumen,
                   actual=u, aristas_ruta=aristas_vecinos, cambios=tuple(cambios))

    ruta, aristas = reconstruir(pred, origen, destino)
    if not ruta or dist[destino] == inf:
        yield resultado(dist, pred, origen, destino, comparaciones=comparaciones, relajaciones=relajaciones)
        return
    terminos = " + ".join(numero(grafo.por_id[e].peso) for e in aristas) or "0"
    yield paso("Camino más corto",
               f"Camino más corto: {' → '.join(map(str, ruta))}\n"
               f"Siguiendo nodoAntecesor desde el {destino}: {' ← '.join(map(str, reversed(ruta)))}\n"
               f"AcumuladoPeso = {terminos} = {numero(dist[destino])}",
               ruta=ruta, aristas_ruta=aristas, costo=dist[destino], final=True)