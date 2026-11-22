"""Floyd–Warshall (Warshall–Floyd): distancias mínimas entre todos los pares."""
from math import inf
from .comun import Paso, numero


def floyd_warshall(grafo, origen=23, destino=70):
    grafo.validar_extremos(origen, destino)
    nodos = grafo.nodos
    idx = {n: i for i, n in enumerate(nodos)}
    n = len(nodos)
    d = [[inf] * n for _ in nodos]
    siguiente = [[None] * n for _ in nodos]
    primera_arista = [[None] * n for _ in nodos]
    for i in range(n):
        d[i][i] = 0
    for u, v, peso, eid in grafo.arcos():
        i, j = idx[u], idx[v]
        if peso < d[i][j]:
            d[i][j] = peso
            siguiente[i][j] = j
            primera_arista[i][j] = eid
    oi, di = idx[origen], idx[destino]
    comparaciones = relajaciones = 0

    def paso(titulo, detalle, **extra):
        return Paso(titulo, detalle, {v: d[oi][j] for j, v in enumerate(nodos)},
                    matriz=tuple(tuple(fila) for fila in d),
                    comparaciones=comparaciones, relajaciones=relajaciones, **extra)

    yield paso("Matriz inicial D⁰", "Diagonal = 0; aristas directas = peso; ausencia de conexión = ∞.\n"
               "Si hay aristas paralelas, se usa el menor peso.\n"
               "Cada paso siguiente procesa todos los pares (i,j) con un intermedio k.")
    for k, nodo_k in enumerate(nodos):
        cambios = []
        # Copia de la etapa previa: D^k usa únicamente D^(k−1).
        anterior = [fila[:] for fila in d]
        sig_anterior = [fila[:] for fila in siguiente]
        arista_anterior = [fila[:] for fila in primera_arista]
        for i in range(n):
            for j in range(n):
                comparaciones += 1
                candidato = anterior[i][k] + anterior[k][j]
                if candidato < anterior[i][j]:
                    d[i][j] = candidato
                    siguiente[i][j] = sig_anterior[i][k]
                    primera_arista[i][j] = arista_anterior[i][k]
                    cambios.append((nodos[i], nodos[j], anterior[i][j], candidato,
                                    anterior[i][k], anterior[k][j]))
                    relajaciones += 1
        lineas = [f"D[{u},{v}]: min({numero(antes)}, {numero(ik)} + ({numero(kj)})) = {numero(nuevo)}"
                  for u, v, antes, nuevo, ik, kj in cambios]
        yield paso(f"Etapa {k+1}/{n} · intermedio k = {nodo_k}",
                   f"D(i,j) ← min(D(i,j), D(i,k) + D(k,j)).\n"
                   f"{n*n} pares comprobados; {len(cambios)} distancias mejoradas.\n"
                   f"D[{origen},{destino}] = {numero(d[oi][di])}.\n\n" +
                   ("\n".join(lineas) if lineas else "No hubo mejoras en esta etapa."),
                   actual=nodo_k, intermedio=nodo_k, cambios=tuple(cambios), pasada=k+1)

    ciclos = [k for k in range(n) if d[k][k] < 0]
    afectados = {(i, j) for k in ciclos for i in range(n) for j in range(n)
                 if d[i][k] < inf and d[k][j] < inf}
    for i, j in afectados:
        d[i][j] = -inf
    if (oi, di) in afectados:
        yield paso("No existe un costo mínimo finito", "Un ciclo negativo alcanzable afecta a la ruta solicitada.",
                   final=True, error=True, costo=-inf)
        return
    if d[oi][di] == inf:
        yield paso("Destino inalcanzable", f"No existe una ruta de {origen} a {destino}.", final=True)
        return
    ruta, aristas, actual = [origen], [], oi
    while actual != di:
        prox = siguiente[actual][di]
        eid = primera_arista[actual][di]
        if prox is None or eid is None or len(aristas) >= n:
            raise RuntimeError("No se pudo reconstruir la ruta de la matriz.")
        aristas.append(eid)
        ruta.append(nodos[prox])
        actual = prox
    yield paso("Resultado final", "Ruta: " + " → ".join(map(str, ruta)) +
               f"\nCosto mínimo = {numero(d[oi][di])}.\n"
               f"Se completaron {n} etapas y {n**3} comparaciones.",
               ruta=tuple(ruta), aristas_ruta=tuple(aristas), costo=d[oi][di], final=True)
