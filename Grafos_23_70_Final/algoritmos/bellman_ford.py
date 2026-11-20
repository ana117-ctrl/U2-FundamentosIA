"""Bellman–Ford paso a paso: mismo procedimiento que las diapositivas (260 pasos).

Estructura de los pasos:
  0.0 y 0.1      inicio e inicialización de d y Π
  p.1 ... p.N    Relax sobre cada arco de la pasada p (pregunta, respuesta, proceso)
  p.fin          cierre de la pasada p
  V.1            verificación de ciclos negativos
  Fin            camino más corto y reconstrucción con Π
"""
from math import inf
from .comun import Paso, numero, reconstruir, resultado


def _peso(w):
    """Los pesos negativos van entre paréntesis: d[u] + (-11)."""
    return f"({numero(w)})" if w < 0 else numero(w)


def bellman_ford(grafo, origen=23, destino=70):
    grafo.validar_extremos(origen, destino)
    nodos = grafo.nodos
    dist = {n: inf for n in nodos}
    pred = {}
    # Mismo orden de arcos que las diapositivas: por nodo de origen y luego por destino.
    arcos = sorted(grafo.arcos(), key=lambda a: (a[0], a[1]))
    comparaciones = relajaciones = 0

    def paso(clave, descripcion, detalle, **extra):
        titulo = f"Paso {clave}" + (f" · {descripcion}" if descripcion else "")
        return Paso(titulo, detalle, dist.copy(), pred.copy(),
                    frontera=tuple(n for n in nodos if dist[n] < inf),
                    comparaciones=comparaciones, relajaciones=relajaciones, **extra)

    yield paso("0.0", "", f"Encontrar el camino más corto del vértice {origen} a cada uno "
                          f"de los otros vértices (en especial al {destino}).")
    dist[origen] = 0
    yield paso("0.1", "", "Inicializar los vectores d y Π.", actual=origen)

    for ronda in range(1, len(nodos)):
        cambios = 0
        for i, (u, v, peso, eid) in enumerate(arcos, 1):
            comparaciones += 1
            du, dv = dist[u], dist[v]
            mejora = du != inf and dv > du + peso
            if mejora:
                dist[v] = du + peso
                pred[v] = (u, eid)
                cambios += 1
                relajaciones += 1
                proceso = (f"d[{v}] = d[{u}] + w( {u} , {v} ) = {numero(du)} + {_peso(peso)} "
                           f"= {numero(dist[v])}  y  Π[{v}] = {u}")
            else:
                proceso = "No se hace nada." + (f"  (d[{u}] = ∞, aún no se alcanza {u})" if du == inf else "")
            detalle = (f"Pregunta:  ¿ d[{v}] > d[{u}] + w( {u} , {v} ) ?\n"
                       f"   con valores: {numero(dv)} > {numero(du)} + {_peso(peso)} ?\n"
                       f"Respuesta:  {'SI' if mejora else 'NO'}\n"
                       f"Proceso:  {proceso}")
            yield paso(f"{ronda}.{i}", f"Aplicar Relax al Arco ({u},{v})", detalle,
                       actual=u, arista=eid, pasada=ronda, cambios=(v,) if mejora else ())
        if cambios:
            texto = f"Fin de la pasada {ronda}: hubo cambios en d, se repite otra pasada sobre todos los arcos."
        else:
            texto = (f"Fin de la pasada {ronda}: ya no hubo cambios, d y Π son estables; "
                     f"no hace falta completar las |V|−1 = {len(nodos) - 1} pasadas.")
        yield paso(f"{ronda}.fin", "", texto, pasada=ronda)
        if cambios == 0:
            break

    # Una pasada más sobre todos los arcos detecta ciclos negativos alcanzables desde el origen.
    comparaciones += len(arcos)
    malos = [(u, v, peso) for u, v, peso, _ in arcos if dist[u] != inf and dist[u] + peso < dist[v]]
    if not malos:
        yield paso("V.1", "", f"Verificación de ciclos negativos: se recorren los {len(arcos)} arcos una vez más "
                              f"y para ninguno se cumple d[v] > d[u] + w(u,v) (0 arcos). "
                              f"No hay ciclos negativos alcanzables desde {origen}.")
    else:
        evidencia = [f"{u} → {v} aún mejoraría: {numero(dist[u] + p)} < {numero(dist[v])}" for u, v, p in malos]
        yield paso("V.1", "", f"Verificación de ciclos negativos: {len(malos)} arcos todavía se pueden mejorar.\n"
                              + "\n".join(evidencia))
        afectados, pendientes = {v for _, v, _ in malos}, [v for _, v, _ in malos]
        while pendientes:
            u = pendientes.pop()
            for v, _, _ in grafo.adyacencia[u]:
                if v not in afectados:
                    afectados.add(v)
                    pendientes.append(v)
        for v in afectados:
            dist[v] = -inf
            pred.pop(v, None)
        if destino in afectados:
            yield paso("Fin", "", "Un ciclo negativo alcanzable permite reducir el costo sin límite y después "
                                  "llegar al destino.\nNo se muestra una ruta mínima porque no existe.",
                       final=True, error=True, costo=-inf)
            return

    ruta, aristas = reconstruir(pred, origen, destino)
    if not ruta or dist[destino] == inf:
        yield resultado(dist, pred, origen, destino, comparaciones=comparaciones, relajaciones=relajaciones)
        return
    terminos = " + ".join(_peso(grafo.por_id[e].peso) for e in aristas) or "0"
    yield paso("Fin", "", f"Camino más corto {origen} → {destino}:  {' → '.join(map(str, ruta))}  "
                          f"con costo d[{destino}] = {terminos} = {numero(dist[destino])}.\n"
                          f"Se reconstruye siguiendo Π desde {destino} hacia atrás: "
                          f"{' ← '.join(map(str, reversed(ruta)))}.",
               ruta=ruta, aristas_ruta=aristas, costo=dist[destino], final=True)