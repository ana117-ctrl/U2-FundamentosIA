"""Pruebas de rutas, orientación, aristas paralelas y ciclos negativos."""
import math
import unittest

from grafo import Grafo, Arista, cargar_grafo
from algoritmos import ALGORITMOS, dijkstra, bellman_ford, floyd_warshall, a_estrella
from algoritmos.a_estrella import calcular_heuristica


def pequeno(aristas, dirigido=True, nodos=None):
    ns = sorted(nodos or {n for u, v, _ in aristas for n in (u, v)})
    return Grafo("Prueba", dirigido, {n: (i*40, (i % 3)*50) for i, n in enumerate(ns)},
                 [Arista(f"t{i}", u, v, w) for i, (u, v, w) in enumerate(aristas)])


def referencia(grafo, origen):
    """Programación dinámica independiente: caminos de hasta k aristas."""
    anterior = {n: math.inf for n in grafo.nodos}
    anterior[origen] = 0
    for _ in range(len(grafo.nodos)-1):
        nuevo = anterior.copy()
        for u, v, peso, _ in grafo.arcos():
            nuevo[v] = min(nuevo[v], anterior[u]+peso)
        anterior = nuevo
    return anterior


class PruebasAlgoritmos(unittest.TestCase):
    def comprobar_ruta(self, g, fin, origen, destino, costo):
        self.assertEqual(fin.costo, costo)
        if costo == math.inf:
            self.assertEqual(fin.ruta, ())
            return
        self.assertEqual(fin.ruta[0], origen)
        self.assertEqual(fin.ruta[-1], destino)
        self.assertEqual(len(fin.aristas_ruta), len(fin.ruta)-1)
        self.assertEqual(len(set(fin.ruta)), len(fin.ruta))
        total = 0
        for u, v, eid in zip(fin.ruta, fin.ruta[1:], fin.aristas_ruta):
            a = g.por_id[eid]
            self.assertTrue((a.origen, a.destino) == (u, v) or
                            (not g.dirigido and (a.origen, a.destino) == (v, u)))
            total += a.peso
        self.assertEqual(total, costo)

    def test_cuatro_rutas_solicitadas(self):
        for nombre, algoritmo in ALGORITMOS.items():
            with self.subTest(algoritmo=nombre):
                es_bellman = nombre == "Bellman–Ford"
                g = cargar_grafo("bellman_ford" if es_bellman else "positivo")
                fin = list(algoritmo(g, 23, 70))[-1]
                self.assertEqual(fin.ruta, (23, 16, 17, 70) if es_bellman else (23, 2, 17, 70))
                self.comprobar_ruta(g, fin, 23, 70, 2 if es_bellman else 14)

    def test_transcripcion(self):
        positivo, bf = cargar_grafo(), cargar_grafo("bellman_ford")
        for g in (positivo, bf):
            self.assertEqual(len(g.nodos), 30)
            self.assertEqual(len(g.aristas), 63)
        self.assertIn(66, positivo.nodos)
        self.assertNotIn(666, positivo.nodos)
        self.assertIn(666, bf.nodos)
        self.assertNotIn(66, bf.nodos)
        self.assertFalse(positivo.dirigido)
        self.assertTrue(bf.dirigido)
        self.assertTrue(all(a.peso > 0 for a in positivo.aristas))
        self.assertEqual(sum(a.peso < 0 for a in bf.aristas), 7)
        self.assertEqual(sorted(a.peso for a in positivo.aristas if {a.origen,a.destino} == {12,16}), [3,16])
        self.assertIn((12,16,16), [(a.origen,a.destino,a.peso) for a in bf.aristas])
        self.assertIn((12,16,3), [(a.origen,a.destino,a.peso) for a in bf.aristas])

    def test_oraculo_independiente_en_grafos_reales(self):
        for tipo in ("positivo", "bellman_ford"):
            g = cargar_grafo(tipo)
            funciones = (bellman_ford, floyd_warshall) if g.dirigido else tuple(ALGORITMOS.values())
            # Incluye zona izquierda, derecha y los extremos de las aristas paralelas.
            for origen, destino in ((23,70),(29,70),(70,29),(12,16),(16,12),(0,27)):
                costo = referencia(g, origen)[destino]
                for algoritmo in funciones:
                    with self.subTest(tipo=tipo, par=(origen,destino), algoritmo=algoritmo.__name__):
                        fin = list(algoritmo(g,origen,destino))[-1]
                        self.comprobar_ruta(g,fin,origen,destino,costo)

    def test_heuristica_consistente_y_admisible(self):
        g = cargar_grafo()
        h, escala = calcular_heuristica(g,70)
        real = referencia(g,70)
        self.assertGreater(escala,0)
        self.assertEqual(h[70],0)
        for n in g.nodos:
            self.assertLessEqual(h[n],real[n]+1e-9)
        for u,v,w,_ in g.arcos():
            self.assertLessEqual(h[u],w+h[v]+1e-9)

    def test_floyd_matriz_completa_y_etapas(self):
        g = cargar_grafo()
        pasos = list(floyd_warshall(g,23,70))
        self.assertEqual(len(pasos), len(g.nodos)+2)
        self.assertEqual(pasos[-1].comparaciones,len(g.nodos)**3)
        for i,n in enumerate(g.nodos):
            esperado=referencia(g,n)
            self.assertEqual(list(pasos[-1].matriz[i]),[esperado[v] for v in g.nodos])
        for previo,actual in zip(pasos,pasos[1:]):
            for f1,f2 in zip(previo.matriz,actual.matriz):
                self.assertTrue(all(b<=a for a,b in zip(f1,f2)))

    def test_aristas_paralelas_y_peso_cero(self):
        g=pequeno([(1,2,9),(1,2,0),(2,3,2),(1,3,10)],False)
        for algoritmo in ALGORITMOS.values():
            fin=list(algoritmo(g,1,3))[-1]
            self.comprobar_ruta(g,fin,1,3,2)
            self.assertEqual(fin.aristas_ruta[0],"t1")

    def test_sin_ruta_y_origen_igual_destino(self):
        g=pequeno([(1,2,4)],nodos=[1,2,3])
        for algoritmo in ALGORITMOS.values():
            for o,d,c in ((2,1,math.inf),(1,3,math.inf),(1,1,0)):
                self.comprobar_ruta(g,list(algoritmo(g,o,d))[-1],o,d,c)

    def test_negativos_sin_ciclo(self):
        g=pequeno([(1,2,2),(1,3,5),(3,2,-7),(2,4,1)])
        for algoritmo in (bellman_ford,floyd_warshall):
            fin=list(algoritmo(g,1,4))[-1]
            self.comprobar_ruta(g,fin,1,4,-1)
            self.assertEqual(fin.ruta,(1,3,2,4))
        for algoritmo in (dijkstra,a_estrella):
            with self.assertRaises(ValueError):
                list(algoritmo(g,1,4))

    def test_ciclo_negativo_que_afecta_destino(self):
        g=pequeno([(1,2,1),(2,3,-3),(3,2,1),(3,4,2)])
        for algoritmo in (bellman_ford,floyd_warshall):
            fin=list(algoritmo(g,1,4))[-1]
            self.assertTrue(fin.error)
            self.assertEqual(fin.costo,-math.inf)
            self.assertEqual(fin.ruta,())

    def test_ciclo_negativo_no_afecta_destino(self):
        g=pequeno([(1,2,1),(2,3,-3),(3,2,1),(1,4,8)])
        for algoritmo in (bellman_ford,floyd_warshall):
            fin=list(algoritmo(g,1,4))[-1]
            self.comprobar_ruta(g,fin,1,4,8)
            self.assertEqual(fin.distancias[2],-math.inf)

    def test_ciclo_negativo_inalcanzable(self):
        g=pequeno([(1,4,8),(2,3,-3),(3,2,1)])
        for algoritmo in (bellman_ford,floyd_warshall):
            fin=list(algoritmo(g,1,4))[-1]
            self.comprobar_ruta(g,fin,1,4,8)
            self.assertEqual(fin.distancias[2],math.inf)

    def test_instantaneas_no_se_modifican(self):
        g=cargar_grafo()
        for algoritmo in ALGORITMOS.values():
            generador=algoritmo(g,23,70)
            primero=next(generador)
            lista=list(generador)
            self.assertEqual(primero.distancias[70],math.inf)
            self.assertEqual(lista[-1].distancias[70],14)

    def test_nodo_inexistente(self):
        g=cargar_grafo()
        for algoritmo in ALGORITMOS.values():
            with self.assertRaises(ValueError):
                list(algoritmo(g,999,70))


if __name__ == "__main__":
    unittest.main()
