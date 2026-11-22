# Grafos: ruta del nodo 23 al 70

Aplicación de escritorio en Python, preparada para Zorin OS. Dibuja los dos grafos proporcionados y permite seguir Dijkstra, Bellman–Ford, Warshall–Floyd y A* paso a paso. Funciona sin internet después de instalar Python y Tkinter.

## Inicio rápido en Zorin OS

1. Extrae `Grafos_23_70.zip`.
2. Abre la carpeta `Grafos_23_70`, haz clic derecho en un espacio vacío y elige **Abrir en una terminal**.
3. Instala los componentes del sistema una sola vez:

```bash
sudo apt update
sudo apt install python3 python3-tk
```

4. Inicia la aplicación:

```bash
bash ejecutar.sh
```

También puedes usar `/usr/bin/python3 main.py`. No necesitas `pip`, un entorno virtual, NetworkX, Matplotlib ni una tarjeta gráfica dedicada. Se requiere Python 3.10 o posterior; el lanzador usa el Python del sistema para encontrar Tkinter instalado mediante `apt`.

## Cómo usarla

- Elige el algoritmo en la lista. El programa cambia automáticamente al grafo correspondiente.
- El origen ya es **23** y el destino **70**. Puedes cambiarlos por cualquier nodo existente.
- **Reproducir / Pausar:** anima el procedimiento. El control de segundos indica la pausa entre pasos: menos segundos, más rápido.
- **Paso / Anterior:** recorre el procedimiento y revisa los cálculos sin perder el estado.
- **Reiniciar:** vuelve a la inicialización.
- **Ver resultado:** salta al final y traza la ruta por segmentos verdes.
- La barra inferior permite saltar a cualquier paso.
- Usa la rueda para ampliar el grafo, arrastra para moverlo y pulsa **Ajustar** para volver a encuadrarlo.
- Haz clic en un nodo o en su fila para consultar su distancia. En A* también aparecen `h` y `f` en la barra inferior.
- **Matriz**, disponible en Warshall–Floyd, muestra todas las distancias de la etapa actual y permite exportarlas a CSV. Se actualiza al avanzar o retroceder en la ventana principal.
- **Guardar procedimiento** exporta a TXT todas las operaciones de la ejecución, incluso si todavía no has reproducido la animación completa.
- **Aristas** muestra la transcripción; **Imagen original** abre el grafo de referencia con barras de desplazamiento.
- Atajos en la ventana principal: espacio para reproducir/pausar y flechas izquierda/derecha para cambiar de paso.

### Colores

| Color | Significado |
|---|---|
| Contorno morado | Nodo de origen |
| Contorno coral | Nodo de destino |
| Amarillo | Nodo o arista examinada; intermedio `k` en Warshall–Floyd |
| Azul | Nodo alcanzado pendiente de procesar |
| Gris azulado | Nodo cerrado en Dijkstra o A* |
| Verde | Ruta mínima final |
| Etiqueta coral | Peso negativo |

## Resultados de 23 a 70

| Algoritmo | Grafo usado | Ruta | Costo |
|---|---|---|---:|
| Dijkstra | Positivo, no dirigido | 23 → 2 → 17 → 70 | 14 |
| Bellman–Ford | Dirigido, con pesos negativos | 23 → 16 → 17 → 70 | 2 |
| Warshall–Floyd | Positivo, no dirigido | 23 → 2 → 17 → 70 | 14 |
| A* | Positivo, no dirigido | 23 → 2 → 17 → 70 | 14 |

Grafo positivo: **3 + 5 + 6 = 14**. Grafo Bellman–Ford: **−11 + 7 + 6 = 2**. Son grafos diferentes: sus costos no se comparan como si un algoritmo fuera mejor que otro.

## Transcripción de las imágenes

Cada grafo tiene **30 nodos y 63 aristas**. Los identificadores no son consecutivos: llegar al nodo 70 no implica que existan 70 nodos.

- La primera imagen no tiene flechas: se interpreta como un grafo **no dirigido**, con cada conexión utilizable en ambos sentidos. Sus colores previos no condicionan la búsqueda.
- Se preservan las dos aristas paralelas entre **12 y 16**, de pesos **16 y 3**. Los algoritmos conservan el identificador de la arista elegida; la ruta final resalta la conexión correcta.
- En la primera imagen, `6–15` tiene peso **3** y `7–1` peso **20**. Son conexiones diferentes que se cruzan en el dibujo.
- En Bellman–Ford se respetan todas las flechas, incluyendo **12 → 16, peso 16**, y **12 → 16, peso 3**.
- Bellman–Ford incluye siete pesos negativos: `11→21: −2`, `11→24: −7`, `6→9: −7`, `9→10: −3`, `3→69: −6`, `23→16: −11` y `5→20: −2`.
- El nodo inferior se llama **66** en el grafo positivo y **666** en el de Bellman–Ford, como aparece en cada imagen. No pertenece a la ruta final solicitada.
- Un cruce de líneas sin un círculo no crea un nodo. La disposición visual se mantiene aproximadamente y se usa un fondo limpio para facilitar la lectura.
- Los archivos JSON son la fuente de datos. Puedes corregir allí un peso o una conexión y volver a abrir el programa. Las imágenes originales se incluyen para comprobarlos.

## Qué muestra cada procedimiento

### Dijkstra

Inicializa `d[origen]=0` y el resto en infinito. Usa una cola de prioridad para extraer el nodo de menor distancia pendiente. Examina sus vecinos y actualiza `d[v]` cuando `d[u]+peso < d[v]`; guarda su predecesor. Termina cuando fija el destino.

El panel muestra cada extracción, suma, comparación y actualización. Las distancias de nodos que permanecen pendientes cuando se alcanza el destino **todavía pueden ser provisionales**. No admite pesos negativos.

### Bellman–Ford

Inicializa las distancias y recorre todos los arcos en el orden del JSON durante un máximo de `|V|−1` pasadas. Actualiza en el momento de encontrar una mejora. Una pasada sin cambios permite terminar anticipadamente.

Después comprueba todos los arcos una vez más para detectar ciclos negativos alcanzables. Si un ciclo puede llegar al destino, informa que no existe un mínimo finito. Si afecta únicamente otra zona, marca esos nodos con `−∞` y conserva el resultado válido del destino. Los ciclos negativos inalcanzables desde el origen no alteran la consulta.

**El grafo proporcionado no tiene ciclos negativos**, aunque sí tiene aristas negativas. Desde 23 termina después de cuatro pasadas, incluida la última sin cambios; luego realiza la comprobación adicional.

### Warshall–Floyd / Floyd–Warshall

Se implementa el algoritmo de **caminos mínimos con pesos**, con matrices de distancias y de siguiente salto. No es únicamente el cierre transitivo booleano de Warshall.

La matriz inicial tiene ceros en la diagonal, pesos de conexiones directas y `∞` donde no hay conexión. Para aristas paralelas, usa el menor peso. Procesa los nodos intermedios en orden numérico:

```text
D_nueva[i,j] = min(D_anterior[i,j], D_anterior[i,k] + D_anterior[k,j])
```

Cada paso visual agrupa las **900 comparaciones** de un intermedio `k`; el panel enumera todas las mejoras de esa etapa. Hay **30 etapas y 27 000 comparaciones**, además de la inicialización y el resultado. Las mejoras que no caben en el panel se pueden leer con su barra de desplazamiento o exportando el TXT. La ventana Matriz tiene desplazamiento horizontal y vertical.

### A*

Usa `f(n)=g(n)+h(n)`, donde `g` es el costo acumulado y `h` estima lo que falta. La heurística se calcula con las coordenadas del dibujo:

```text
alpha = min(peso(u,v) / distancia_geometrica(u,v))
h(n) = alpha * distancia_geometrica(n,destino)
```

El mínimo se toma sobre aristas cuyos extremos tienen posiciones distintas. Si no existen, se utiliza `alpha=0`. Al escalar de esta forma, `h` es consistente: `h(u) <= peso(u,v)+h(v)`, por la desigualdad triangular. Por eso no sobreestima el costo real y preserva la optimalidad. Usar directamente píxeles como costo no tendría esa garantía. Esta implementación requiere pesos no negativos.

La tabla muestra `g`; al seleccionar un nodo aparecen `g`, `h` y `f` en la barra inferior. Los valores se redondean solo para mostrarlos; los cálculos usan su precisión completa.

## Archivos principales

| Archivo | Función |
|---|---|
| `main.py` | Inicio de la aplicación y modos de consola/pruebas |
| `ejecutar.sh` | Lanzador para Zorin OS |
| `interfaz.py` | Dibujo, animación, controles, tablas y exportación |
| `grafo.py` | Modelo de nodos y aristas, lectura y validación de datos |
| `algoritmos/dijkstra.py` | Implementación de Dijkstra |
| `algoritmos/bellman_ford.py` | Implementación de Bellman–Ford |
| `algoritmos/floyd_warshall.py` | Implementación de Warshall–Floyd |
| `algoritmos/a_estrella.py` | Implementación de A* y su heurística |
| `algoritmos/comun.py` | Instantáneas de pasos y reconstrucción de rutas |
| `datos/positivo.json` | Grafo positivo transcrito |
| `datos/bellman_ford.json` | Grafo dirigido transcrito |
| `referencias/` | Las dos imágenes originales |
| `pruebas/test_algoritmos.py` | Pruebas automatizadas |

Los algoritmos están programados explícitamente, sin llamar a funciones de caminos mínimos de una biblioteca. Generan instantáneas independientes que la interfaz reproduce; retroceder no modifica el cálculo.

## Comprobar resultados sin abrir una ventana

```bash
python3 main.py --consola
python3 main.py --verificar
```

Las pruebas revisan las cuatro rutas, su suma de pesos, el sentido de cada arista, aristas paralelas, nodos sin ruta, origen igual al destino, pesos cero, heurística admisible, la matriz completa de Floyd y ciclos negativos relevantes e irrelevantes. Comparan los cálculos con una implementación independiente de programación dinámica.

## Si no abre

- **`No module named tkinter`:** ejecuta `sudo apt install python3-tk` y vuelve a usar `bash ejecutar.sh`.
- **Estás dentro de Conda, uv o un entorno virtual:** usa `bash ejecutar.sh` o `/usr/bin/python3 main.py` para tomar el Python del sistema.
- **`can't open file main.py`:** abre la terminal dentro de la carpeta extraída, donde están `main.py` y `ejecutar.sh`.
- **`no display name` / `$DISPLAY`:** necesitas ejecutar la interfaz desde el escritorio de Zorin, no desde una sesión de servidor sin pantalla. El modo `--consola` funciona sin interfaz gráfica.
- **Todo se ve pequeño:** amplía con `+` o la rueda; puedes mover el grafo arrastrando. La ventana se puede maximizar.
- **Bellman–Ford parece tardado:** muestra cada arco de cada pasada, incluyendo los que no mejoran. Reduce la pausa entre pasos o usa **Ver resultado**.

Referencias de instalación: [Tkinter, documentación oficial de Python](https://docs.python.org/3/library/tkinter.html) y [paquete python3-tk en Ubuntu](https://packages.ubuntu.com/noble/python3-tk).
