from .dijkstra import dijkstra
from .bellman_ford import bellman_ford
from .floyd_warshall import floyd_warshall
from .a_estrella import a_estrella

ALGORITMOS = {
    "Dijkstra": dijkstra,
    "Bellman–Ford": bellman_ford,
    "Warshall–Floyd": floyd_warshall,
    "A*": a_estrella,
}
