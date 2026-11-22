#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Usa el Python del sistema en Zorin para encontrar python3-tk instalado con apt.
grafos_python=/usr/bin/python3
if [[ ! -x "$grafos_python" ]]; then
    grafos_python=python3
fi
if ! "$grafos_python" -c 'import tkinter' >/dev/null 2>&1; then
    echo 'Falta Tkinter. Ejecuta una sola vez:'
    echo 'sudo apt update'
    echo 'sudo apt install python3 python3-tk'
    exit 1
fi
exec "$grafos_python" main.py "$@"
