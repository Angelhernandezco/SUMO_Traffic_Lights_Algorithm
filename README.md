# SUMO Traffic Lights Algorithm

Experimentos de progresión semafórica en el corredor **J0→J2→J10→J16** con SUMO, TraCI y un único PPO en J0.

El estado técnico y la evidencia están en [Contexto de sincronización del corredor](docs/CORRIDOR_SYNC_CONTEXT.md). `analysis/` conserva la [comparación actual](analysis/corridor_full/REPORT.md), las métricas y referencias de validación, y un [historial breve del corredor](analysis/CORRIDOR_HISTORY.md).

V1/V2/V3 coordinan J2, J10 y J16 en dirección sur→norte. Desde la terminal de PyCharm: `.\.venv\Scripts\python.exe run_sync.py --mode v3 --seed 42 --gui`. Cambia `v3` por `v1` o `v2`. Las ejecuciones guardan un resumen en `results/`; `--traces` exporta también los CSV detallados.
