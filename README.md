# SUMO Traffic Lights Algorithm

Experimentos de progresión semafórica en el corredor **J0→J2→J10→J16** con SUMO, TraCI y un único PPO en J0.

El estado técnico, la evidencia y la siguiente hipótesis están en [Contexto de sincronización del corredor](docs/CORRIDOR_SYNC_CONTEXT.md). Los resultados detallados de cada corrida se conservan en `analysis/`; [el historial anterior](docs/EXPERIMENT_HISTORY.md) es sólo archivo experimental.

V1/V2/V3 coordinan actualmente J2, J10 y J16 en dirección sur→norte. Para visualizar una corrida desde la terminal de PyCharm: `.\.venv\Scripts\python.exe analysis\run_corridor_sync.py --mode v3 --seed 42 --gui`. Cambia `v3` por `v1` o `v2`; para repetir el antiguo experimento sólo J0→J2, añade `--scope j2` a V1/V3. [Comparación del corredor completo](analysis/corridor_full/REPORT.md).
