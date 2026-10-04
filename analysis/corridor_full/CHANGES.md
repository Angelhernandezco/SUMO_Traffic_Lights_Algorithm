# Archivos de esta extensión

## Código modificado

- `analysis/j0_j2_dynamic.py`: J2Advance admite TLS/fase receptora parametrizados; los valores por defecto conservan el V1 local.
- `analysis/j0_j2_causal_window.py`: CausalJ2Window admite TLS/fase/tiempo acumulado; sus valores por defecto conservan el comportamiento histórico usado por V3.
- `analysis/j0_j2_preannounce.py`: PreannouncedWindow usa los parámetros del receptor; el ejecutor histórico sigue siendo sólo J2.
- `analysis/run_corridor_sync.py`: V1/V3 completos por defecto; `--scope j2` conserva el replay anterior, GUI y las tres lane metrics.
- `analysis/run_v1_gui.py`: abre V1 completo por defecto; permite alcance histórico explícito.

## Código nuevo

- `analysis/corridor_full.py`: V1/V3 completos, observación pasiva común para V1/V2/V3, comparación por vehículo, validación de fases y ejecución por demanda.
- `analysis/corridor_full/analyze.py`: tablas por demanda, agregados, cobertura física de ventanas y comparación con resultados locales previos.
- `analysis/corridor_full/validate_mechanism.py`: nueve escenarios nativos de SUMO para los tres receptores.
- `analysis/corridor_full/validate_outputs.py`: comprobación de métricas históricas y reproducción exacta con un hilo de CPU.

## Documentación modificada

- `README.md`: alcance completo vigente y comando manual.
- `docs/CORRIDOR_SYNC_CONTEXT.md`: distingue versiones históricas/locales y actuales/completas; incorpora resultados 42–46.

## Evidencia nueva

- `runs/`: quince resultados principales (3 versiones × 5 demandas), JSON y CSV de vehículos, eventos, ventanas, acciones, fases y muestras temporales.
- `local_replay/`: replays V1/V3 de demanda 42 con el comportamiento original de J0→J2.
- `single_thread_replay/` y `single_thread_full_replay/`: validación de reproducción con un hilo de CPU, V1/V3 locales y las tres versiones completas.
- `validation_before/`: copias de los tres controladores antes de parametrizar los receptores.
- `per_demand.csv`, `aggregate.json`, `window_arrival_audit.csv`, `REPORT.md`, `mechanism_checks.json`, `replay_validation.json`.

Validaciones: cero violaciones de fase/duración, cero teleports, cohortes y rutas iguales entre versiones. Los replays locales reproducen las métricas históricas. Con un hilo de CPU, JSON y siete CSV son idénticos a la ejecución original en demanda 42, tanto para V1/V3 locales como para V1/V2/V3 completos. Los hashes de red, checkpoint, policy/ y controlador V2 permanecen iguales.

No se modificaron los resultados anteriores, policy/, checkpoint, la red ni los offsets. No hubo entrenamiento, commit o push. Los cambios previos de la rama en la red, el historial y los documentos eliminados ya existían antes de esta tarea.
