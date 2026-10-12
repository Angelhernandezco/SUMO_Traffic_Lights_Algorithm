# Experimento emparejado de 240 corridas

Protocolo: bajo (semillas 1001–1030, periodo 2 s) y alto (2001–2030,
periodo 0,5 s), cuatro configuraciones por demanda. Son etiquetas relativas,
sin clasificación por capacidad. Semilla SUMO 42, horizonte 3600 s, paso 1 s.
No hay entrenamiento. PPO determinista sólo en J0, checkpoint
`policy/models/model_future_v39_yellow_test36_3.pth`. Amarillos separados de 4 s,
orden cíclico y verdes dinámicos 5–45 s. V3 significa `PreannouncedWindow`
original, con desplazamientos acumulados 12/22/32 s; ninguna variante V3.x.

| mode | Configuración | Offsets J0/J2/J10/J16 | Verdes |
|---|---|---|---|
| plain | Normal | 0/0/0/0 | 15 s en todos |
| plain_offset | Normal sincronizado | 0/0/72/24 | 15 s en todos |
| only_ppo | Sólo PPO | 0/0/0/0 | PPO J0; receptores 15 s |
| v3 | PPO sincronizado | 0/0/72/24 | PPO J0; receptores V3 original |

## Comandos desde la raíz del repositorio

```powershell
.\.venv\Scripts\python.exe -m unittest tests.check_matched_experiment -v
.\.venv\Scripts\python.exe -m tools.run_matched_experiment --stage inspect
.\.venv\Scripts\python.exe -m tools.run_matched_experiment --stage validate
.\.venv\Scripts\python.exe -m tools.run_matched_experiment --stage prepare
.\.venv\Scripts\python.exe -m tools.run_matched_experiment --stage run
.\.venv\Scripts\python.exe -m tools.run_matched_experiment --stage export
.\.venv\Scripts\python.exe -m tools.run_matched_parallel --self-test
.\.venv\Scripts\python.exe -m tools.run_matched_parallel --autotune
.\.venv\Scripts\python.exe -m tools.run_matched_parallel --trial-six
.\.venv\Scripts\python.exe -m tools.finish_matched_experiment --self-test
.\.venv\Scripts\python.exe -m tools.finish_matched_experiment
```

Sin `--stage` sólo se inspecciona. `validate` genera temporalmente dos demandas
de referencia y ejecuta cuatro simulaciones históricas (no incluidas en las 240).
`prepare` realiza esa validación si falta y genera las 60 nuevas demandas.
`run` realiza los pasos pendientes y ejecuta las 240; repetirlo reanuda.
`export` reconstruye el CSV sin simular. `--output-dir` permite otro directorio.
Cambiar código, recursos, entorno o protocolo requiere un directorio nuevo:
no se mezclan experimentos. No se hacen commits ni push.

## Demanda y archivos

`tools/randomTrips.py` conserva sus opciones por defecto para selección de
orígenes/destinos y usa `--route-file --validate` para enrutamiento con duarouter.
Periodo 2 s: 1800 intentos/h; periodo 0,5 s: 7200 intentos/h. Las referencias
contienen 1126 y 4501 vehículos **programados**, no necesariamente insertados.
La semilla aleatoriza orígenes/destinos; no define la intensidad.
La reproducción exige iguales vehículos, atributos, salidas y aristas; ignora
comentarios y fechas XML. No se ajustan cantidades ni se remuestrean semillas.

`demand_manifest.csv` registra intentos, trips crudos antes de validación,
pérdidas de generación/enrutamiento, vehículos programados, periodo, intensidad,
comandos y hashes. `manifest.json` congela protocolo, entorno y hashes del código
y recursos. `runs.sqlite` conserva resultados compactos, firmas canónicas,
estados, errores y tiempos de intentos. Las 60 rutas son los únicos archivos
de demanda conservados; trips y alternativas de enrutamiento son temporales.
No se exportan trazas por vehículo.

### Ejecución paralela autorizada

`tools/run_matched_parallel.py` es un coordinador separado: reutiliza sin cambios
los procesos de simulación y métricas congelados. Antes de iniciarlo se debe
detener el ejecutor secuencial; nunca ejecutar ambos coordinadores a la vez.
Un único propietario publica SQLite/CSV. Los hilos sólo lanzan procesos aislados
SUMO/Python; no comparten una conexión TraCI. Las claves y comprobaciones de
reanudación originales conservan las corridas ya terminadas.

`--autotune` repite cuatro corridas completadas en paralelo y compara sus
resúmenes, columnas históricas y siete firmas canónicas. Las repeticiones son
validación y no sustituyen resultados ni añaden filas al CSV principal. Si la
estimación de rendimiento supera el secuencial y conserva RAM suficiente,
repite cinco; selecciona cinco sólo si mejora al menos un 5 % sobre cuatro y
deja al menos 1 GiB disponible durante la prueba. La comparación mide trabajo
completado por tiempo total, no latencia individual. Los tiempos secuenciales
históricos proceden de otra carga del equipo y son una referencia aproximada.

`parallel_execution.json` conserva pruebas, tiempos, RAM disponible mínima,
concurrencia elegida y hash del coordinador. Cada evidencia SQLite identifica
la concurrencia y hash usados; cada fila guarda el comando de reproducción.
El identificador del protocolo de simulación sigue siendo el mismo porque no
cambian recursos, demandas, controladores ni métricas. `--workers 4` permite
reanudar directamente con cuatro sin repetir la prueba; valores admitidos 1–10.
`--trial-six` requiere la prueba anterior de cinco y repite seis corridas ya
completadas. Conserva el historial de pruebas y elige seis sólo si mejora al
menos un 5 % el rendimiento estimado y la RAM disponible mínima es ≥1 GiB;
en otro caso vuelve a cinco. Estas seis repeticiones tampoco añaden filas.
`parallel.lock` impide dos propietarios paralelos. Si una interrupción deja
ese archivo, verificar que el PID registrado ya terminó antes de retirarlo.
La continuación autorizada con diez usa `--workers 10`. Cada resultado mantiene
las comprobaciones originales y se exporta por un único propietario.
`parallel_runtime.json` registra RAM disponible actual/mínima, trabajadores
activos y resultados terminados de la sesión. Para detener sin perder trabajos
en vuelo, crear `parallel_stop.request` en el directorio del experimento: el
coordinador deja de lanzar trabajos y publica los que estaban activos antes de
salir. Retirar esa solicitud después de terminar permite reanudar.
La publicación del coordinador reintenta bloqueos transitorios de Windows
durante reemplazos atómicos. Un fallo persistente del seguimiento opcional de
RAM no detiene los trabajadores. SQLite sigue siendo la fuente de resultados.

`per_demand.csv` contiene una fila por corrida terminada, en unidades originales.
Durante ejecución puede contener menos de 240 filas: `validation.json` indica
incompleto hasta terminar. Nunca se rellenan resultados fallidos con ceros.

## Diccionario de métricas y denominadores

La cohorte J2 es `-E0→E1→E5`, depart programado ≤3420 s. La cohorte completa
es `-E0→E1→E5→E10` con salida posterior por J16, también ≤3420 s. `strict`
es el subconjunto que continúa por E13. Los conjuntos programados deben coincidir
entre configuraciones; los completados pueden diferir por censura.
Parada significa velocidad <0,1 m/s en muestras de 1 s.

| Columnas | Unidad / definición |
|---|---|
| seed, traffic_level, mode, label | Semilla de demanda, nivel relativo y configuración |
| experiment_id, run_id, demand_id | Identificadores únicos del protocolo, corrida y demanda |
| route_path, route_sha256, route_semantic_sha256 | Archivo relativo al experimento, hash de bytes y hash XML sin comentarios |
| generation_period_s, nominal_attempts_per_hour | Segundos entre intentos, intentos/h nominales |
| attempted_trips, generated_trips, generation_losses, routing_losses | Conteos antes y después de generación/enrutamiento |
| scheduled_vehicles | Vehículos del XML programados durante el horizonte |
| inserted_vehicles, completed_vehicles, arrived | Vehículos insertados; llegados al destino (`completed_vehicles` = `arrived`) |
| pending, pending_active, pending_insertion | `getMinExpectedNumber`, vehículos activos, vehículos pendientes de inserción |
| pending_not_yet_attempted | Vehículos programados cuyo primer intento redondeado al paso todavía no ocurrió al final observado |
| pending_insertion_total | Cola histórica `pending_insertion` + `pending_not_yet_attempted`; vehículos |
| programmed_balance_residual | Programados − insertados − pendientes de inserción |
| inserted_balance_residual | Insertados − llegados − activos |
| pending_balance_residual | Pendientes totales − activos − pendientes de inserción |
| programmed_balance_after_boundary, pending_balance_after_boundary | Residuos anteriores menos pendientes sin primer intento; vehículos |
| cohort_J2, cohort_full, strict_cohort | Denominadores **programados** (vehículos) |
| completed_J2 | Denominador completado J2: miembros que alcanzaron E5, como el histórico |
| completed_full, completed_strict | Miembros con cruce observado de J16 |
| pending_J2, corridor_pending, pending_strict | Programados menos completados de cada cohorte |
| cohort_J2_sha256, cohort_full_sha256, cohort_strict_sha256 | Firmas de IDs programados, independientes de completitud |
| cohort_J2_observed_E1 | Miembros de la cohorte J2 con al menos una muestra en E1 |
| no_stop_J2, no_stop_full, strict_no_stop | Completados sin parada observada en el tramo correspondiente |
| no_stop_J2_pct | 100 × no_stop_J2 / cohort_J2 |
| no_stop_J2_completed_pct | 100 × no_stop_J2 / completed_J2 |
| no_stop_full_pct | 100 × no_stop_full / cohort_full |
| no_stop_full_completed_pct | 100 × no_stop_full / completed_full |
| completed_J2_pct, completed_full_pct | 100 × completados / programados |
| waiting_E1_s | veh·s histórico: espera en E1 sólo de miembros que alcanzaron E5 |
| waiting_E1_all_observed_s | veh·s en E1 de **todos** los miembros observados, incluidos pendientes; no imputa espera futura ni previa a E1 |
| mean_E1_to_E5_s, n_mean_E1_to_E5 | Media de primera muestra E1 a primera muestra E5 (s); número de completados usados |
| mean_J0_to_J16_s, median_J0_to_J16_s, p90_J0_to_J16_s, n_J0_to_J16 | Tiempo desde cruce observado J0 hasta cruce J16 (s); muestra con ambos tiempos válidos |
| stops_per_vehicle, n_stops_per_vehicle | Episodios de parada sobre E1/E5/E10 por vehículo completado; tamaño de muestra |
| full_waiting_s | veh·s acumulados sobre E1/E5/E10 de toda la cohorte completa, incluidos pendientes |
| secondary_J2_s, secondary_J10_s, secondary_J16_s, secondary_total_s | veh·s de accesos no-corredor de cada receptor y suma; incluye sentido contrario, excluye J0 |
| network_waiting_s | veh·s detenidos en toda la red, incluidas conexiones internas |
| waiting_time | veh·s detenidos en carriles controlados únicos |
| effective_flow | veh·s en movimiento en esos carriles; **no** throughput |
| avg_queue_length | Media temporal de vehículos detenidos/carril controlado |
| physical_window_coverage_J2_pct, physical_window_coverage_J10_pct, physical_window_coverage_J16_pct | Porcentaje de tiempo de ventanas de llegada observado cubierto por verde; no equivale a sin-parada |
| physical_window_count_*, physical_arrival_window_*_s, physical_green_covered_*_s | Número de ventanas y denominador/numerador temporales de la cobertura |
| constraint_violations | Auditoría de duraciones y orden semafórico; no infracciones individuales ante rojo |
| teleports | Número de IDs distintos que iniciaron teleport, como la métrica histórica |
| missing_metrics, quality_flags | Motivos de ausencia y señales de teleports, censura, pendientes o residuos |
| horizon_s, actual_end_s, step_s, SUMO_seed | Horizonte, último instante muestreado, paso (s) y semilla SUMO |
| offsets, delays_s, runtime_programs | Offsets, desplazamientos acumulados (s), programas iniciales |
| checkpoint, checkpoint_sha256, controller_class | Checkpoint activo y controlador; vacíos para controles sin PPO cuando procede |
| green_min_s, green_max_s | Límites históricos del modo; usar columnas por intersección para distinguir receptores fijos |
| J0_green_min_s, J0_green_max_s, receiver_green_min_s, receiver_green_max_s, yellow_s | Límites verdes efectivos por grupo y amarillo separado (s) |
| deterministic, network_sha256, code_sha256, environment | Determinismo, hashes y versiones para reproducción |
| run_command, sumo_command_template | Comando del lote y argumentos SUMO; la red temporal aplica los offsets registrados |
| scope, order, reference_equal | Corredor completo, orden 1–4; comparación histórica no aplicable a demandas nuevas |
| elapsed_wall_s | Tiempo de ejecución real, s; diagnóstico, no métrica de tráfico |

Conteos vacíos: cero. Porcentajes con denominador cero y medias sin muestras:
vacíos con motivo registrado. Waiting E1 de toda la cohorte es ausente si nadie
fue observado en E1; si hubo observaciones sin detención, es cero legítimo.
No se normalizan valores ni se sustituyen filas por agregados. Se conserva la
precisión aplicada por las métricas históricas (distribuciones a dos decimales).

## Validación y recuperación

Las cuatro corridas históricas se comparan contra
`analysis/six_configurations/per_demand.csv`. V3 compara también el resumen y
las siete firmas de `analysis/corridor_full/reference.json`. Los archivos locales
de `results/` se usan sólo si existen. No son una dependencia obligatoria.
Tolerancia absoluta de flotantes: 1e-12; conteos y firmas exactos.

Cada corrida usa proceso aislado, sin GUI. La clave única incluye experimento,
nivel, semilla y modo. SQLite publica resultados mediante transacciones; el CSV
se reconstruye atómicamente, sin append. Se omiten resultados completados y se
repite una vez cada fallo. Una interrupción en estado running puede reanudarse;
los fallos persistentes permanecen documentados. No se cambian parámetros para
forzar terminaciones ni se amplía el horizonte.

Se requieren 240 filas únicas, 60 demandas emparejadas, restricciones legales y
hashes intactos para declarar completo el lote. Teleports, vehículos pendientes
y residuos se conservan señalados, sin excluir corridas ni exigir red vacía.
La contabilidad sin teleports/remociones debe cerrar; un residuo se registra para
investigar pérdidas/remociones, no se corrige artificialmente. Las pruebas
estadísticas quedan para después.

### Borde temporal de las demandas con periodo 0,5 s

`pending_insertion` conserva la definición histórica: tamaño de
`traci.simulation.getPendingVehicles()`, es decir, cola de inserción con retraso.
No incluye vehículos cuyo primer intento aún no ha ocurrido. SUMO 1.26 redondea
la salida 3599,5 s al intento en 3600 s; cuando TraCI devuelve tiempo 3600 s,
el último paso procesado por este protocolo fue 3599 s. Por ello, si el XML
conserva ese último vehículo, puede quedar un residuo de 1 entre programados,
insertados y cola. Se mantiene en pendientes totales, sin imputarlo a activos
ni alterar la columna histórica. No es por sí solo una pérdida por teleport.
La validación final coteja ese residuo con los horarios del XML por demanda.
`tools.finish_matched_experiment` exige la matriz exacta de 240 resultados,
verifica de forma independiente hashes, programados y parámetros, añade sólo
esos cuatro conteos auxiliares y deja intactas las métricas históricas. Persiste
los campos en SQLite para que el exportador original pueda reconstruir el CSV.
Registra el hash y definición del enriquecimiento en metadata y en
`validation.json`; sólo se ejecuta al terminar las 240 corridas.
Esto se comprueba en el código oficial de SUMO 1.26:
[redondeo de salidas](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_26_0/src/microsim/MSVehicleContainer.cpp)
y [avance del reloj después de la inserción](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_26_0/src/microsim/MSNet.cpp).

## Resultado final del lote

Finalizado: 240 corridas, 60 demandas y cuatro configuraciones por demanda.
El CSV contiene 240 filas únicas y 114 columnas, incluidas todas las columnas
históricas. Las cuatro reproducciones históricas pasaron; las demandas y cohortes
programadas coinciden entre configuraciones y los recursos protegidos conservan
sus hashes. No hubo violaciones de duración u orden semafórico ni métricas
faltantes en este lote.

Las 120 corridas de tráfico alto registraron teleports: 2786 identificadores de
vehículos sumados entre corridas, sin deduplicar entre simulaciones. Las de
tráfico bajo no registraron teleports. Los activos pendientes al horizonte
variaron entre 19 y 458 vehículos. Se conservan todas las corridas y su censura.

Después de separar los vehículos cuyo primer intento de inserción queda fuera
del horizonte, el balance de programados cierra en todas las corridas. Persisten
residuos de 1–4 vehículos en el balance de insertados/activos/completados de 24
corridas de tráfico alto, todas con teleports. Su causa individual no se puede
determinar con los registros compactos conservados; no se atribuye automáticamente
a pérdidas ni se fuerza el balance a cero. Las claves y residuos exactos están en
`validation.json`, sección `independent_final_audit.unexplained_balance_runs`.
Esta limitación debe acompañar el uso estadístico posterior del CSV.

El coordinador terminó correctamente y el seguimiento programado fue eliminado
a petición del usuario. No se realizaron commit ni push.
