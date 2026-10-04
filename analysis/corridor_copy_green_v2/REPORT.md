# V2 actual: copia del verde PPO en J2/J10/J16

## Comportamiento y alcance

- PPO permanece exclusivamente en J0, determinista, con el checkpoint oficial `model_future_v39_yellow_test36_3.pth`. Sin entrenamiento ni cambios de política o red.
- Al observar el primer segundo de cada verde recto de J0, se envía su duración ya decidida a los tres receptores. No se anticipa otra inferencia ni se emplea preaviso desde la fase anterior.
- Los tiempos iniciales por tramo son 11/10/10 s: objetivos acumulados desde el inicio J0 de 11/21/31 s. Son parametrizables y todavía no están calibrados conjuntamente para esta V2.
- Los receptores ajustan sólo verdes no-corredor pendientes, acortándolos o extendiéndolos lo necesario dentro de 5–45 s. Amarillos completos de 4 s y orden cíclico. Fase receptora J2=2, J10=4, J16=2, verificada por conexiones en runtime.
- Cada verde asignado ejecuta exactamente la duración decidida por PPO, incluidos verdes menores de 15 s. Una copia ya asignada no se sustituye ni se mezcla con otra.
- Cuando el inicio es inalcanzable se registra el desfase y se sirve en una apertura legal. Un verde ya iniciado sólo se aprovecha si todavía puede completar exactamente la duración copiada; su apertura anticipada y cobertura parcial quedan registradas.
- Los objetivos de J10/J16 se calculan desde J0. Un retraso real en J2 no desplaza automáticamente los objetivos posteriores.
- Se coordina únicamente el sentido J0→J2→J10→J16. V1/V3 siguen coordinando sólo J0→J2, con J10/J16 fijos. Esta comparación cambia tanto la copia de duración como el número de receptores controlados.
- La V2 causal anterior queda fuera del ejecutor activo; sus componentes necesarios para V3 y los resultados históricos se conservan.

## Validación determinista

Demanda 42, SUMO seed 42, 3600 s, paso 1 s y offsets iniciales 0/0/72/24 en copia temporal. Red original intacta. Los offsets definen el arranque; V2 modifica después duraciones y calendarios de los tres receptores.

| Receptor | Solicitudes | Copias completas exactas | Inicio objetivo alcanzable | Error de inicio mediana / P90 / máximo |
|---|---:|---:|---:|---:|
| J2 | 57 | 57/57 | 38/57 | 0 / 4 / 4 s |
| J10 | 57 | 57/57 | 57/57 | 0 / 0 / 0 s |
| J16 | 57 | 56/56 | 57/57 | 0 / 0 / 0 s |

El último verde de J16 queda censurado por el horizonte y no se cuenta como copia completa. Las fases iniciales parcialmente observadas por offsets se distinguen de fases completas; no se consideran violaciones de duración. **170/170 copias completas exactas**, sin violaciones de verdes, amarillos u orden.

Auditoría independiente: todas las duraciones y timestamps enlazan con la acción fuente de J0 y con la fase ejecutada. Los hashes de los controladores V1/V3, red, checkpoint y `policy/` coinciden con la validación anterior.

Siete pruebas en SUMO, sin PPO: copia de 5, 25 y 45 s; frente imposible; uso legal de verde actual; verde actual demasiado avanzado para copiar 5 s; y dos solicitudes sin fusionar sus verdes. Todas pasaron. En el ejemplo de 25 s, los tres receptores ejecutaron exactamente 25 s en sus aperturas desplazadas.

## Comparación preliminar, demanda 42

V1 y V3 se toman de las corridas ya verificadas con las tres métricas de carril. No se reutilizan cifras de la V2 causal histórica.

| Métrica | V1 | V2 copia | V3 |
|---|---:|---:|---:|
| J0→J2 sin parada | 31/42 (73.81%) | 34/42 (80.95%) | 38/42 (90.48%) |
| Waiting E1, cohorte recta | 19 | 68 | 9 |
| Tiempo medio E1→E5 | 14.52 s | 14.76 s | 12.07 s |
| Waiting secundario J2 | 9999 | 12022 | 17909 |
| `waiting_time`, carriles controlados | 45554 | 44220 | 53343 |
| `effective_flow`, veh·s en movimiento | 31499 | 31264 | 32225 |
| `avg_queue_length`, veh/carril | 0.26362 | 0.25590 | 0.30870 |
| Waiting de toda la red, veh·s | 45560 | 44223 | 53353 |
| Vehículos pendientes | 25 | 25 | 42 |

Las tres métricas nuevas utilizan `get_lane_metrics`, los 48 carriles controlados sin duplicados y las mismas fórmulas de `plain.py`, incluyendo verdes y amarillos. `waiting_time` suma detenidos por paso; con paso 1 s equivale a veh·s. El flujo efectivo no es throughput: acumula tiempo en movimiento. El waiting global incluye además vehículos fuera de esos carriles y se conserva por separado.

V2 mejora el porcentaje sin parada en J2 y reduce el waiting global alrededor de 2.93% frente a V1, pero aumenta waiting E1 y secundario de J2. Los vehículos que sí paran pueden esperar más, aunque sean menos. **Copiar la duración funciona técnicamente; esta única demanda no demuestra superioridad general.**

La cohorte que mantiene exactamente todos los movimientos rectos hasta E13 tiene sólo cuatro vehículos en esta demanda: cuatro cruzaron J16 y dos pasaron los tres receptores sin parar. Tiempo J0→J16 medio 35 s. No extrapolar esta muestra ni confundirla con cohortes históricas que admitían otros movimientos en J16.

## Ejecución desde la terminal de PyCharm

```powershell
.\.venv\Scripts\python.exe analysis\run_corridor_sync.py --mode v2 --seed 42 --gui --delay 100
```

Cambiar `--mode` a `v1` o `v3` conserva sus controladores actuales. Omitir `--gui` ejecuta sin ventana. `--travel-times 11 10 10` cambia únicamente los tiempos de V2. Por defecto se crea una carpeta nueva con fecha e identificador en `analysis/corridor_sync_runs/`; no se sobrescriben resultados anteriores.

Artefactos: `seed42/seed42_v2.json`, CSV de eventos, vehículos, fases, acciones y timeline; `seed42/validation.json`; `mechanism_checks.json`. Código: `../corridor_copy_green_v2.py` y `../run_corridor_sync.py`. No commit ni push.
