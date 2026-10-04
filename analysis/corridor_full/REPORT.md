# V1/V3 completos y comparación con V2

Demanda 42–46, 3600 s, SUMO seed 42, PPO determinista oficial sólo en J0. Red/checkpoint/offsets intactos. Los controladores originales siguen disponibles con alcance `j2`.

## Alcance y dirección

Antes, V1/V3 coordinaban únicamente J2. Ahora el ejecutor activo coordina J2/J10/J16. Sentido sur→norte: J0→J2→J10→J16 por -E0→E1→E5→E10→E13. Fases rectas XML: J0/J2/J16=2; J10=4.

V1 anuncia desde el inicio recto de J0, objetivos acumulados 11/21/31 s, sólo acorta verdes no-corredor; el verde corredor permanece en 15 s. V3 anuncia al decidir la fase anterior, prepara 12/22/32 s, corrige el frente y extiende la cola con cruces reales de J0, hasta 45 s. V2 copia la duración PPO, con objetivos 11/21/31 s. Los tres usan J0 como referencia; no retiman los receptores con demoras reales en J2/J10. Los 10 s de los tramos posteriores son aproximaciones iniciales, sin nueva calibración.

## Resultados por demanda

Cohorte completa: entra recto por J0, continúa por E1/E5/E10 y cruza J16; puede girar después de J16. La salida estrictamente recta E13 se informa aparte. Depart<=3420 s; porcentaje usa todos los vehículos elegibles, incluidos pendientes.

| Demanda | Versión completa | J2 sin parada | Tres receptores sin parada | Media J0→J16 (s) | Waiting corredor (veh·s) | Waiting red (veh·s) | Sec. J2 | Pendientes red/corredor |
|---|---|---|---|---:|---:|---:|---:|---|
| 42 | V1 | 28/42 (66.67%) | 9/19 (47.37%) | 44.00 | 116 | 44024 | 13639 | 24/0 |
| 42 | V2 | 34/42 (80.95%) | 10/19 (52.63%) | 48.74 | 222 | 44223 | 12022 | 25/0 |
| 42 | V3 | 39/42 (92.86%) | 16/19 (84.21%) | 37.16 | 5 | 50977 | 19354 | 39/0 |
| 43 | V1 | 32/50 (64.00%) | 12/20 (60.00%) | 36.60 | 16 | 43673 | 13001 | 24/0 |
| 43 | V2 | 34/50 (68.00%) | 12/20 (60.00%) | 44.90 | 176 | 44029 | 11816 | 23/0 |
| 43 | V3 | 49/50 (98.00%) | 17/20 (85.00%) | 37.45 | 4 | 48343 | 15734 | 24/0 |
| 44 | V1 | 23/48 (47.92%) | 9/19 (47.37%) | 36.79 | 15 | 46171 | 13875 | 22/0 |
| 44 | V2 | 34/48 (70.83%) | 10/19 (52.63%) | 39.00 | 64 | 45543 | 12135 | 22/0 |
| 44 | V3 | 46/48 (95.83%) | 18/19 (94.74%) | 35.53 | 1 | 50243 | 16038 | 29/0 |
| 45 | V1 | 32/45 (71.11%) | 13/20 (65.00%) | 36.60 | 9 | 40972 | 12223 | 29/0 |
| 45 | V2 | 36/45 (80.00%) | 14/20 (70.00%) | 44.75 | 176 | 43461 | 11257 | 30/0 |
| 45 | V3 | 43/45 (95.56%) | 19/20 (95.00%) | 35.45 | 1 | 45999 | 14949 | 30/0 |
| 46 | V1 | 25/47 (53.19%) | 8/15 (53.33%) | 35.80 | 11 | 45518 | 13439 | 31/0 |
| 46 | V2 | 40/47 (85.11%) | 15/15 (100.00%) | 35.33 | 0 | 45143 | 11542 | 24/0 |
| 46 | V3 | 43/47 (91.49%) | 14/15 (93.33%) | 34.87 | 1 | 52237 | 18122 | 30/0 |

## Resumen de cinco demandas

Porcentajes agregados ponderados por vehículos; waiting: media ± desviación entre demandas. Tiempo: media de las medias por demanda, no media individual global.

| Versión | J2 sin parada | Corredor sin parada | Salida recta E13 | Waiting red medio ± SD | Media J0→J16 | Sec. J2 medio ± SD | Paradas/veh. |
|---|---|---|---|---:|---:|---:|---:|
| V1 | 140/232 (60.34%) | 51/93 (54.84%) | 17/27 (62.96%) | 44072 ± 2017 | 37.96 | 13235 ± 651 | 0.45 |
| V2 | 178/232 (76.72%) | 61/93 (65.59%) | 18/27 (66.67%) | 44480 ± 848 | 42.54 | 11754 ± 358 | 0.37 |
| V3 | 220/232 (94.83%) | 84/93 (90.32%) | 27/27 (100.00%) | 49560 ± 2439 | 36.09 | 16839 ± 1831 | 0.10 |

## Antes y después: demanda 42

| Configuración | J2 sin parada | Corredor sin parada | Media J0→J16 | Waiting red |
|---|---|---|---:|---:|
| V1 anterior, sólo J2 | 31/42 | 7/19 | 59.11 | 45560 |
| V3 anterior, sólo J2 | 38/42 | 6/19 | 59.79 | 53353 |
| V1 completo | 28/42 | 9/19 | 44.00 | 44024 |
| V2 completo | 34/42 | 10/19 | 48.74 | 44223 |
| V3 completo | 39/42 | 16/19 | 37.16 | 50977 |

## Ventanas y aperturas

La cobertura física mide la ventana entre primera/última llegada observada de los vehículos que alcanzan cada receptor, con muestras de 1 s y las cotas superiores descritas abajo. No usa la duración PPO como denominador.

| Versión | Cobertura física J2 | J10 | J16 |
|---|---:|---:|---:|
| V1 | 67.14% | 97.60% | 99.08% |
| V2 | 82.15% | 88.50% | 97.37% |
| V3 | 94.07% | 90.06% | 100.00% |

| V3: receptor | Avisos alcanzables/elegibles | Preparaciones vacías | Aperturas asociadas | Frentes objetivo verdes/ventanas ocupadas |
|---|---|---:|---:|---|
| J2 | 283/283 | 80 | 283 | 203/203 |
| J10 | 283/283 | 80 | 203 | 203/203 |
| J16 | 283/283 | 80 | 203 | 203/203 |

Una apertura puede haberse iniciado antes del aviso y aprovecharse; las aperturas asociadas incluyen algunas preparaciones vacías. No equivalen al número de ciclos con vehículos. Los eventos CSV guardan el inicio real y cada frente objetivo; el desplazamiento de apertura se debe calcular como actual_start-target, ya que el target_error heredado de V1 vale cero cuando ya está verde.

## Recomendación

V3 completo es la referencia de mejor progresión: 84/93 sin parada y 27/27 en salida E13, con 12 veh·s de waiting del corredor frente a 167 de V1 y 638 de V2. Su waiting global medio es 12,45% mayor que V1 y 11,42% mayor que V2. V1 tiene el menor waiting global medio, con una diferencia pequeña (0,92%) frente a V2 y peor porcentaje sin parada. V2 ofrece un compromiso de progresión/coste secundario, pero algunos vehículos que pierden su verde copiado acumulan esperas largas. V3 no domina todas las demandas: en 46, V2 logra 15/15 y V3 14/15. No se ha demostrado un ganador estadístico del waiting global con sólo estas cinco demandas.

Extender el alcance bajó el waiting global medio de V1 de 45572 a 44072 y de V3 de 51997 a 49560 veh·s. En J2, V1 bajó de 154/232 a 140/232 sin parada; V3 conserva 220/232 agregados. El PPO es el mismo, pero sus acciones pueden variar porque el control de los receptores cambia el tráfico que observa J0. El siguiente paso razonable es estudiar el coste secundario y calibrar los tramos posteriores sobre esta referencia completa; no se añadió esa lógica en este cambio.

## Métricas adicionales y validación

`per_demand.csv` incluye waiting E1, secundario de cada receptor, mediana/P90 de recorrido, paradas, llegadas/pendientes y las tres métricas de `get_lane_metrics`: `waiting_time`, `effective_flow`, `avg_queue_length`. Effective flow son veh·s en movimiento; no es throughput. Waiting red cuenta vehículos detenidos en toda la red, incluidas conexiones internas; waiting_time cuenta sólo carriles controlados únicos.

`runs/*_events.csv` conserva avisos, cortes, alcance legal y aperturas. `window_arrival_audit.csv` compara ventanas físicas y frente/cola desplazados para los vehículos que realmente llegan a cada receptor. Una llegada no observada dentro de 5 m se registra como cota superior del cruce a resolución de 1 s, con estado desconocido. La cobertura de una ventana objetivo no equivale al porcentaje de vehículos sin parada.

Cero violaciones de verdes 5–45 s, amarillos completos de 4 s y orden cíclico; los fragmentos iniciales por offset y terminales por horizonte quedan censurados. Cero teleports. Mismas rutas e IDs de cohorte entre versiones. Nueve pruebas nativas de SUMO verificaron los tres receptores, causalidad y cierre vacío. Los replays locales reprodujeron las métricas previas; la ejecución con un hilo de CPU reprodujo exactamente JSON y siete CSV de V1/V3 locales y V1/V2/V3 completos en demanda 42.

## Reproducción

```powershell
.\.venv\Scripts\python.exe analysis\run_corridor_sync.py --mode v1 --seed 42 --gui
.\.venv\Scripts\python.exe analysis\run_corridor_sync.py --mode v3 --seed 42 --gui
# Histórico: añadir --scope j2
.\.venv\Scripts\python.exe analysis\corridor_full.py --seeds 42 43 44 45 46
```

No se modificaron policy/, checkpoint, red ni offsets; sin entrenamiento, commit o push.
