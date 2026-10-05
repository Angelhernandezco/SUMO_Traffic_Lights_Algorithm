# V3: equilibrio entre progresión y espera

Demandas 42–46, SUMO seed 42, 3600 s, paso 1 s. PPO oficial determinista sólo en J0. Offsets 0/0/72/24; viajes acumulados V3 12/22/32 s. Bases reutilizadas de la comparación validada de seis configuraciones.

Objetivo práctico: al menos 84/93 sin parada, tiempo medio ≤38 s, waiting global <48495 y secundario <30041 veh·s. Umbrales de comparación, no garantías del controlador.

| Versión | Corridas | Sin parada | Tiempo J0→J16 | Waiting E1 | Waiting corredor | Waiting secundario | Waiting red ± DE | Pendientes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain con offsets | 5 | 79/93 (84.95%) | 40.40 | 271.0 | 152.0 | 30041.0 | 48495.0 ± 589.9 | 26.8 |
| V1 — inicio | 5 | 51/93 (54.84%) | 37.96 | 33.8 | 33.4 | 28464.6 | 44071.6 ± 2016.7 | 26.0 |
| V2 — copia | 5 | 61/93 (65.59%) | 42.54 | 30.2 | 127.6 | 25328.0 | 44479.8 ± 848.1 | 24.8 |
| V3 — ventana | 5 | 84/93 (90.32%) | 36.09 | 4.0 | 2.4 | 33156.6 | 49559.8 ± 2439.4 | 30.4 |
| V3.1 — recortes según colas J2 | 5 | 82/93 (88.17%) | 35.14 | 3.6 | 3.0 | 29454.0 | 45476.2 ± 1589.8 | 27.0 |
| V3.2 — colas + cierre causal J2 | 5 | 78/93 (83.87%) | 35.97 | 6.2 | 5.2 | 29093.2 | 46264.0 ± 1723.7 | 27.6 |
| V3.3 — colas + recorte diferido J2 | 5 | 76/93 (81.72%) | 35.73 | 5.8 | 5.2 | 29433.0 | 45437.4 ± 1679.4 | 25.8 |
| V3.4 — recortes según colas del corredor | 5 | 81/93 (87.10%) | 37.67 | 2.2 | 29.2 | 26572.4 | 42874.2 ± 690.8 | 25.4 |
| V3.5 — colas + apertura alineada | 5 | 82/93 (88.17%) | 37.16 | 3.4 | 23.0 | 27020.8 | 43484.6 ± 1109.5 | 26.8 |
| V3.6 — colas + margen de llegada | 5 | 82/93 (88.17%) | 35.52 | 2.0 | 11.6 | 27251.0 | 43573.6 ± 1381.5 | 27.0 |
| V3.7 — colas + margen de llegada 2 s | 5 | 87/93 (93.55%) | 34.74 | 0.4 | 11.8 | 26895.6 | 43524.6 ± 1151.5 | 27.2 |

Todos los waiting son medias por corrida en veh·s. E1 usa los 232 vehículos rectos J0→J2; corredor completo usa 93 vehículos. Secundario suma aproximaciones no-corredor de J2/J10/J16, incluido el sentido contrario; red incluye todos los vehículos. El recorrido empieza al cruzar J0.

## Resultados emparejados por demanda

| Versión | Demanda | Sin parada | Paradas/veh. | Tiempo | Waiting E1 | Waiting secundario | Waiting red | Pendientes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain_offset | 42 | 16/19 | 0.158 | 42.32 | 207 | 30471 | 47867 | 25 |
| plain_offset | 43 | 18/20 | 0.150 | 36.15 | 362 | 30408 | 49266 | 27 |
| plain_offset | 44 | 16/19 | 0.211 | 40.84 | 105 | 30578 | 48650 | 22 |
| plain_offset | 45 | 16/20 | 0.300 | 43.60 | 321 | 28904 | 47937 | 33 |
| plain_offset | 46 | 13/15 | 0.267 | 39.07 | 360 | 29844 | 48755 | 27 |
| v1 | 42 | 9/19 | 0.526 | 44.00 | 22 | 28969 | 44024 | 24 |
| v1 | 43 | 12/20 | 0.400 | 36.60 | 40 | 28100 | 43673 | 24 |
| v1 | 44 | 9/19 | 0.526 | 36.79 | 43 | 30265 | 46171 | 22 |
| v1 | 45 | 13/20 | 0.350 | 36.60 | 22 | 25136 | 40972 | 29 |
| v1 | 46 | 8/15 | 0.467 | 35.80 | 42 | 29853 | 45518 | 31 |
| v2 | 42 | 10/19 | 0.526 | 48.74 | 68 | 25629 | 44223 | 25 |
| v2 | 43 | 12/20 | 0.500 | 44.90 | 29 | 25161 | 44029 | 23 |
| v2 | 44 | 10/19 | 0.474 | 39.00 | 23 | 26235 | 45543 | 22 |
| v2 | 45 | 14/20 | 0.350 | 44.75 | 15 | 23761 | 43461 | 30 |
| v2 | 46 | 15/15 | 0.000 | 35.33 | 16 | 25854 | 45143 | 24 |
| v3 | 42 | 16/19 | 0.158 | 37.16 | 7 | 35125 | 50977 | 39 |
| v3 | 43 | 17/20 | 0.150 | 37.45 | 2 | 31719 | 48343 | 24 |
| v3 | 44 | 18/19 | 0.053 | 35.53 | 3 | 33625 | 50243 | 29 |
| v3 | 45 | 19/20 | 0.050 | 35.45 | 2 | 29642 | 45999 | 30 |
| v3 | 46 | 14/15 | 0.067 | 34.87 | 6 | 35672 | 52237 | 30 |
| v3.1 | 42 | 15/19 | 0.211 | 36.84 | 4 | 29475 | 44695 | 24 |
| v3.2 | 42 | 15/19 | 0.211 | 37.74 | 4 | 28821 | 45955 | 25 |
| v3.3 | 42 | 14/19 | 0.263 | 37.21 | 9 | 29394 | 45033 | 24 |
| v3.4 | 42 | 16/19 | 0.158 | 39.53 | 4 | 27302 | 42808 | 25 |
| v3.5 | 42 | 16/19 | 0.158 | 39.53 | 4 | 27302 | 42808 | 25 |
| v3.6 | 42 | 16/19 | 0.158 | 39.63 | 2 | 28290 | 43945 | 26 |
| v3.7 | 42 | 16/19 | 0.211 | 38.89 | 1 | 28265 | 44173 | 27 |
| v3.1 | 43 | 19/20 | 0.050 | 35.30 | 5 | 29102 | 45809 | 29 |
| v3.2 | 43 | 18/20 | 0.100 | 36.70 | 5 | 28091 | 45420 | 28 |
| v3.3 | 43 | 16/20 | 0.200 | 36.00 | 8 | 29498 | 45600 | 26 |
| v3.4 | 43 | 18/20 | 0.100 | 36.70 | 0 | 26055 | 42374 | 21 |
| v3.5 | 43 | 19/20 | 0.050 | 36.10 | 0 | 26588 | 43466 | 27 |
| v3.6 | 43 | 16/20 | 0.200 | 35.05 | 1 | 26795 | 43538 | 27 |
| v3.7 | 43 | 18/20 | 0.100 | 34.70 | 1 | 25477 | 42489 | 25 |
| v3.1 | 44 | 16/19 | 0.158 | 34.89 | 3 | 29991 | 46011 | 25 |
| v3.2 | 44 | 13/19 | 0.316 | 34.32 | 6 | 29533 | 46558 | 26 |
| v3.3 | 44 | 18/19 | 0.053 | 34.74 | 0 | 30463 | 46615 | 22 |
| v3.4 | 44 | 17/19 | 0.105 | 37.95 | 3 | 27785 | 43776 | 24 |
| v3.5 | 44 | 18/19 | 0.053 | 35.74 | 3 | 27702 | 44246 | 25 |
| v3.6 | 44 | 18/19 | 0.053 | 33.63 | 5 | 28737 | 45369 | 25 |
| v3.7 | 44 | 19/19 | 0.000 | 33.68 | 0 | 27077 | 43389 | 26 |
| v3.1 | 45 | 19/20 | 0.050 | 34.45 | 2 | 27365 | 43300 | 30 |
| v3.2 | 45 | 18/20 | 0.100 | 35.80 | 6 | 26934 | 44389 | 32 |
| v3.3 | 45 | 15/20 | 0.250 | 35.85 | 7 | 26631 | 42817 | 30 |
| v3.4 | 45 | 16/20 | 0.200 | 38.30 | 3 | 25042 | 42080 | 28 |
| v3.5 | 45 | 16/20 | 0.200 | 39.10 | 3 | 25255 | 42058 | 30 |
| v3.6 | 45 | 18/20 | 0.100 | 35.55 | 0 | 25177 | 41508 | 31 |
| v3.7 | 45 | 20/20 | 0.000 | 32.90 | 0 | 25471 | 42434 | 28 |
| v3.1 | 46 | 13/15 | 0.133 | 34.20 | 4 | 31337 | 47566 | 27 |
| v3.2 | 46 | 14/15 | 0.067 | 35.27 | 10 | 32087 | 48998 | 27 |
| v3.3 | 46 | 13/15 | 0.133 | 34.87 | 5 | 31179 | 47122 | 27 |
| v3.4 | 46 | 14/15 | 0.067 | 35.87 | 1 | 26678 | 43333 | 29 |
| v3.5 | 46 | 13/15 | 0.133 | 35.33 | 7 | 28257 | 44845 | 27 |
| v3.6 | 46 | 14/15 | 0.067 | 33.73 | 2 | 27256 | 43508 | 26 |
| v3.7 | 46 | 14/15 | 0.067 | 33.53 | 0 | 28188 | 45138 | 30 |

## Recomendación comparativa

Candidata recomendada: **v3.7**. Cumple los cuatro umbrales prácticos de equilibrio.

| Referencia | Diferencia sin parada | Cambio tiempo | Cambio waiting secundario | Cambio waiting red |
|---|---:|---:|---:|---:|
| Plain con offsets | +8 vehículos | -14.00% | -10.47% | -10.25% |
| V3 — ventana | +3 vehículos | -3.75% | -18.88% | -12.18% |

Las comparaciones son emparejadas por demanda; un mejor agregado puede coexistir con pérdidas en una demanda individual. Estas mismas cinco demandas guiaron las iteraciones: no son una validación independiente en tráfico desconocido.

Límite restante: el waiting de la cohorte completa sube de 2.4 a 11.8 veh·s por corrida, aunque mejoren la progresión y el tiempo medio. La demanda 42 concentra la mayor parte de esa espera; no todas las métricas individuales mejoran.

## Variantes y límites

- V3.1 cambia sólo la distribución del adelanto en J2: primero recorta las fases pendientes con menos vehículos detenidos; desempata por vehículos presentes y orden original. No reordena fases ni modifica el adelanto total. J10/J16 conservan V3.

- V3.2 añade a V3.1 el cierre temprano sólo en J2: liberación terminada, cola objetivo ya cubierta, ninguna solicitud pendiente distinta y E1 completamente vacío. No supone tamaño de pelotón; el amarillo se ejecuta por SUMO, sin saltar fases.

- V3.3 parte de V3.1, sin cierre temprano: conserva el aviso pero difiere la aplicación de recortes de J2 hasta la última oportunidad legal con 1 s de margen. El primer cruce real puede corregir el frente antes de comprometer un recorte; se cierran solicitudes vacías o vencidas.

- V3.4 parte de V3.1 y aplica el mismo reparto por colas en J2/J10/J16. Sólo amplía el ámbito de ese reparto; no incluye cierre temprano ni preparación diferida. Conserva los objetivos 12/22/32 s.

- V3.5 añade a V3.4 la corrección de aperturas demasiado tempranas: si el siguiente verde natural comienza al menos 15 s antes del frente, puede alargar legalmente verdes no-corredor pendientes (máximo 45 s), dando primero tiempo a las colas mayores. Intenta abrir 1 s antes del target por el muestreo, sin cambiar el target ni reservar la duración del próximo verde PPO.

- V3.6 añade a V3.5 un margen de apertura de 1 s también cuando hay que adelantar el próximo verde. El target y las ventanas validadas siguen siendo cruce real +12/22/32 s; sólo prepara la apertura un paso de muestreo antes para cubrir llegadas algo adelantadas.

- V3.7 cambia únicamente el margen de V3.6 de 1 a 2 s, para contemplar el muestreo del cruce de J0 y de la llegada. Los targets causales y el resto del algoritmo permanecen iguales.

Se preservan V1/V2/V3, PPO, checkpoint, red y demandas. Validación automática de verdes 5–45 s, amarillos 4 s, orden cíclico, eventos únicos, frente/cola causales y coherencia de muestreo de 1 s. Las firmas de las trazas se guardan sin exportar datos crudos. Las tres lane metrics, pendientes, distribuciones de viaje y configuración se conservan en per_demand.csv.

## Ejecución

```powershell
.venv\Scripts\python.exe run_sync.py --mode v3.7 --seed 42 --gui
```

También se pueden ejecutar v1, v2, v3 y todas las variantes conservadas v3.1–v3.7. Sin --gui se ejecuta headless; --traces exporta datos detallados sólo cuando se pide explícitamente.
