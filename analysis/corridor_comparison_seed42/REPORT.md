# Comparación medida V1 / V2 copia / V3 — demanda 42

## Método y grupos de vehículos

Se reprodujeron las tres ejecuciones deterministas, 3600 s, SUMO seed 42, checkpoint oficial y misma demanda. Sólo se añadió observación pasiva de vehículos en J10/J16: no se cambió ningún controlador, PPO, red, offset ni parámetro de sincronización. Todos los campos de los resúmenes anteriores se reprodujeron exactamente y los archivos protegidos conservaron sus hashes.

Grupos distintos:

- **42 vehículos J0→J2:** ruta `-E0 E1 E5`, salida hasta 3420 s.
- **19 vehículos que cruzan el corredor completo:** ruta `-E0 E1 E5 E10` y salida posterior de J16. Tras cruzarlo, cuatro siguen por E13, doce giran hacia E12 y tres hacia E11. Se excluyen cinco vehículos que terminan en E10 y no cruzan J16.
- **4 vehículos totalmente rectos hasta E13:** subgrupo de los 19; es el grupo estricto registrado en el reporte inicial de V2. No representa toda la circulación que atraviesa los tres receptores.

Sin parada significa que nunca se observó velocidad menor de 0.1 m/s en E1/E5/E10. El tiempo completo empieza en el primer cruce real de J0 y termina en el cruce de J16, ambos observados en carril interno. Paso 1 s: los tiempos tienen resolución de 1 s. Los 19 cruzan J16 en las tres variantes, sin pendientes dentro de esta cohorte.

V1/V3 coordinan sólo J2; V2 copia el verde PPO en J2/J10/J16 con tiempos iniciales 11/10/10 s. Por tanto, la comparación cambia tanto la estrategia como el número de receptores controlados.

## Resultados

| Métrica | V1 | V2 copia | V3 |
|---|---:|---:|---:|
| Waiting de toda la red, veh·s | 45560 | **44223** | 53353 |
| Sin parada en los tres receptores, cohorte de 19 | 7/19 (36.84%) | **10/19 (52.63%)** | 6/19 (31.58%) |
| Tiempo medio cruce J0→J16 | 59.11 s | **48.74 s** | 59.79 s |
| Mediana cruce J0→J16 | 56 s | **38 s** | 58 s |
| P90 cruce J0→J16 | 86.4 s | 92.8 s | 89.2 s |
| Máximo cruce J0→J16 | 97 s | 103 s | 96 s |
| Paradas por vehículo, cohorte de 19 | 1.16 | **0.53** | 0.79 |
| Waiting E1+E5+E10, cohorte de 19, veh·s | 352 | **222** | 424 |
| Sin parada en J2, cohorte de 42 | 31/42 (73.81%) | 34/42 (80.95%) | **38/42 (90.48%)** |
| Waiting E1, cohorte de 42, veh·s | 19 | 68 | **9** |
| Tiempo medio E1→E5, cohorte de 42 | 14.52 s | 14.76 s | **12.07 s** |
| Waiting secundario J2, veh·s | **9999** | 12022 | 17909 |
| Vehículos que completan su ruta en toda la red | 1101 | 1101 | 1084 |
| Vehículos pendientes en toda la red | 25 | 25 | 42 |
| `waiting_time`, carriles controlados | 45554 | **44220** | 53343 |
| `effective_flow`, veh·s en movimiento | 31499 | 31264 | 32225 |
| `avg_queue_length`, veh/carril | 0.26362 | **0.25590** | 0.30870 |

El flujo efectivo acumula tiempo en movimiento; no cuenta vehículos completados y no determina por sí solo cuál variante es mejor. El waiting global incluye vehículos fuera de los carriles controlados; `waiting_time` de `get_lane_metrics` conserva el alcance de `plain.py`.

## Qué mejora y qué empeora

**Waiting global:** V2 reduce 2.93% frente a V1 y 17.11% frente a V3. El tiempo medio completo baja 17.54% y 18.48%, respectivamente. La espera de los 19 vehículos baja 36.93% y 47.64%.

**Progresión completa:** V2 suma tres vehículos sin parada frente a V1 y cuatro frente a V3. Por identidad de vehículo, seis pasan a circular sin parada y tres pierden esa condición frente a V1; seis ganan y dos pierden frente a V3. Doce de los 19 tienen viajes más cortos con V2 y siete tardan más, frente a cada referencia. La mejora promedio no beneficia a todos.

**V3 destaca sólo en el primer tramo:** entre los mismos 19, pasan sin detenerse en J2 12/19 con V1, 15/19 con V2 y 18/19 con V3. En J10 son 7/19, 15/19 y 8/19. La mejora local de V3 se pierde en gran parte al llegar al siguiente semáforo fijo. J16 registra 19/19, 17/19 y 17/19 sin parada: V2 tampoco mejora cada intersección individual.

**Dónde se redistribuye el waiting de toda la intersección:**

| Intersección | V1 | V2 | V3 |
|---|---:|---:|---:|
| J0 | 8911 | 9737 | 9084 |
| J2 | 12174 | 15811 | 20021 |
| J10 | 16858 | 12187 | 16575 |
| J16 | 7611 | 6485 | 7663 |

Frente a V1, J10/J16 ahorran 5797 veh·s mientras J0/J2 añaden 4463 veh·s. Son esperas de todos los movimientos entrantes, no exclusivamente del corredor. V2 reduce el total con una redistribución entre intersecciones.

**El aumento de waiting E1 está concentrado:** con V2 se detienen ocho vehículos y esperan 8.5 s por detenido, frente a once y 1.73 s con V1. El vehículo 1086 acumula 53 de los 68 veh·s de V2 (77.94%). Su verde receptor fue `[2237,2247)`, de 10 s; llegó a la zona de stop line en 2247 con amarillo y cruzó en 2298. Este registro demuestra pérdida de la ventana por el final del verde; no determina por sí solo cuánto contribuyeron colas anteriores a su llegada tardía. Con V3 el mismo vehículo esperó 4 s, y con V1 no se detuvo.

**Los viajes lentos siguen siendo un problema:** V2 mejora media y mediana, pero su P90 y máximo son mayores que en ambas referencias. Copiar correctamente la duración no garantiza que cada vehículo alcance la ventana de llegada.

**Subgrupo totalmente recto hasta E13:** los tres métodos tienen 2/4 sin parada. V2 no aumenta ese conteo, aunque baja el tiempo medio a 35 s, frente a 54.25 s en V1 y 54.5 s en V3. Su muestra es demasiado pequeña para generalizar.

## Conclusión

En esta demanda, V2 ofrece el mejor resultado conjunto de waiting global y progresión a través de los tres receptores. V3 sigue siendo mejor localmente en J0→J2. No se ha demostrado que V2 sea superior con otras demandas: sólo está medida aquí la demanda 42. La siguiente validación adecuada es comparar las tres en las mismas demandas 42–46, manteniendo estas definiciones y sin cambiar los controladores.

Artefactos: `v1/summary.json`, `v2/summary.json`, `v3/summary.json`, sus `per_vehicle.csv` y `comparison.json`. `measure.py` realiza exclusivamente observación pasiva. Los controladores y resultados anteriores se conservaron; no hubo commit ni push.
