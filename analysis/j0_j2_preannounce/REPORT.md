# V3 J0→J2: preaviso causal desde la fase anterior

V3 recupera claramente la progresión del movimiento recto, pero empeora el balance de tráfico frente a V1. El preaviso hace alcanzables todos los frentes evaluables de estas cinco demandas; no justifica todavía extender el controlador a J10.

## Contexto inspeccionado y discrepancias

Se leyeron completos `docs/CORRIDOR_SYNC_CONTEXT.md` y los tres archivos adjuntos de contexto. Los adjuntos siguen describiendo C0/C1, shadow y auditorías de una etapa anterior como vigentes. No corresponden a los controladores aislados actuales y no se usaron para V3. El documento principal sugería conocer la duración del próximo verde recto; V3 sólo utiliza la acción ya decidida de la fase XML 0. V2 registra preparación al comenzar el recto, pero no recorta J2 hasta el primer cruce real.

La red original actual tiene únicamente programa 0 de 76 s en J0/J2/J10/J16. La copia temporal mantiene los offsets de referencia **0/0/72/24** (J0/J2/J10/J16). No se modificaron red, checkpoint, `policy/`, V1 ni V2.

## Implementación y protocolo

El controlador separado está en `analysis/j0_j2_preannounce.py`. Al decidir la fase XML 0 de J0 anuncia `inicio_anterior + duración_decidida + 4 s`, y prepara J2 para ese inicio recto `+12 s`. No realiza la inferencia del siguiente verde hasta su momento ordinario. El primer cruce real `-E0→E1` corrige el mismo evento a `cruce+12`; los siguientes amplían su cola. No se anuncia ni reserva toda la duración del verde recto PPO. Se conserva la extensión causal de V2, con máximo 45 s.

Sólo se recortan verdes no-corredor pendientes, hasta el adelanto necesario o el máximo legal. Una corrección cancela los recortes futuros que dejaron de hacer falta; los ya ejecutados no se restauran. Si el frente es inalcanzable se conserva el orden y se atiende la primera apertura legal. Las solicitudes vacías se cierran al terminar la liberación y su amarillo completo; esto permite asociar los cruces durante amarillo a la misma liberación (hubo seis en V3). Se cancelan sus recortes pendientes. Las solicitudes servidas y las censuradas por el horizonte también se cierran.

Demandas 42–46 ya existentes, horizonte 3600 s, SUMO seed 42, paso 1 s y PPO determinista del checkpoint oficial `model_future_v39_yellow_test36_3.pth`. SHA-256 del checkpoint: `a3b50bcbb4419749a348261659d6254687207a9e141d20c10f06cbddd7abc6d9`; red: `812c4283ffc00c035ba46a113f3caecfa7d28d812e67413c79cac3829a66e033`. V2/V3 se ejecutaron emparejadas por demanda. Se reprodujo V1 T=11 sin cambios para medir waiting global y pendientes, y se comprobó coincidencia exacta con sus métricas históricas. V2 también reprodujo exactamente las métricas históricas de las cinco demandas.

## Resultados por demanda

Sin parada y waiting E1 se refieren a la cohorte estricta `-E0→E1→E5`, salida hasta 3420 s. Parada: velocidad <0,1 m/s. Tiempo: primera muestra en E1 hasta primera muestra en E5, conservando la definición de V1. Waiting secundario J2: accesos `-E5`, `-E6`, `-E7`, todos sus vehículos. Waiting total: vehículo-segundos a velocidad <0,1 m/s en **toda la red**. J2 total incluye todos sus accesos y rutas, no sólo la cohorte.

| Demanda | Control | Sin parada | Waiting E1 | E1→E5 medio | Waiting secundario J2 | Waiting total red | Pendientes red |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | V1 | 31/42 (73,81%) | 19 | 14,52 s | 9999 | 45560 | 25 |
| 42 | V2 | 25/42 (59,52%) | 42 | 15,31 s | 10404 | 47670 | 25 |
| 42 | V3 | 38/42 (90,48%) | 9 | 12,07 s | 17909 | 53353 | 42 |
| 43 | V1 | 33/50 (66,00%) | 34 | 13,90 s | 9831 | 47314 | 22 |
| 43 | V2 | 27/50 (54,00%) | 97 | 15,86 s | 9885 | 46369 | 27 |
| 43 | V3 | 49/50 (98,00%) | 1 | 11,66 s | 13561 | 51058 | 31 |
| 44 | V1 | 29/48 (60,42%) | 45 | 14,75 s | 9698 | 45710 | 23 |
| 44 | V2 | 20/48 (41,67%) | 122 | 17,35 s | 10345 | 45413 | 22 |
| 44 | V3 | 45/48 (93,75%) | 8 | 12,71 s | 14212 | 51305 | 27 |
| 45 | V1 | 31/45 (68,89%) | 25 | 13,51 s | 8767 | 44260 | 30 |
| 45 | V2 | 24/45 (53,33%) | 104 | 16,44 s | 10458 | 45551 | 31 |
| 45 | V3 | 45/45 (100%) | 0 | 11,76 s | 15114 | 51029 | 28 |
| 46 | V1 | 30/47 (63,83%) | 37 | 14,87 s | 9039 | 45018 | 27 |
| 46 | V2 | 18/47 (38,30%) | 119 | 17,68 s | 9771 | 45073 | 24 |
| 46 | V3 | 43/47 (91,49%) | 6 | 12,11 s | 16147 | 53242 | 33 |

Waiting en veh·s. Todos los vehículos estrictos cruzaron J2; **cero pendientes de la cohorte**, cero teleports. Pendientes red incluye vehículos activos y pendientes de inserción: en demanda 42 hay uno esperando inserción en los tres modos; el resto de pendientes está activo. El waiting termina a 3600 s y no incluye la espera futura de esos pendientes.

| Agregado de cinco demandas | V1 T=11 | V2 causal | V3 |
| :--- | ---: | ---: | ---: |
| Sin parada | 154/232 (66,38%) | 114/232 (49,14%) | **220/232 (94,83%)** |
| Media ± DE de porcentajes entre demandas | 66,59 ± 5,09% | 49,36 ± 8,97% | 94,74 ± 4,12% |
| Waiting E1 | 160 | 484 | **24** |
| Tiempo E1→E5, ponderado por vehículos | 14,31 s | 16,55 s | **12,06 s** |
| Waiting secundario J2 | 47334 | 50863 | **76943** |
| Waiting total J2 | 60526 | 65194 | 90410 |
| Waiting total red | 227862 | 230076 | **259987** |
| Pendientes red, suma de las cinco corridas | 127 | 129 | 161 |
| Cobertura temporal de ventana `cruces+12` | 60,12% | 46,95% | **100%** |

Frente a V1, V3 gana **28,45 puntos** de progresión, reduce waiting E1 **85%** y tiempo E1→E5 **15,70%**, pero aumenta waiting secundario **62,55%** y total de red **14,10%**. Frente a V2 gana 45,69 puntos; el secundario sube 51,27% y el total 13%. Waiting secundario y total empeoran en las cinco demandas frente a V1 y V2. Las acciones y liberaciones de J0 pueden variar por realimentación del tráfico; no se emparejaron artificialmente los IDs de liberación entre controladores.

## Alcanzabilidad, aperturas y preparaciones vacías

Alcanzable significa que J2 puede estar verde en el frente respetando el recorrido y el límite de 45 s; no exige que la apertura coincida exactamente con ese instante. Un verde anterior que aún cubre el frente también sirve.

| Demanda | Avisos evaluables alcanzables | Ventanas reales cubiertas completas | Preparaciones vacías | Recorte aplicado a vacías | Recorte aplicado total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 56/56 | 39/39 | 17 | 146 s | 632 s |
| 43 | 57/57 | 41/41 | 16 | 194 s | 710 s |
| 44 | 57/57 | 41/41 | 16 | 163 s | 697 s |
| 45 | 56/56 | 42/42 | 14 | 150 s | 660 s |
| 46 | 57/57 | 39/39 | 18 | 158 s | 712 s |

Hubo **285 avisos**, dos fuera del horizonte evaluable y **283/283 alcanzables**. Incluyen 81 vacíos (28,62%). Los 202 ocupados evaluables tuvieron su frente y toda la ventana objetivo en verde. En V2 sólo 52/205 frentes quedaron verdes tras conocer el primer cruce. No hubo frentes imposibles en estas demandas V3; el manejo de imposibilidad se verificó aparte en SUMO.

Al aviso, 277 aperturas estaban previstas exactamente en el frente provisional y seis antes. Tras el primer cruce real, 174/202 aperturas asociadas coincidieron con el frente corregido y 28 fueron anteriores. `apertura − frente`: media −0,65 s, mediana 0, P90 0, rango −13 a 0 s. Las aperturas tempranas **seguían verdes en la muestra del frente**. El aviso dejó 26–39 s hasta el frente, mediana 27 s; la corrección del primer cruce fue mediana 0, P90 1 y máximo 9 s.

De los 81 avisos vacíos, 80 alcanzaron una apertura antes de conocerse el vacío y uno se canceló antes de abrir. Consumieron **811 s de recorte**, el 23,78% de los 3411 s aplicados. Ningún recorte pendiente se aplicó después del cierre. V1 recortó 3274 s, repartidos en fases 0/4/6 = 2582/0/692; V3 repartió 744/221/2446. El coste no se explica sólo por el total recortado: cambia mucho qué fase pierde verde y cuándo. Estos datos no permiten atribuir todo el deterioro a las preparaciones vacías sin otro experimento.

Todos los verdes corredores completos de V3 duraron 15 s: no hizo falta extensión en la demanda real. La extensión causal y el tope 45 sí se comprobaron en casos controlados, sin reservar toda la duración PPO.

## Marcas temporales y validación

El inicio registrado de J0 es la **primera muestra tras aplicar el estado**, `t_decisión+1`. Con esa convención, `inicio_anterior + verde_decidido + 4` coincidió exactamente con todos los inicios rectos observados. La siguiente inferencia se hizo en `inicio_recto−1`, nunca al emitir el aviso. Los intervalos observados de J2 son `[primera_muestra_verde, primera_muestra_siguiente_fase)`. Para cubrir un target puntual se incluye su muestra: duración requerida `tail−inicio+1`.

La cobertura del 100% corresponde a la ventana objetivo `cruces+12`, no a todas las llegadas físicas. Entre 191 frentes con llegada física observable, 165 (86,39%) encontraron fase corredor en la muestra de llegada. `primera_llegada − front_target`: media +0,81 s, mediana 0, P90 +4 s, rango −5 a +10 s. El recorrido cruce interno J0→proximidad J2 tuvo media 12,66 s y mediana 12 s. La cohorte usó 232 cruces internos reales de J0, sin sustituirlos por entrada a E1.

Llegada física: primera muestra a ≤5 m de J2 o cruce entre muestras como cota superior de 1 s. En V3, el color individual al llegar fue verde en 150, rojo en 22 y ambiguo por transición entre muestras en 60. Un rojo en proximidad no implica parada completa; el resultado de 220/232 usa la velocidad real durante toda E1.

Validaciones: cero infracciones en fases completas de J0/J2 (verdes 5–45, amarillos 4, orden cíclico); cero discrepancias entre intervalos y las 3600 muestras de cada corrida; cero solicitudes duplicadas; todos los eventos cerrados; J10/J16 fijos con aperturas cada 76 s y offsets originales. Las fases en curso al cortar el horizonte están marcadas como censuradas, sin provocar una transición anticipada; en demanda 46 V3 se observaron los primeros tres segundos del último amarillo programado de cuatro. No se confunde esa observación parcial con un amarillo completo de tres segundos.

Seis comprobaciones adicionales con SUMO verificaron: verde natural sin recortes, frente imposible servido en primera apertura legal, corrección que cancela recortes pendientes, preparación vacía cerrada, extensión causal hasta 45 y censura terminal sin reabrir solicitudes. La auditoría independiente revisó los 285 eventos, los tiempos de inferencia, las aperturas y los recortes. Diez reproducciones V1/V2 coincidieron con los resultados históricos. Se comprobaron hashes de red, checkpoint, código PPO y controladores anteriores; no cambiaron. Las suscripciones de velocidad usadas sólo para medir waiting global coincidieron con 10269 consultas directas.

La primera corrida V3 de demanda 45 se archivó en `initial_terminal_censor/`: permitió detectar una reapertura de una solicitud posterior al horizonte. Se corrigió únicamente ese cierre y se repitió la demanda. Los resultados finales usan esa repetición; las métricas principales no cambiaron.

## Recomendación y reproducción

El **preaviso queda validado como mecanismo de factibilidad** para J0→J2 en estas cinco demandas. V3 mejora claramente el movimiento principal, pero **no debe reemplazar V1 como referencia equilibrada ni encadenarse todavía a J10**: desplaza demasiada espera a tráfico secundario y empeora el total. Antes de extenderlo conviene estudiar el coste de las preparaciones vacías y la distribución de recortes entre fases de J2, conservando PPO y las restricciones. No se implementó otro mecanismo.

Resultados completos: `paired_results.csv`, `paired_deltas.csv`, `aggregate.json`, `trace_validation.json`, `mechanism_checks.json`, `protected_hashes.json` y los JSON/CSV por demanda/control (acciones, fases J0, eventos, ventanas, vehículos y traza J2). V1/V2 originales y sus resultados se conservan.

Con Python 3.11, dependencias existentes de `.venv` y SUMO en PATH/SUMO_HOME, una reproducción sin sobrescribir datos puede ejecutar:

```powershell
& "$env:TEMP\sumo-python311-v3\python.exe" analysis/j0_j2_preannounce.py --seeds 42 43 44 45 46 --modes v2 v3 v1 --output-dir analysis/j0_j2_preannounce_repeat
```

El runtime portátil se preparó en una carpeta temporal; no se cambió `.venv`. No hubo entrenamiento, cambios en `policy/`, red ni offsets, commit o push.
