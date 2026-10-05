# Comparación de seis configuraciones

Demandas 42–46; horizonte 3600 s; paso 1 s; SUMO seed 42. Ejecución secuencial en el orden solicitado.

Plain usa cuatro verdes de 15 s y amarillos de 4 s, offsets cero. Plain con offset usa 0/0/72/24. Sólo PPO usa el checkpoint oficial determinista en J0 y receptores fijos con offsets cero. V1/V2/V3 conservan su implementación y sus offsets base 0/0/72/24.

| Configuración | Sin parada J2/J10/J16 | Paradas/veh. | Media J0→J16 (s) | Waiting E1 total | Waiting corredor total | Waiting secundario medio | Waiting red medio ± DE | Pendientes red medios |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain - verde fijo | 0/93 (0.00%) | 1.385 | 85.47 | 1355 | 4340 | 26233.4 | 50197.8 ± 1189.6 | 28.8 |
| Plain - offsets fijos | 79/93 (84.95%) | 0.217 | 40.40 | 1355 | 760 | 30041.0 | 48495.0 ± 589.9 | 26.8 |
| Solo PPO en J0 | 0/93 (0.00%) | 2.772 | 114.99 | 5439 | 6406 | 26339.8 | 50011.6 ± 758.2 | 29.8 |
| V1 - adelanto por inicio | 51/93 (54.84%) | 0.454 | 37.96 | 169 | 167 | 28464.6 | 44071.6 ± 2016.7 | 26.0 |
| V2 - copia de verde PPO | 61/93 (65.59%) | 0.370 | 42.54 | 151 | 638 | 25328.0 | 44479.8 ± 848.1 | 24.8 |
| V3 - preaviso y ventana causal | 84/93 (90.32%) | 0.095 | 36.09 | 20 | 12 | 33156.6 | 49559.8 ± 2439.4 | 30.4 |

## Resultados por demanda

| Configuración | Demanda | Sin parada completa | Media J0→J16 | Waiting E1 | Waiting secundario | Waiting red | Pendientes red/corredor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain - verde fijo | 42 | 0/19 | 82.74 | 206 | 26262 | 50403 | 26/0 |
| Plain - verde fijo | 43 | 0/20 | 83.50 | 363 | 25535 | 48921 | 30/0 |
| Plain - verde fijo | 44 | 0/19 | 84.37 | 106 | 27035 | 51764 | 27/0 |
| Plain - verde fijo | 45 | 0/20 | 90.95 | 318 | 25393 | 49110 | 31/0 |
| Plain - verde fijo | 46 | 0/15 | 85.80 | 362 | 26942 | 50791 | 30/0 |
| Plain - offsets fijos | 42 | 16/19 | 42.32 | 207 | 30471 | 47867 | 25/0 |
| Plain - offsets fijos | 43 | 18/20 | 36.15 | 362 | 30408 | 49266 | 27/0 |
| Plain - offsets fijos | 44 | 16/19 | 40.84 | 105 | 30578 | 48650 | 22/0 |
| Plain - offsets fijos | 45 | 16/20 | 43.60 | 321 | 28904 | 47937 | 33/0 |
| Plain - offsets fijos | 46 | 13/15 | 39.07 | 360 | 29844 | 48755 | 27/0 |
| Solo PPO en J0 | 42 | 0/19 | 111.74 | 956 | 26361 | 50238 | 27/0 |
| Solo PPO en J0 | 43 | 0/20 | 118.70 | 1190 | 25475 | 49538 | 31/0 |
| Solo PPO en J0 | 44 | 0/19 | 120.79 | 1191 | 27143 | 51198 | 28/0 |
| Solo PPO en J0 | 45 | 0/20 | 108.25 | 977 | 25662 | 49246 | 35/0 |
| Solo PPO en J0 | 46 | 0/15 | 115.47 | 1125 | 27058 | 49838 | 28/0 |
| V1 - adelanto por inicio | 42 | 9/19 | 44.00 | 22 | 28969 | 44024 | 24/0 |
| V1 - adelanto por inicio | 43 | 12/20 | 36.60 | 40 | 28100 | 43673 | 24/0 |
| V1 - adelanto por inicio | 44 | 9/19 | 36.79 | 43 | 30265 | 46171 | 22/0 |
| V1 - adelanto por inicio | 45 | 13/20 | 36.60 | 22 | 25136 | 40972 | 29/0 |
| V1 - adelanto por inicio | 46 | 8/15 | 35.80 | 42 | 29853 | 45518 | 31/0 |
| V2 - copia de verde PPO | 42 | 10/19 | 48.74 | 68 | 25629 | 44223 | 25/0 |
| V2 - copia de verde PPO | 43 | 12/20 | 44.90 | 29 | 25161 | 44029 | 23/0 |
| V2 - copia de verde PPO | 44 | 10/19 | 39.00 | 23 | 26235 | 45543 | 22/0 |
| V2 - copia de verde PPO | 45 | 14/20 | 44.75 | 15 | 23761 | 43461 | 30/0 |
| V2 - copia de verde PPO | 46 | 15/15 | 35.33 | 16 | 25854 | 45143 | 24/0 |
| V3 - preaviso y ventana causal | 42 | 16/19 | 37.16 | 7 | 35125 | 50977 | 39/0 |
| V3 - preaviso y ventana causal | 43 | 17/20 | 37.45 | 2 | 31719 | 48343 | 24/0 |
| V3 - preaviso y ventana causal | 44 | 18/19 | 35.53 | 3 | 33625 | 50243 | 29/0 |
| V3 - preaviso y ventana causal | 45 | 19/20 | 35.45 | 2 | 29642 | 45999 | 30/0 |
| V3 - preaviso y ventana causal | 46 | 14/15 | 34.87 | 6 | 35672 | 52237 | 30/0 |

## Definiciones y comprobaciones

Cohorte completa: -E0→E1→E5→E10 y cruce de J16, incluida cualquier salida posterior; depart ≤3420 s. Salida recta E13 se registra aparte. Sin parada: velocidad <0,1 m/s nunca observada en E1/E5/E10. Paradas por vehículo cuentan episodios en esos enlaces; J0→J16 empieza al cruce real de J0, excluyendo la espera previa.

Waiting en veh·s. E1 total corresponde a la cohorte recta J0→J2; waiting corredor a la cohorte completa sobre E1/E5/E10. Waiting secundario suma aproximaciones no-corredor de J2/J10/J16, incluido el sentido contrario; excluye J0. Waiting red cuenta todos los vehículos detenidos, incluidas conexiones internas. Tiempos y paradas del resumen son medias de las medias por demanda; porcentajes son ponderados por vehículos. Waiting red y secundario son medias de cinco corridas; E1 y corredor en esta tabla son sumas de las cinco.

Las tres lane metrics se conservan en per_demand.csv: waiting_time (carriles controlados únicos), effective_flow (veh·s en movimiento, no throughput) y avg_queue_length (vehículos/carril). La cobertura física de ventanas se calcula con llegadas observadas, incluyendo cotas superiores a resolución de 1 s; no equivale a progresión sin parada.

V1/V2/V3 se comparan con reference.json antes de añadir etiquetas: métricas y siete firmas canónicas exactas. Programas iniciales 0 de 76 s, orden cíclico, amarillos completos de 4 s y verdes 5–45 s comprobados; fragmentos iniciales/finales censurados. Las bases fijas verifican además verdes de exactamente 15 s. Se usan los mismos vehículos de cohorte en las seis configuraciones; código, red, checkpoint y demandas protegidos por hashes.

## Primer tramo y métricas de carriles

Medias por corrida; las cohortes J0→J2 suman 232 vehículos, mientras la cohorte que recorre todo el corredor suma 93.

| Configuración | Sin parada J2 | Tiempo E1→E5 (s) | Waiting E1 medio | Waiting corredor medio | Lane waiting | Effective flow | Cola media/carril |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain - verde fijo | 206/232 (88.79%) | 15.34 | 271.0 | 868.0 | 50191.2 | 31672.2 | 0.2905 |
| Plain - offsets fijos | 206/232 (88.79%) | 15.31 | 271.0 | 152.0 | 48489.6 | 31091.2 | 0.2806 |
| Solo PPO en J0 | 45/232 (19.40%) | 38.10 | 1087.8 | 1281.2 | 50005.8 | 32391.6 | 0.2894 |
| V1 - adelanto por inicio | 140/232 (60.34%) | 14.57 | 33.8 | 33.4 | 44069.2 | 31150.0 | 0.2550 |
| V2 - copia de verde PPO | 178/232 (76.72%) | 13.86 | 30.2 | 127.6 | 44475.0 | 31311.2 | 0.2574 |
| V3 - preaviso y ventana causal | 220/232 (94.83%) | 12.37 | 4.0 | 2.4 | 49553.8 | 31601.0 | 0.2868 |

## Interpretación

- La onda fija con offsets obtiene 79/93 recorridos sin parada frente a 0/93 sin offsets. Reduce el tiempo medio de 85,47 a 40,40 s y el waiting global un 3,39%, pero aumenta el waiting secundario un 14,51%.
- Sólo PPO con receptores sin offsets mantiene 0/93 recorridos sin parada. El tiempo medio aumenta a 114,99 s y la espera del corredor a 1281,2 veh·s por corrida; el waiting global se mantiene próximo al plain.
- V1 tiene el menor waiting global (44071,6 veh·s): 12,16% menos que plain y 11,88% menos que sólo PPO. Su progresión completa queda en 54,84%, por debajo de la onda fija con offsets.
- V2 eleva la progresión frente a V1 a 65,59% y consigue el menor waiting secundario (25328 veh·s), pero su tiempo medio (42,54 s) y waiting del corredor (127,6 veh·s por corrida) superan los de V1. El porcentaje sin parada y el promedio de espera describen aspectos distintos de la distribución.
- V3 logra la mayor progresión (84/93, 90,32%) y el menor tiempo medio (36,09 s). Frente a la onda fija añade 5 vehículos sin parada y reduce el tiempo un 10,65%, pero aumenta el waiting global un 2,20% y el secundario un 10,37%. Frente a V1, su waiting global aumenta un 12,45%.
- Ningún vehículo de la cohorte completa queda pendiente; sí quedan vehículos de la red al terminar los 3600 s. V3 presenta el mayor promedio de pendientes (30,4), V2 el menor (24,8).

Las cifras describen estas cinco demandas. La comparación sólo PPO frente a V1/V2/V3 cambia tanto la coordinación como los offsets base, conforme a las configuraciones solicitadas; no aísla el efecto del controlador por sí solo. La comparación plain frente a plain con offsets sí mantiene los demás factores constantes.

V1: aviso desde inicio del verde, objetivos acumulados 11/21/31 s y recorte de verdes no-corredor. V2: copia de la duración PPO, objetivos 11/21/31 s. V3: preaviso desde la decisión del verde anterior, corrección del frente y cola según cruces reales, desplazamientos 12/22/32 s. Los tres coordinan J2/J10/J16; PPO controla únicamente J0 con policy/models/model_future_v39_yellow_test36_3.pth.

Recomendación: mantener plain con offsets como referencia de onda fija, V1 como referencia de waiting global, V2 como referencia de coste secundario y V3 como referencia de progresión. Estas corridas no muestran una versión que gane simultáneamente en todos los objetivos.
