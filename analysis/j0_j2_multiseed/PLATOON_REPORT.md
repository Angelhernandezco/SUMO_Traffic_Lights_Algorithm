# Liberaciones rectas de J0 como pelotones individuales

## Alcance y método

Se auditaron las 20 corridas V1 existentes: demandas 42–46, `T_J0_J2` de 9, 10, 11 y 12 s, horizonte 3600 s, PPO determinista en J0 y el controlador J2 V1 sin cambios. J10 y J16 siguieron fijos. La repetición sólo añadió lecturas TraCI de 1 s. En las 20 combinaciones, los vehículos que cruzaron J2, los que no se detuvieron, el waiting de E1, el tiempo medio E1→E5 y el waiting secundario de J2 coincidieron con `results.csv`.

La cohorte estricta es `-E0→E1→E5`, con 232 vehículos distintos entre las cinco demandas, observados una vez por cada T (928 trayectorias). Las cuatro repeticiones de una semilla no son observaciones independientes. Cada vehículo se asignó a un verde recto de J0 por su primera muestra dentro de la intersección J0. La llegada a J2 se definió como la primera muestra a ≤5 m de su TLS; cuando el vehículo recorrió esos últimos metros entre muestras, se registró la entrada en la intersección como cota superior de llegada (error temporal ≤1 s). Los estados semafóricos de esos casos se infirieron sólo cuando el último estado en E1 y el estado al cruzar coincidieron; si hubo cambio se marcaron ambiguos.

Los archivos completos están en `platoon_audit/releases.csv` (una fila por verde) y `platoon_audit/vehicles.csv` (una fila por vehículo), enlazados por `demand_seed`, `T_s` y `release_id`. Incluyen el inicio/duración del verde PPO, cruce individual de J0, llegada a J2, estado de J2, parada, primer/centro/último vehículo, dispersión y errores respecto al inicio objetivo y real del verde de J2. También se conservaron los CSV separados por semilla/T y una comprobación de cada replay.

## Tamaño y dispersión de los pelotones

Hubo 1130 verdes rectos de J0. En 638 pasó al menos un vehículo de la cohorte; 492 estuvieron vacíos para esta ruta. Entre los 638 ocupados hubo 400 con 1 vehículo, 186 con 2 y 52 con 3. El verde PPO de los ocupados duró mediana 10 s, P90 15 s y rango 9–23 s.

En los 238 pelotones con al menos dos vehículos, el intervalo entre primer y último cruce de J0 tuvo mediana 1,5 s, media 2,93 s y P90 9 s. El intervalo entre primera y última llegada a J2 tuvo mediana 4 s, media 4,04 s y P90 8 s. Cuando ninguno se detuvo en J2, la mediana de dispersión de llegada fue 3 s (117 pelotones); cuando al menos uno se detuvo, fue 5 s (121 pelotones). Parte de la dispersión observada en J2 es, por tanto, una consecuencia del semáforo/cola local y no de la liberación inicial.

## Desfase físico frente al objetivo de J2

El primer vehículo llegó a J2 una mediana de 13 s después del inicio del verde de J0 (media 13,84 s, P10–P90 11–17 s, rango 9–23 s). La mediana del pelotón llegó a 14 s (media 14,60 s, P10–P90 12–18 s). Por vehículo, el tramo entre el primer cruce interno de J0 y la llegada a J2 tuvo mediana 13 s; sin parada en E1 fue 12 s (620 trayectorias), y con parada 17 s (308). La diferencia evidencia que la llegada medida también incorpora retenciones producidas por J2.

Se define **desfase positivo** como llegada posterior al inicio de verde objetivo o real de J2. Los valores de la tabla son medianas en segundos, por liberación ocupada:

| T (s) | Liberaciones ocupadas | Primer vehículo − objetivo | Mediana del pelotón − objetivo | Primer vehículo − verde real J2 |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 161 | +4 | +5 | +2 |
| 10 | 158 | +3 | +4 | +1,5 |
| 11 | 158 | +2 | +3 | 0 |
| 12 | 161 | +1 | +2 | 0 |

Globalmente, el primer vehículo llegó 3 s después del objetivo (media +3,34 s, P10–P90 0–8 s); el centro llegó 4 s después (media +4,10 s, P10–P90 1–8 s). J2 abrió realmente una mediana de 3 s después del objetivo en estas liberaciones (media +2,88 s, P10–P90 0–6 s). Por eso el desfase del primer vehículo respecto al **verde real** fue mediana +1 s y P10–P90 −3 a +5 s. El objetivo nominal y el verde que V1 consigue no deben confundirse.

El estado de J2 al aproximarse los 928 vehículos fue verde en 612, rojo en 277 y ambiguo durante un cambio de fase en 39. De las llegadas, 755 se observaron directamente dentro de 5 m y 134 se infirieron por estado estable entre las dos muestras que acotan el cruce; las 39 restantes quedaron sin etiqueta de color. Un rojo a 5 m no implica necesariamente parada completa, y un verde a 5 m no descarta una parada anterior en E1. Las 928 filas incluyen ambos datos por separado.

## Señales observables en J0

La primera pasada real por J0 sucedió en el primer segundo del verde en 543/638 pelotones (85,1 %); en 95/638 (14,9 %) tardó al menos otro segundo. En los 426 pelotones cuyo primer vehículo **no se detuvo en J2**, la llegada del primero tuvo mediana 12 s desde el inicio de J0 cuando cruzó de inmediato (353 casos), frente a 15 s cuando el cruce se retrasó (73 casos). Este contraste aísla mejor el transporte físico de los retrasos creados por J2.

Entre los 385 pelotones enteramente libres de paradas en J2, la correlación de rangos con la demora del primer arribo fue:

| Señal medida en J0 | Correlación con primera llegada |
| --- | ---: |
| Demora hasta el primer cruce real de J0 | +0,395 |
| Duración del verde PPO | −0,063 |
| Número total de vehículos en el acceso al inicio | −0,098 |
| Número total detenido en el acceso al inicio | −0,106 |
| Vehículos rectos a ≤30 m al inicio, identificados por ruta | −0,382 |

La demora del primer cruce fue la señal causal más clara y mostró asociación positiva dentro de cada semilla (correlaciones de rangos +0,18 a +0,63 en pelotones totalmente libres). La cantidad de vehículos rectos próximos al stop line también se asoció con una llegada temprana; aquí esa etiqueta proviene de las rutas de SUMO y no equivale a una medición ordinaria del acceso. Los tres carriles de `-E0` admiten el movimiento recto junto con otros giros, por lo que el simple número total de vehículos no recupera bien esa información. El número finalmente liberado y la dispersión se conocen después de empezar el verde: sirven para explicar y cubrir el **ancho** del pelotón, pero no como predictores previos de su inicio.

## Lectura para un futuro T dinámico

La señal candidata es el **instante efectivo del primer cruce recto de J0**, o un indicador temprano equivalente de la salida física del frente del pelotón. Permite actualizar la referencia temporal desde el verde anunciado hacia la liberación realmente producida. En trayectorias sin parada, el tiempo adicional cruce J0→proximidad de J2 tuvo mediana 12 s (P10–P90 8–16 s). Como estimación inicial de llegada, el frente puede situarse alrededor de `primer_cruce_J0 + 12 s`, con dispersión que deberá respetar la legalidad de fases de J2. Si se busca abrir **antes** de la llegada, el objetivo de verde necesita margen previo; este análisis no calibra ese margen ni valida una nueva regla de control.

La duración PPO no explicó el desplazamiento del frente, aunque sí determina cuánto dura la liberación. Ninguna de estas correlaciones demuestra que un `T` adaptativo mejore la progresión; la llegada observada puede desplazarse por colas o rojos de J2 y las cuatro T reutilizan la misma demanda. No se entrenó ningún modelo ni se cambió el controlador.
