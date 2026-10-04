# V2 causal de ventana real J0→J2

## Diseño y protocolo

Se mantuvo el checkpoint fijo `policy/models/model_future_v39_yellow_test36_3.pth`, evaluación PPO determinista, `min_green=5`, `max_green=45` y amarillo de J0 de 4 s. J0 siguió siendo el único semáforo manejado por PPO. La lógica V1 y la V2 anterior basada en la duración PPO no se modificaron; esta V2 causal vive en `analysis/j0_j2_causal_window.py`.

Al comenzar un verde recto de J0 se registra el estado de J2 y su recorrido legal pendiente. No se recorta ningún verde hasta observar el primer cruce real `-E0→E1`. Desde ese cruce, `front_target = first_cross + 12 s`; cada cruce adicional de la misma liberación actualiza `tail_target = last_cross + 12 s`. J2 recorta sólo verdes no-corredor pendientes, sin saltar fases, y al entrar en el verde corredor lo extiende lo necesario para cubrir la cola cuando es legal, hasta 45 s. No se supone un número de vehículos; una liberación vacía no dispara recortes. No hay predictor ni entrenamiento.

El disparador incluye todos los vehículos `-E0→E1` (62 en la ruta de la semilla 42), incluso los que después giran en J2. La evaluación de progresión usa, como V1, la cohorte estricta `-E0→E1→E5` con salida hasta 3420 s (42 vehículos en la semilla 42). Las ventanas y su cobertura temporal se calculan para todos los cruces que activaron V2. Los tiempos de llegada a la proximidad de J2 se miden para esos mismos vehículos; la espera y progresión se atribuyen sólo a la cohorte estricta.

La comparación usa exactamente las demandas 42–46, 3600 s y la misma semilla interna SUMO 42; V1 tiene su T calibrado a 11 s y V2 usa +12 s desde cada cruce real. Las cinco repeticiones V1 reprodujeron los resultados multi-semilla validados (`no_stop`, waiting E1, tiempo E1→E5 y waiting secundario). Cada ejecución comprobó programa TLS 0 de 76 s en la red temporal, J2 en orden cíclico, verdes 5–45 s, amarillos 4 s y J10/J16 fijos en 76 s. Hubo **cero violaciones** de fase, duración u orden. Una prueba SUMO adicional confirmó extensión de verde J2 a 21 s y su límite de 45 s.

## Comparación por semilla

| Demanda | Sin parada V1 | Sin parada V2 | Waiting E1 V1→V2 (s) | Tiempo medio E1→E5 V1→V2 (s) | Waiting secundario J2 V1→V2 (s) | Cobertura ventana V1→V2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 31/42 (73,81 %) | 25/42 (59,52 %) | 19→42 | 14,52→15,31 | 9999→10404 | 51,90→57,14 % |
| 43 | 33/50 (66,00 %) | 27/50 (54,00 %) | 34→97 | 13,90→15,86 | 9831→9885 | 63,21→54,95 % |
| 44 | 29/48 (60,42 %) | 20/48 (41,67 %) | 45→122 | 14,75→17,35 | 9698→10345 | 52,75→35,42 % |
| 45 | 31/45 (68,89 %) | 24/45 (53,33 %) | 25→104 | 13,51→16,44 | 8767→10458 | 73,33→51,14 % |
| 46 | 30/47 (63,83 %) | 18/47 (38,30 %) | 37→119 | 14,87→17,68 | 9039→9771 | 56,31→37,07 % |

En total, V1 dejó pasar sin parada **154/232 (66,38 %)** y V2 **114/232 (49,14 %)**: caída de **17,24 puntos**. La diferencia emparejada media entre semillas fue −17,23 puntos (rango −25,53 a −12,00). El waiting E1 pasó de **160 a 484 vehículo-s** (+324; 3,03 veces). El tiempo físico medio cruce J0→proximidad de J2 subió **13,65→14,31 s** (+0,66 s); el tiempo medio E1→E5 promediado por semilla subió **14,31→16,53 s** (+2,22 s). El waiting secundario de J2 sumó **47 334→50 863 vehículo-s** (+3529; +7,46 %), aunque V2 aplicó menos segundos de recorte a verdes no-corredor (2842 frente a 3274 en V1).

La cobertura temporal ponderada de las ventanas reales fue **60,12 % en V1** frente a **46,95 % en V2**. Los instantes objetivo individuales `cruce_J0+12` que cayeron en verde J2 fueron **43,57 %** frente a **35,48 %**. En la semilla 42 la cobertura temporal V2 subió, pero la cobertura de vehículos y la progresión bajaron; por eso el porcentaje de segundos verdes dentro de una ventana no basta para juzgar el resultado.

## Factibilidad, pérdidas y extensión

De 205 ventanas V2 no censuradas, **52** quedaron completamente cubiertas (25,4 %), **34** parcialmente (16,6 %) y **119** sin cobertura (58,0 %). El `front_target` no estuvo en verde en **153/205 (74,6 %)**. En los 153 casos, el inicio más temprano legal de J2, tras recortar todos los verdes no-corredor permitidos hasta el mínimo de 5 s, seguía siendo posterior al frente; no hubo frentes legalmente alcanzables que el controlador dejara escapar. La apertura real llegó después del frente una mediana de **3 s** entre los fallos (P90 10,8 s; máximo 17 s). Ninguna cola exigió superar 45 s de verde corredor.

El primer vehículo cruzó J0 en el primer segundo del verde en **180/205** ventanas. Por tanto, la principal limitación no fue una espera larga para observar el pelotón: a menudo restaban sólo unos 12 s para el frente y J2 aún debía recorrer fases verdes y amarillas obligatorias. El verde corredor se extendió en tres ocasiones reales, hasta 25, 22 y 19 s según la semilla; los demás verdes completados duraron 15 s. El ancho de la ventana rara vez fue el cuello de botella; lo fue llegar a su inicio.

## Llegada física y error de +12 s

En las 232 trayectorias estrictas V2, `cruce J0→proximidad J2` tuvo mediana **12 s** para los 114 vehículos sin parada en E1 y **17 s** para los 118 que se detuvieron. Entre las 193 ventanas con llegada observable antes del horizonte, `primera llegada − front_target` tuvo mediana **0 s**, media +2,07 s y P90 +7 s; `última llegada − tail_target` tuvo mediana **+2 s**, media +2,66 s y P90 +8 s. El +12 s describe razonablemente el vuelo libre, mientras que la cola o el rojo de J2 retrasan la llegada medida.

El estado de J2 a la llegada de los 232 vehículos de evaluación V2 fue verde en 122, rojo en 97 y ambiguo para 13 que recorrieron los últimos metros durante un cambio de fase entre muestras de 1 s. Todos los cruces J0 de esa cohorte fueron observados en la vía interna, sin sustituirlos por la entrada a E1.

De 162 liberaciones V2 que incluyeron vehículos de evaluación, el primer vehículo que disparó la ventana giró en J2 en 12. La diferencia entre ese cruce y el primer cruce de un vehículo que continuó recto fue mediana 0 s, P90 0 s y máximo 8 s. La mezcla de destinos merece atención, pero no explica por sí sola la caída general: los 153 frentes perdidos eran legalmente tardíos con las reglas actuales.

## Conclusión

**V2 causal no recupera claramente J0→J2.** La ventana basada en cruces reales es técnicamente válida y el +12 s aproxima la llegada libre, pero la mayor parte de los frentes se conoce cuando J2 ya no puede alcanzar legalmente su verde corredor. La extensión de la cola funciona y respeta el límite de 45 s, aunque hizo falta pocas veces. La progresión, espera del corredor, tiempo de viaje y waiting secundario empeoraron frente a V1 T=11 en las cinco semillas. **El mecanismo aún no está listo para encadenarse a J10**; antes hace falta resolver la factibilidad temporal de apertura de J2 bajo las mismas restricciones de fase.

Los datos reproducibles están en `paired_results.csv`, `paired_deltas.csv`, `aggregate.json` y los CSV por semilla de eventos, ventanas y vehículos. No se cambió `policy/`, la red original ni los controladores V1/antigua V2; no hubo commit ni push.
