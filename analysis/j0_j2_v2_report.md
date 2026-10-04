# J0 → J2: ventana completa de liberación, V2

> **Experimento anterior.** Esta “V2” usó la duración del verde PPO para proyectar una ventana con `T=9 s`. Es distinta de la [V2 causal posterior](j0_j2_causal_window/REPORT.md), que usa cruces reales `+12 s`. Se conservan aquí sus resultados originales.

## Protocolo

Se ejecutaron tres simulaciones deterministas de 3600 s con el mismo archivo de rutas y semilla 42: J2 fijo, V1 (adelanto del inicio) y V2 (adelanto del inicio más duración del verde corredor para cubrir el final). J0 usó el mismo checkpoint PPO `model_future_v39_yellow_test36_3.pth`, verdes 5–45 s y amarillos de 4 s; J10 y J16 mantuvieron los offsets 72 y 24, respectivamente, sin control dinámico. `T_J0_J2=9 s` en las tres simulaciones. La ventana prevista es `[inicio verde recto J0 + T, fin verde recto J0 + T)`. La V2 intenta iniciar el verde corredor de J2 para cubrir su principio y fija su duración entre 15 y 45 s para cubrir el final; sólo acorta los verdes no corredores previos hasta el mínimo legal de 5 s.

La cohorte primaria son 42 vehículos con ruta recta `-E0 E1 E5`, que efectivamente cruzan J0 hacia J2. Los 42 llegaron a E5 en cada simulación. La evaluación anterior usó 100 rutas `E1 E5 E10`; contiene entradas laterales y salidas que empiezan en E1, de modo que no aísla la liberación recta de J0. Se conserva como cohorte amplia secundaria para poder cotejar los resultados históricos.

## Resultado principal

| Métrica, 42 vehículos rectos | J2 fijo | V1: inicio | V2: ventana |
|---|---:|---:|---:|
| Cruzan J2 sin detenerse | 10/42 (23,81 %) | 24/42 (57,14 %) | 24/42 (57,14 %) |
| Tiempo J0→J2, media | 37,90 s | 14,12 s | 14,12 s |
| Tiempo J0→J2, mediana | 39 s | 15 s | 15 s |
| Waiting en E1, total | 975 vehículo-s | 35 vehículo-s | 35 vehículo-s |
| Waiting secundario de J2 | 13806 vehículo-s | 10167 vehículo-s | 10167 vehículo-s |
| Cobertura temporal de ventana PPO | 10,96 % | 64,98 % | 64,98 % |
| Ventanas PPO completas | 1/55 | 16/56 | 16/56 |

La cohorte amplia histórica conserva sus resultados: 20 % sin parada con J2 fijo y 25 % con V1 o V2. Sobre la cohorte recta, la V1 ya mejora 33,33 puntos respecto a J2 fijo. La V2 no cambia ningún resultado frente a V1 con `T=9 s`.

## Cobertura y liberación real

- J0 produjo 56 ventanas previstas en V1 y V2. La cobertura temporal agregada del verde de J2 fue **64,98 %** en ambas. Dieciséis ventanas quedaron cubiertas completamente y 19 alcanzaron al menos 80 % de cobertura. La cobertura media por ventana fue 65,36 %.
- En 40 de las 56 liberaciones, J2 no podía iniciar su verde para el principio de la ventana respetando los mínimos de verdes no corredores. Otro inicio objetivo caía 4 s después de un verde ya iniciado y estaba cubierto. Quince inicios eran alcanzables exactamente. Ningún final de ventana exigía más de 45 s de verde desde el inicio real de J2.
- La duración requerida para llegar al final de la ventana desde el inicio real de J2 tuvo media **6,5 s**, mediana 6,5 s y máximo **14 s**. Por ello los **15 s existentes** siempre bastaron para el final; V2 no extendió ningún verde en el experimento principal. Los 56 verdes corredores asociados duraron 15 s.
- Se observaron 42 entradas reales a E1 durante verdes rectos de J0, distribuidas en 30 liberaciones. La primera entrada de cada liberación ocurrió en media **4,27 s** después del inicio del verde PPO (mediana 4 s; rango 2–8 s). En media, la última entrada ocurrió **4,10 s** antes de su final. Si se proyectan estas entradas reales `+9 s`, **23/42** caen en verde de J2. Es una proyección de liberaciones observadas, no una medida de llegada al stop line de J2.
- El PPO dio 9 s de verde recto en 34 de las 56 liberaciones; el máximo observado fue 19 s. Aunque algunas acciones duran más de 15 s, el inicio tardío de J2 reduce la duración adicional necesaria para cubrir el final de esas ventanas. La pérdida de cobertura está al principio, fuera del alcance del recorte legal disponible.

## Validación y decisión

En el caso principal, V1 y V2 ejecutaron las mismas fases y duraciones. El verificador de la simulación confirmó verdes de J2 entre 5 y 45 s, amarillos de 4 s y orden de fases intacto; J10 y J16 repitieron sus inicios cada 76 s. Como comprobación aislada del código, con `T=20 s` y horizonte 500 s la V2 extendió un verde de J2 hasta 25 s y elevó la cobertura temporal del 87,01 % de V1 al 100 %; esta comprobación no forma parte de la comparación principal ni modifica la conclusión para `T=9 s`.

**Conclusión:** sincronizar también el final de la ventana no recupera progresión adicional en esta demanda, porque los 15 s de verde de J2 ya cubren el final. La V1 sí mejora de forma clara el movimiento recto frente a J2 fijo, pero aún deja 18/42 vehículos con alguna parada breve y una cobertura temporal del 64,98 %. El problema pendiente es el inicio tardío legalmente no alcanzable, no la falta de verde al final. Estos datos pertenecen a una sola semilla; J0 puede ajustar sus acciones por la realimentación del tráfico entre casos.
