# Primer experimento dinámico J0 → J2

> **Corrección de cohorte (experimento V2):** esta evaluación histórica usó las rutas `E1 E5 E10`, que incluyen giros hacia E1 y vehículos que comienzan en E1. No es una muestra exclusiva de liberaciones rectas de J0. La nueva evaluación en `j0_j2_v2_report.md` usa `-E0 E1 E5` para esa pregunta y conserva esta cohorte amplia como referencia secundaria.

## Configuración

- Rama: `master-slave-clean-v2`; SUMO 1.26.0; demanda `maps/master_slave.rou.xml`; semilla 42; paso 1 s; horizonte principal 3600 s.
- Los cuatro TLS de la red original tienen un único programa `0`, ocho fases 15/4 s y ciclo 76 s, con idénticos estados y orden. Se verificó en XML y con TraCI al iniciar SUMO. La copia temporal sólo aplica los offsets equilibrados: J0=0, J2=0, J10=72, J16=24.
- J0 usa evaluación determinista del checkpoint `model_future_v39_yellow_test36_3.pth` (SHA-256 `a3b50bcbb4419749a348261659d6254687207a9e141d20c10f06cbddd7abc6d9`), verde 5–45 s y amarillo 4 s.
- A: J2 fijo. B: target = inicio del verde recto de J0 + 9 s; se acortan exclusivamente verdes no corredores pendientes de J2 hasta un mínimo de 5 s. No se cambian fase, orden ni amarillos. J10 y J16 siguen fijos; sus inicios medidos repiten cada 76 s.
- Cohorte: 100 vehículos con ruta recta consecutiva `E1 E5 E10`, salida hasta t=3420 s y cruce completo de J2 en ambos casos. Parada: velocidad <0,1 m/s en E1. Waiting secundario: suma, por segundo, de vehículos detenidos en los accesos `-E5`, `-E6`, `-E7` de J2.

## Resultado A/B a 3600 s

| Medida | A: J2 fijo | B: J2 dinámico | Cambio B−A |
|---|---:|---:|---:|
| Cruces de J2 sin parada | 20/100 (20 %) | 25/100 (25 %) | +5 puntos porcentuales |
| Tiempo J0→J2, media | 36,72 s | 25,25 s | −11,47 s (−31,2 %) |
| Tiempo J0→J2, mediana | 39 s | 24,5 s | −14,5 s |
| Waiting J0→J2, total | 2126 vehículo-s | 1031 vehículo-s | −1095 (−51,5 %) |
| Waiting J0→J2, media por vehículo | 21,26 s | 10,31 s | −10,95 s |
| Waiting secundario J2 | 13806 vehículo-s | 10167 vehículo-s | −3639 (−26,4 %) |

De los mismos 100 vehículos, 18 pasaron de detenerse a no detenerse y 13 hicieron el cambio inverso. En el horizonte de 2000 s, sobre 55 vehículos, el cruce sin parada pasó de 21,82 % a 30,91 % y el waiting secundario bajó de 7426 a 5585 vehículo-s.

## Viabilidad de los targets

- 56 liberaciones rectas de J0 generaron 56 targets. Quince (26,8 %) permitían iniciar exactamente el siguiente verde de J2 a tiempo sin infringir los mínimos; 41 no. Dieciséis (28,6 %) quedaron dentro de ±1 s del inicio pedido. En 16 casos el target cayó dentro del verde de J2; son conjuntos distintos. El primer target era 4 s posterior al inicio natural del verde, por lo que estaba cubierto aunque no exigía adelanto.
- Adelanto solicitado: media 14,45 s, mediana 14 s, P90 21 s. Adelanto realmente obtenido: media 10,98 s, mediana 10 s, P90 15 s. La reducción aplicada coincidió con el adelanto medido en los 56 eventos.
- De los 41 targets no alcanzables exactamente, 40 quedaron tarde: retraso medio 4,85 s y máximo 7 s. El error firmado del inicio real respecto al target fue media +3,39 s, mediana +4 s, P90 +6 s, mínimo −4 s y máximo +7 s.
- Hubo 76 recortes de verde no corredor: 55 en fase 0 (526 s agregados) y 21 en fase 6 (89 s). No se recortó la fase 4. Todas las fases verdes completas de J2 quedaron entre 5 y 15 s y todos sus amarillos duraron 4 s; se observaron 443 fases completas sin infracciones.

## Interpretación

La intervención reduce claramente el tiempo y el waiting de este tramo, y no muestra coste en waiting secundario con esta demanda. La progresión sin parada sólo aumenta cinco puntos en el horizonte completo. La causa principal es el límite legal: la mayoría de targets requieren más adelanto del que permiten los verdes pendientes, y J2 puede iniciar su verde hasta 7 s tarde. No hay evidencia todavía de una onda recuperada de forma clara ni base para copiar el mecanismo sin cambios a J10/J16. Conviene resolver primero la viabilidad y repetir con más semillas; esta comparación usa una sola semilla y J0 puede modificar sus acciones por la realimentación del tráfico.
