# Historial experimental master/slave

> **Archivo histórico.** C0/C1, FILTERED, shadow, predictor y recuperación de calendario no forman parte del experimento actual. El estado vigente está en [CORRIDOR_SYNC_CONTEXT.md](CORRIDOR_SYNC_CONTEXT.md). Las cifras de abajo corresponden a redes y controladores anteriores.

Este resumen conserva lo aprendido de la rama histórica policy-learning y sus informes. Las implementaciones, scripts, logs y artefactos experimentales no se copian a esta rama.

## Regla para leer las cifras

Hay dos generaciones de datos. Los análisis anteriores al arreglo describen J2 con ciclo **erróneo de 96 s** y amarillo de 24 s. Los informes revalidados posteriores usan J2 con ciclo **corregido de 76 s** (15/4 repetido cuatro veces). No mezclar cifras ni usar resultados J2=96 s para diseñar sobre esta red. Los informes antiguos siguen siendo evidencia histórica, no resultados vigentes.

## Cronología

### 1. Infraestructura master/slave

Se amplió la red a J0, J2, J10 y J16; J0 mantuvo la intención de ser el master PPO y las otras señales operaron con programas estáticos. Esto aportó un corredor SUMO sobre el cual estudiar coordinación. La infraestructura de red no equivalía todavía a una estrategia de progresión.

### 2. Tracking de vehículos y platoones

Se añadieron identidad de vehículo, rutas y agrupación temporal de liberaciones para medir cómo el tráfico de J0 aparecía en intersecciones posteriores. El tracking permitió analizar progreso por vehículo y grupo. Un platoón observado no siempre permanece compacto ni constituye una unidad indivisible downstream.

### 3. Phase B ETA

Se probaron estimaciones cinemáticas de llegada libre a las siguientes intersecciones. Sirvieron como referencia para saber cuándo un vehículo podría llegar sin demoras de señal/cola; no estiman por sí solas servicio ni cruce real. En la evaluación posterior con red corregida, Fase B tuvo errores de cruce grandes frente al predictor encadenado, especialmente downstream. El MAE de Fase B no debe confundirse con ETA garantizada.

### 4. C0 shadow

C0 generó forecasts y evaluó decisiones hipotéticas sin escribir cambios a las señales. Fue útil para instrumentar elegibilidad, razones de abstención y ventanas receptoras sin efecto causal sobre SUMO. Un resultado shadow no prueba que el cambio habría mejorado el tráfico.

### 5. C1 con adelanto de hasta 5 s

C1 aplicó adelantos acotados a señales slave. En la repetición determinista sobre J2 corregido, ADVANCE dio waiting total 23,239 s frente a 24,204 s OFF y throughput 609 en ambos; hubo 24 intervenciones y 112 s acumulados de calendario modificado. El promedio del corredor común mejoró de 109.39 a 105.05 s, mientras waiting J10 subió de 4,135 a 4,544 s. Es mejora parcial bajo esa demanda, no progresión robusta garantizada. Se observaron ventanas/calendarios que derivaron.

### 6. Descubrimiento del amarillo J2 incorrecto

La configuración tenía una fase J2 de 24 s (rrrrrrrrrryyyyyrrrrr) donde las fases equivalentes usan 4 s. Eso inflaba el ciclo a 96 s, con 36 s de amarillo total y 15 s de verde receptor. Los análisis que concluyeron que J2 requería adelantos medianos cercanos a 46 s o que muchas oportunidades quedaban muy lejos correspondían a esa red defectuosa.

### 7. Corrección de J2: 96 → 76 s

Se sustituyó solo esa duración por 4 s. El ciclo resultante es 4×15 s de verde + 4×4 s de amarillo = 76 s. El baseline OFF reportado sobre la red corregida tuvo waiting J0/J2/J10/J16 = 3977/9237/4135/6855 s, total 24,204 s y throughput 609. Frente a la corrida anterior con J2=96, el total fue 26,865 s y J2 waiting 12,613 s; estas son corridas de redes distintas y no deben tratarse como una comparación aislada de controlador.

### 8. Predictor encadenado revalidado

Los nuevos informes volvieron a medir sobre J2=76 s. El forecast de cruce emitido desde J0 logró MAE de 2.48–4.74 s según TLS/modo evaluado, pero P90 absoluto llegó a 9.40 s en OFF J10. Hay censura, cobertura distinta y casos pequeños; un MAE bajo no constituye garantía temporal.

Reanclar al observar una etapa upstream puede empeorar al saltar a otra ocurrencia de verde: se documentan ejemplos de errores de 49–81 s. Vehículos delante no necesariamente forman una cola detenida; multiplicarlos por headway sin validar movimiento/ventana produjo falsos saltos de ciclo. Predictor encadenado sirve como insumo y diagnóstico con incertidumbre, no como coordinador ya validado.

### 9. Auditoría de autoridad temporal

La auditoría de esa etapa con J2=76 encontró que 5 s pueden dar alivio local limitado, pero son insuficientes para progresión continua general. En una envolvente geométrica, adelanto puro requirió medianas de 43–44 s y P90 de 52–55 s para la progresión completa; esa autoridad excede C1 y no demuestra el efecto de un controlador real. Se observaron 24 intervenciones reales de C1 repartidas entre J2/J10/J16 (9/6/9). Una mejora local de J2 puede perderse en J10/J16.

### 10. Filtro downstream / admission-aware (FILTERED)

El filtro de admisión buscó evitar entradas cuya ventana J2 parecía claramente perdida. En D0 FILTERED redujo waiting total de la corrida C1 CURRENT (23,239 s) a 22,863 s y elevó throughput de 609 a 613. Parte de la mejora vino con cambios en el costo por origen y movimiento; no todos los grupos mejoraron. No es una prueba de mejora universal del corredor.

### 11. Auditoría por origen y movimiento

La descomposición mostró por qué un total agregado puede ocultar redistribución: favorecer movimientos del corredor puede desplazar espera a tráfico directo/lateral y a grupos que no recorren toda la ruta. La auditoría separó accesos, movimientos, categorías, paradas y waiting. Mejorar la cohorte del corredor o el total de un episodio no garantiza que el costo esté equilibrado por origen.

### 12. Robustez multi-demanda

El último informe de esa línea evaluó cuatro demandas pareadas. Δ es FILTERED − CURRENT; Δ cohorte corresponde a los vehículos completados en ambos modos de cada demanda.

| Demanda | Δ waiting total | Δ waiting cohorte | Lectura |
|---|---:|---:|---|
| D0 oficial | −376 s | −6.95 s/veh. | Mejora global y de cohorte en este episodio. |
| D1 estrés | −30,899 s | −5.62 s/veh. | Teleports y 52 inserciones menos en FILTERED; no es evidencia limpia de mejora. |
| D2 seed101 | +873 s | +7.97 s/veh. | Empeora total y cohorte. |
| D3 seed202 | +127 s | −10.00 s/veh. | Casi neutro en total; la mejora de cohorte desplaza costo a otros tráficos. |

Conclusión del informe: **CASO C — sensible a la demanda**. El beneficio D0 no generalizó en D2/D3. Los alimentadores laterales E empeoraron waiting por vehículo en los cuatro pares. FILTERED mostró que la admisión downstream importa, pero no es una arquitectura robusta de sincronización.

La línea experimental C1/FILTERED se cerró allí. Sus hallazgos quedan como antecedentes, no como código de partida ni como evidencia para activar C2. Son cuatro demandas, no una caracterización exhaustiva de todas las condiciones.

## Lecciones históricas

- El error de amarillo J2 distorsionaba calendario y waiting; por eso resultados deben etiquetarse con la red exacta.
- Un adelanto local ≤5 s puede ayudar a waiting en una demanda y no crear progresión completa robusta.
- El predictor puede ser preciso en promedio y aún sufrir saltos de ventana grandes al reanclar.
- Repetir correcciones pequeñas por slave puede acumular deriva y mover el problema a downstream.
- FILTERED mejoró el episodio D0, pero el beneficio no fue robusto ante demandas moderadas y redistribuyó costo entre orígenes/movimientos.
- Los alimentadores laterales E empeoraron waiting por vehículo en 4/4 pares FILTERED/CURRENT.
- Waiting total, progreso J0→J2→J10→J16, throughput y costo por origen/movimiento son dimensiones relacionadas, no intercambiables.

## Fuentes históricas

Los informes fuente en la rama histórica incluyen analysis/post_j2_predictor/REPORT.md, analysis/post_j2_authority/REPORT.md, analysis/post_j2_admission/REPORT.md, analysis/post_j2_admission_flow_audit/REPORT.md, analysis/post_j2_admission_robustness/REPORT.md y analysis/offline_c2_report.md. La última contiene explícitamente resultados anteriores calculados con J2=96 s; se referencia solo como registro histórico. Los archivos fuente permanecen en el workspace histórico y no fueron copiados.
