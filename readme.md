# README.md

# SUMO Traffic Lights Algorithm — Estado actual del proyecto

## Resumen

Este proyecto busca entrenar un **controlador semafórico con PPO** para un cruce en SUMO bajo un enfoque deliberadamente simple:

- **single agent**
- **single scalar action**
- **orden fijo de fases** mediante `phase_cursor` cíclico
- la policy **no elige la siguiente fase**
- la policy **solo ajusta la duración del verde de la fase actual**
- **test determinista**
- **mismos valores de `min_green` y `max_green` en train y test**

La meta no es cambiar la secuencia de fases ni meter reglas externas, sino lograr que la policy aprenda por sí sola a:

- dar **más verde a la fase dominante**
- dejar las fases débiles **cerca del mínimo**
- y **adaptarse** cuando el tráfico se vuelve más balanceado

---

## Estado actual

### Base consolidada actual
**v39**

Modelo/base de referencia actual:
- `model_future_v39`

### Mejor resultado observado sin amarillo
- **Best deterministic waiting:** `6211`
- obtenido en training y reproducido en test determinista
- la familia v39 además mostró una banda bastante consistente entre aproximadamente:
  - `6500`
  - `6825`
  - `6850`
  - `6829`
  - `6563`
  - `6920`
  - `6998`
  - `6965`
- lo importante es que v39 ya dejó de depender de un solo pico aislado y pasó a comportarse como una familia razonablemente estable

### Rama activa actual
**v39 con amarillo**

Esta rama ya no está en estado “roto”.
Ya implementa amarillo real en simulación y se volvió una línea útil de trabajo.

Su mejor zona observada hasta ahora fue con **30 episodios**, con corridas en:
- `10926`
- `12017`
- `11584`

Con **36 episodios** no hubo mejora clara; la mejor tanda quedó en:
- `11000`
- `12279`
- `12649`

Por eso, a nivel práctico, la recomendación actual del proyecto es usar:
- **32 episodios**

como protocolo operativo para la rama con amarillo.

---

## Lectura general del estado del proyecto

Hoy el proyecto tiene dos capas claras:

### 1. v39 sin amarillo = base consolidada
La v39 sin amarillo ya probó:
- mejor pico observado del proyecto (`6211`)
- consistencia razonable entre corridas
- alineación suficiente entre train y test
- protocolo estable de entrenamiento

Esa base ya se considera **válida y consolidada**.

### 2. v39 con amarillo = rama activa de mejora realista
La rama con amarillo surgió para introducir una transición más realista entre fases.

La primera implementación del amarillo rompía el entorno porque mezclaba el costo del amarillo con el verde útil de la policy.
Eso generó tiempos de espera enormes y mucha varianza.

La implementación actual corrigió eso separando:
- verde real
- amarillo de transición
- reward del verde
- costo total del episodio

Con esa corrección, la rama con amarillo:
- dejó de colapsar
- volvió a aprender
- y alcanzó una banda de resultados mucho más útil

Aun así, **todavía no supera a la v39 sin amarillo**.

---

## Diagnóstico actual

## Qué está pasando realmente

El archivo de tráfico sigue teniendo la misma estructura conceptual:

1. **inicio:** domina fuertemente una aproximación
2. **transición:** empiezan a entrar otras direcciones
3. **más adelante:** el flujo se vuelve más mixto

Eso significa que el comportamiento óptimo no es repartir verde de forma pareja.

Lo esperable sigue siendo algo como:

- al inicio:
  - fases débiles cerca de `5`
  - fase dominante muy cerca de `45`
- después:
  - aumentar otras fases solo cuando acumulen presión real
- más adelante:
  - reparto más balanceado

## Qué ya logra la rama actual con amarillo

La política actual con amarillo ya muestra avances reales:

- las fases débiles suelen quedarse cerca de valores bajos
- las fases dominantes ya reciben bastante más tiempo
- en test determinista ya se observan casos donde la dominante se va a verdes del orden de `31 s`
- el comportamiento depende más del estado que en versiones planas

Eso significa que el agente **sí está entendiendo parcialmente la dominancia relativa**.

## Qué sigue faltando

Aun con esas mejoras, todavía no se alcanza el comportamiento ideal.

Sigue faltando que:

- las fases débiles se peguen todavía más a `min_green`
- la fase dominante se acerque más a `max_green`
- la dominante “explote” mejor el rango alto en escenarios de ultradominancia
- el comportamiento medio disminuya todavía más en escenarios claramente mixtos

En resumen:

- el lado bajo del rango está bastante bien encaminado
- el lado alto todavía no está completamente aprendido

---

## Aprendizajes acumulados de versiones anteriores

## Base sólida que se conserva
La base que sigue teniendo más sentido es la familia **v39**, porque:

- mantuvo PPO puro
- no dependió de heurísticas duras externas
- mostró consistencia real
- y demostró el mejor pico observado hasta ahora

## Qué aprendimos y no queremos repetir

### De las versiones sobreajustadas al fast pass
Se vio que empujar demasiado el ciclado rápido puede inducir una solución degenerada:

- pasar casi todo en mínimo
- mejorar waiting solo por “dar vueltas rápido”
- pero sin servir correctamente la fase dominante

Eso no es lo que se quiere.

### De la primera implementación con amarillo
Se aprendió que **no basta con meter amarillo en SUMO**.

Si el amarillo se mezcla con:
- `executed_duration`
- la reward
- la lógica de extensión útil

entonces el agente paga como si el amarillo fuera verde elegido por la policy, y eso rompe el entrenamiento.

### Lección importante
El amarillo sí puede existir, pero debe tratarse como:
- **transición del entorno**
- no como parte del verde útil que la policy decidió dar

---

## Filosofía vigente del proyecto

Se mantiene sin cambios:

- PPO puro
- single agent
- single scalar action
- orden fijo de fases
- sin early cutoff
- sin reglas duras externas
- sin recortes heurísticos tipo:
  - “si pasa X, entonces duración = Y”

Sí se aceptan:

- mejores features de estado
- mejor reward shaping
- mejoras de entrenamiento
- mejor evaluación
- separación limpia entre verde y amarillo
- test determinista

---

## Estado del controlador actual

## Lo que ya logra
- distinguir razonablemente bien la fase dominante
- reducir bastante el waiting frente a implementaciones amarillas rotas
- reproducir en test el mejor checkpoint guardado
- mantener comportamiento razonablemente consistente entre train y test
- mantener una banda útil de resultados con amarillo cuando se usa un protocolo adecuado

## Lo que todavía no logra
- no llega todavía al nivel de la v39 sin amarillo
- no se va lo suficiente a los extremos en dominancia brutal
- no acerca la fase dominante a `max_green` tan a menudo como se quisiera
- todavía aparece cierto comportamiento conservador en la zona alta del rango

---

## Objetivo inmediato

El objetivo inmediato ya no es cambiar de paradigma.

Tampoco es seguir aumentando episodios indiscriminadamente.

El objetivo ahora es:

- tomar **v39 con amarillo** como rama activa
- usar **32 episodios** como protocolo práctico
- y afinar el comportamiento para que en casos de ultradominancia:
  - la dominante reciba todavía más verde
  - las fases débiles se queden todavía más cerca del mínimo

---

## Cambios a futuro cercano (próxima iteración)

## 1. Afinar la respuesta en dominancia extrema

La prioridad principal ya no es “más estabilidad general”, porque esa parte ya mejoró bastante.

La prioridad ahora es hacer más clara esta regla:

- si una fase domina brutalmente, darle todavía más tiempo
- si una fase está vacía o muy débil, pasarla muy rápido
- si el tráfico está más balanceado, repartir de forma continua

### Intención
Que la red no solo vea “hay presión”, sino que identifique mejor:

- dominancia extrema
- dominancia media
- escenario mixto
- fase irrelevante

---

## 2. Mantener reward alineada, sin volverla rígida

La reward actual ya parece suficientemente congruente a nivel macro.

Por eso, el objetivo no es rehacerla por completo, sino mantener el mismo enfoque:

- reducir waiting global
- servir bien una fase dominante
- mantener fast pass útil
- seguir penalizando extensiones innecesarias

### Qué NO se quiere hacer
- no meter reglas duras
- no truncar manualmente la acción
- no imponer cosas del tipo:
  - “si share > X entonces duración = 45”

---

## 3. Mantener el amarillo como transición limpia del entorno

La rama actual con amarillo debe seguir respetando esta separación:

- verde útil decidido por la policy
- amarillo de transición del entorno
- reward basada en el efecto del verde
- tiempo total del episodio incluyendo también el amarillo

Esa separación ya es parte central del proyecto actual y no se quiere perder.

---

## 4. Mantener test determinista como criterio final

Se seguirá usando:

- train con exploración PPO
- test con política determinista

Porque el criterio real del proyecto sigue siendo:

- que el comportamiento útil aparezca en **test determinista**
- no solo como episodios buenos aislados durante train

---

## Hipótesis de trabajo actual

La hipótesis vigente del proyecto es esta:

> La v39 ya resolvió buena parte de la estabilidad y la implementación correcta del amarillo ya volvió viable la rama realista, pero todavía falta que el agente use con más decisión el extremo alto del rango cuando la dominancia de una fase es muy clara.

Si esta hipótesis es correcta, entonces la siguiente mejora debería producir:

- menos políticas conservadoras en el extremo alto
- más diferencias claras entre dominante fuerte y dominante moderada
- verdes muy cortos en fases débiles
- verdes todavía más largos en la dominante cuando realmente lo merece
- y mantenimiento de la consistencia lograda

---

## Criterio de éxito de la siguiente versión

La siguiente iteración irá en la dirección correcta si en **test determinista** se observa:

- la acción depende más claramente del estado
- las fases débiles se mantienen cerca del mínimo
- la fase dominante se acerca más al máximo cuando realmente domina
- cuando el tráfico se balancea, el reparto del verde también se balancea
- la banda de resultados con amarillo se mantiene o mejora usando el protocolo de 32 episodios

Idealmente, además, la rama con amarillo debería empujar su mejor waiting más cerca de la base v39 sin amarillo.

---

## Restricciones que siguen vigentes

No se quiere cambiar:

- PPO
- una sola acción
- orden fijo de fases
- test determinista
- sin early cutoff
- sin reglas duras externas

No se quiere que el sistema gane únicamente por:

- ciclar rápido a ciegas
- colapsar a mínimo para casi todo
- producir una política casi constante
- o compensar el amarillo con heurísticas externas duras

---

## Resumen ejecutivo

### Punto actual
La **v39** ya es la base consolidada del proyecto.

### Mejor hallazgo global
La base sin amarillo mostró un mejor waiting observado de **`6211`**.

### Estado de la rama con amarillo
La rama con amarillo ya es funcional y prometedora.
Su mejor zona observada hasta ahora está aproximadamente entre:
- `10926`
- `12017`
- `11584`

### Principal limitación actual
Todavía no explota suficientemente el extremo alto del rango cuando una fase domina claramente.

### Próximo paso
La siguiente iteración se enfocará en:

- mantener la base PPO pura
- mantener acción escalar única
- mantener orden fijo de fases
- mantener amarillo bien integrado como transición del entorno
- usar **32 episodios** como protocolo práctico
- y afinar la política para que use mejor el rango alto en dominancia real

---


---

## Glosario de variables que aparecen en los logs

Esta sección resume qué significa cada campo que suele aparecer en los logs de **training** y **testing**.

## 1. Bloque general de SUMO / ejecución

Estas líneas las imprime SUMO alrededor de cada episodio o test:

- `Duration`: tiempo real total que tardó esa ejecución.
- `TraCI-Duration`: parte del tiempo gastada en la comunicación con TraCI.
- `Real time factor`: relación entre tiempo simulado y tiempo real. Más alto = simulación más rápida respecto al reloj real.
- `UPS`: *updates per second* aproximados de la simulación.
- `Inserted`: vehículos que realmente entraron a la red en esa ejecución.
- `Loaded`: vehículos cargados desde la ruta/archivo, aunque no necesariamente todos alcanzaron a entrar.
- `Running`: vehículos que todavía seguían en la red al terminar la ejecución.
- `Waiting`: vehículos detenidos o esperando al final de esa ejecución.
- `Emergency Stops` / `Emergency Braking`: eventos de frenado fuerte o parada de emergencia reportados por SUMO cuando aparecen en el log.

---

## 2. Campos principales del training

Las líneas de training tienen dos formatos principales.

### 2.1 Línea de acumulación antes del update PPO

Ejemplo:

```text
Episode 7/32 | Train waiting: 15025 | Train reward: -1219.84 | rollout_accum=1/2 | update=pending | obs_rms=frozen | entropy_coef=0.0251
```

Significado:

- `Episode 7/32`: episodio actual y total configurado en el entrenamiento.
- `Train waiting`: waiting total observado en **ese episodio de train**.
- `Train reward`: reward total acumulada en **ese episodio de train**.
- `rollout_accum=1/2`: cuántos episodios se han acumulado en el buffer antes de correr el update PPO. En esta versión normalmente se acumulan `2` episodios por update.
- `update=pending`: todavía no se hizo update; solo se está acumulando rollout.
- `obs_rms=live`: la normalización de observaciones (`obs_rms`) todavía se está actualizando.
- `obs_rms=frozen`: la normalización ya se congeló y dejó de actualizarse.
- `entropy_coef`: coeficiente actual del término de entropía, usado para modular exploración.

### 2.2 Línea después del update PPO

Ejemplo:

```text
Episode 22/32 | Rollout eps: 2 | Train waiting(avg): 17719 | Train reward(avg): -1478.01 | Eval waiting(det): 12648 | Eval reward(det): -897.01 | policy_loss=... | value_loss=... | entropy=... | obs_rms=frozen | entropy_coef=...
```

Significado:

- `Rollout eps`: cantidad de episodios usados en ese update PPO.
- `Train waiting(avg)`: waiting promedio de los episodios acumulados antes del update.
- `Train reward(avg)`: reward promedio de esos episodios acumulados.
- `Eval waiting(det)`: waiting de la **evaluación determinista** que se corre justo después del update. Este es el indicador más importante dentro del training.
- `Eval reward(det)`: reward de esa evaluación determinista.
- `policy_loss`: pérdida de la policy en el update PPO.
- `value_loss`: pérdida del crítico / value function en el update PPO.
- `entropy`: entropía media de la policy en ese update; más alta suele implicar policy más dispersa/exploratoria.
- `obs_rms`: estado de la normalización de observaciones en ese punto (`live` o `frozen`).
- `entropy_coef`: coeficiente de entropía usado en ese update.

---

## 3. Resumen de checkpoints durante training

### 3.1 Nuevo mejor checkpoint

Ejemplo:

```text
New best deterministic model saved to ... (best_eval_wait=11000, best_eval_reward=-778.45, episode=34)
```

Significado:

- `best_eval_wait`: mejor waiting determinista visto hasta ese momento en el training.
- `best_eval_reward`: reward correspondiente a ese mismo checkpoint.
- `episode`: episodio en el que apareció ese mejor checkpoint.

### 3.2 Resumen final del training

Ejemplo:

```text
Best eval summary | episode: 34 | waiting(det): 11000 | reward(det): -778.45
```

Significado:

- `episode`: episodio donde apareció el mejor checkpoint guardado.
- `waiting(det)`: mejor waiting determinista del entrenamiento.
- `reward(det)`: reward determinista asociada a ese mejor checkpoint.

---

## 4. Campos principales del testing

### 4.1 Resumen global del test

Ejemplo:

```text
Total waiting: 11000 | Total reward: -778.45
```

Significado:

- `Total waiting`: waiting total del test determinista completo.
- `Total reward`: reward total del test determinista completo.

La intención normal es que este test reproduzca el mejor checkpoint guardado durante training.

---

## 5. Variables del bloque `[DEBUG]`

En test, el proyecto imprime un bloque por decisión de fase cuando el debug está activo. En la rama actual con amarillo, el formato es similar a este:

```text
[DEBUG] phase=3 curr_p=12.55 next_p=0.00 next2_p=0.00 next3_p=0.00 share=0.926 dom_sum=12.550 dom_max=12.550 rank=1 peak_pos=1 action01=[0.646] req_dur=31 green_exec=31 yellow_exec=4 exec_dur=35 served=0.823 cost=1.092 waste=0.000 bad_delay=0.000 fast_pass=0.000 bad_ext=0.007 under_g=0.386 comp=0.000 clar=1.000 dom=0.926 r=4.977
```

### 5.1 Identificación de contexto

- `phase`: índice de la fase actual antes de ejecutar la acción.
- `curr_p`: presión/demanda de la fase actual antes de actuar.
- `next_p`: presión de la siguiente fase del cursor.
- `next2_p`: presión de la fase ubicada dos pasos adelante.
- `next3_p`: presión de la fase ubicada tres pasos adelante.

### 5.2 Variables de dominancia relativa

- `share`: proporción de la presión actual respecto al total; ayuda a saber qué tan dominante es la fase actual.
- `dom_sum`: razón de dominancia de la fase actual contra la **suma** de las otras fases.
- `dom_max`: razón de dominancia de la fase actual contra la **mejor otra fase**.
- `rank`: posición relativa de la fase actual cuando se ordenan las fases por presión. `1` suele significar que es la más fuerte.
- `peak_pos`: posición futura donde aparece la mayor presión alternativa mirando adelante en el orden fijo de fases.
- `clar`: claridad de dominancia. Más alto = la fase dominante está más claramente separada del resto.
- `comp`: nivel de competencia / balance entre fases. Más alto = el escenario está más mezclado y menos claramente dominado.
- `dom`: intensidad suave de dominancia que usa la reward shaping.

### 5.3 Acción y duración ejecutada

- `action01`: salida continua de la policy en escala `0..1`.
- `req_dur`: duración de verde pedida por la policy después de mapear `action01` al rango `min_green..max_green`.
- `green_exec`: segundos reales de verde ejecutados.
- `yellow_exec`: segundos reales de amarillo ejecutados.
- `exec_dur`: duración total realmente ejecutada en esa decisión (`green_exec + yellow_exec`).

### 5.4 Variables de servicio / costo

- `served`: qué tan bien sirvió la fase actual durante esa decisión.
- `cost`: costo principal de esa decisión dentro de la reward actual.
- `waste`: cuánto de esa extensión de verde pareció desperdiciada.
- `bad_delay`: penalización por quedarse demasiado tiempo en la fase actual cuando convendría avanzar.
- `fast_pass`: bono por pasar relativamente rápido una fase poco útil o débil.
- `bad_ext`: penalización por extensión innecesaria.
- `under_g`: penalización por dar **demasiado poco verde** cuando la fase lo necesitaba.
- `r`: reward instantánea de esa decisión/fase.

---

## 6. Nota importante para interpretar los logs

### Base sin amarillo vs rama con amarillo

En la base **sin amarillo**, el bloque debug viejo normalmente mostraba:

- `phase`
- `curr_p`, `next_p`, `next2_p`, `next3_p`
- `share`
- `peak_pos`
- `action01`
- `req_dur`
- `exec_dur`
- `served`
- `waste`
- `bad_delay`
- `fast_pass`
- `bad_ext`
- `under_g`
- `dom`
- `r`

En la rama **con amarillo bien separado**, además aparecen:

- `dom_sum`
- `dom_max`
- `rank`
- `green_exec`
- `yellow_exec`
- `cost`
- `comp`
- `clar`

Esto significa que los logs actuales de la rama con amarillo son más expresivos y permiten interpretar mejor:

- cuánto verde real dio la policy
- cuánto tiempo fue amarillo
- qué tan dominante era la fase
- y qué tan mezclado/competitivo era el estado del tráfico


## Comandos base de trabajo

### Entrenamiento base v39 sin amarillo
```bash
python main.py --policy-train -m model_future_v39 -e 20 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.v39.sumocfg
```

### Test base v39 sin amarillo
```bash
python main.py --policy-test -m model_future_v39 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.v39.sumocfg
```

### Entrenamiento rama v39 con amarillo
```bash
python main.py --policy-train -m model_future_v39_yellow_test -e 32 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.v39.sumocfg
```

### Test rama v39 con amarillo
```bash
python main.py --policy-test -m model_future_v39_yellow_test -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.v39.sumocfg
```

## Fase A: medición master/slave

`configuration.sumocfg` usa `maps/master_slave.net.xml` y `maps/master_slave.rou.xml`.
J0 es el único semáforo controlado por PPO; J2, J10 y J16 ejecutan sus
programas estáticos de SUMO. El escenario v39 anterior está separado en
`configuration.v39.sumocfg`.

```powershell
.\.venv\Scripts\python.exe main.py --policy-test -m model_future_v39_yellow_test36_3 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.v39.sumocfg
.\.venv\Scripts\python.exe main.py --policy-test -m model_future_v39_yellow_test36_3 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.sumocfg
```

`waiting_J0`, `waiting_J2`, `waiting_J10` y `waiting_J16` suman por
segundo los vehículos detenidos en los carriles controlados de cada TLS.
`waiting_total_network` usa la unión de esos carriles (los accesos
controlados de los cuatro TLS, no todas las vías de la red). `Total waiting`
continúa siendo `waiting_J0`, incluido el criterio para guardar el mejor
checkpoint. `throughput` cuenta los vehículos que llegan a destino durante
el episodio. `waiting_corridor` queda para una fase posterior.

J2 conserva una fase amarilla estática de 24 s: programa 0, índice 5,
estado `rrrrrrrrrryyyyyrrrrr`.

## Fase C0: forecast local y coordinación shadow

C0 agrega observación causal durante cada paso del verde y amarillo de J0.
Los valores predeterminados `--eta-mode baseline --coordination-mode off`
conservan la ejecución anterior. Los modos nuevos están disponibles únicamente
con `--policy-test`, que evalúa el checkpoint de forma determinista.

```powershell
.\.venv\Scripts\python.exe main.py --policy-test -m model_future_v39_yellow_test36_3 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.sumocfg --eta-mode local --coordination-mode off --forecast-output logs/c0/cli_off.phase_b.jsonl --coordination-output logs/c0/cli_off.jsonl
.\.venv\Scripts\python.exe main.py --policy-test -m model_future_v39_yellow_test36_3 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.sumocfg --eta-mode local --coordination-mode shadow --forecast-output logs/c0/cli_shadow.phase_b.jsonl --coordination-output logs/c0/cli_shadow.jsonl
```

`local/off` registra las vistas locales sin evaluar ajustes. `local/shadow`
requiere `eta-mode local` y evalúa reducciones hipotéticas de hasta 5 s,
con mínimos experimentales de 10 s para J2/J16 y 36 s para J10. Conserva
todas las duraciones reales, incluido el amarillo de 24 s de J2. El modo
experimental `advance` se describe en la sección C1 siguiente.

El forecast local conserva los IDs y la agrupación de Fase B. Usa miembros
presentes en el acceso receptor cuya ruta y conexión sirve la fase SUMO 2;
admite un remanente de un platoón cerrado originalmente de al menos dos
vehículos. Excluye inserciones directas en E1 y singletons originales.
Su disponibilidad `tiempo + distancia restante / velocidad permitida` es
una estimación optimista de recorrido libre; no predice exactamente el cruce
ni usa llegadas reales, errores o resultados observados. Cada consulta fallida
invalida ese miembro; los miembros válidos de una vista parcial siguen siendo
utilizables. Las vistas con más de un paso de antigüedad no son admisibles.

`--shadow-forecast` y `--forecast-output` mantienen el diagnóstico original
de Fase B. `local` lo habilita automáticamente. `--coordination-output`
guarda un JSONL independiente con configuración, revisiones locales,
retiros, decisiones y resumen. Sin una ruta explícita, C0 escribe
`policy/forecast_runs/<modelo>.c0.jsonl`. Los dos archivos deben ser distintos.

Las reservas shadow consumen un presupuesto ficticio por próxima ocurrencia
receptora y bloquean la siguiente ocurrencia. No desplazan aperturas nominales.
Los candidatos únicos se cuentan por `(TLS, platoón, ocurrencia)`;
las abstenciones por motivo cuentan decisiones por paso y pueden repetirse.
La cobertura cuenta miembros válidos sobre miembros locales elegibles de
platoones cerrados originalmente múltiples; las incompatibilidades se
registran por separado. `local_forecast_retired` registra `no_local_members`.

La validación recuperada y completada está en `logs/c0/report.md`.
Con SUMO 1.26.0, demanda oficial y horizonte 2000 s, los candidatos
geométricos fueron 22/9/14 y las reservas admisibles 8/5/8 para J2/J10/J16.
Estos resultados describen oportunidades sobre el baseline; no beneficios
de C1 ni reducciones de waiting.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_coordination.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_forecast.py -v
.\.venv\Scripts\python.exe tests/validate_c0.py
```

El validador usa el baseline recuperado y la evidencia histórica de
`logs/timeout_fix/`. Reutiliza corridas completas que coincidan con sus
firmas de código y hashes de salida. Intercepta la capa de escritura TraCI
y falla ante cualquier comando dirigido a J2/J10/J16. Compara acciones,
observaciones, reward, duraciones y trazas completas; exige igualdad exacta
con el baseline y JSONL idénticos en las tres repeticiones shadow.

## Fase C1: adelanto limitado de slaves

Hipótesis: ejecutar las reservas admisibles de C0 puede reducir la espera
del corredor y de la red, sin perjudicar demasiado el tráfico transversal
ni el throughput. Se conserva el checkpoint y todo el comportamiento de
J0/PPO; el único factor experimental es el timing real de las slaves.

`--coordination-mode advance` requiere `--eta-mode local` y `--policy-test`.
Usa exactamente `ShadowAdvanceEvaluator` para admitir las reservas y para
revalidarlas con una nueva lectura de señales y el presupuesto real. El
ejecutor comprueba también fase/inicio, programa, ocurrencia receptora y
revisión del forecast. Corre después del observador local en cada paso
existente; no añade pasos de simulación ni cambia el loop de J0.

Solo llama `setPhaseDuration(TLS, nueva_duración_restante)` sobre J2/J10/J16.
Reduce como máximo 5 s por próxima apertura, con una sola intervención y
descanso durante la siguiente ocurrencia receptora completa. Los mínimos
son J2=10 s, J10=36 s y J16=10 s. La reducción es el mínimo entre 5 s, el
tiempo reducible y la distancia temporal a la disponibilidad local; se
redondea hacia abajo a pasos completos y se abstiene si no alcanza un paso.
Se conserva al menos un paso futuro antes del cambio de fase.

No escribe sobre verdes receptores, amarillos ni all-red. No salta fases,
cambia programas ni compensa posteriormente el desplazamiento del ciclo.
J2 conserva su amarillo de 24 s. OFF y SHADOW conservan el esquema y los
eventos JSONL C0. ADVANCE mantiene esos eventos y agrega `c1_execution`,
con reserva/revisión, disponibilidad, apertura previa, duración restante,
reducciones solicitada/efectiva, presupuesto, descanso y resultado/motivo.
El resumen agrega intervenciones reales, segundos reducidos y reservas no
ejecutadas con sus razones. Las oportunidades de ADVANCE pertenecen a su
tráfico real modificado; no se exige conservar los conteos C0 de ese modo.

```powershell
.\.venv\Scripts\python.exe main.py --policy-test -m model_future_v39_yellow_test36_3 -s 2000 --min-green 5 --max-green 45 --sumo-config configuration.sumocfg --eta-mode local --coordination-mode advance --forecast-output logs/c1/cli_advance.phase_b.jsonl --coordination-output logs/c1/cli_advance.jsonl
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe tests/validate_c1.py
```

El validador compara local/off con C0 y ejecuta tres repeticiones ADVANCE
de 2000 s. Audita la capa `_setCmd` dentro del ejecutor para exigir cero
escrituras a J0 y únicamente `setPhaseDuration` a slaves. Verifica la
secuencia completa, duraciones de amarillos/receptores, mínimos, límites,
presupuesto, descanso, vigencia y determinismo exacto de trazas/JSONL.
Los hashes protegen checkpoint, PPO, forecast Fase B, red y demanda.
La evidencia se guarda en `logs/c1/report.md` y `validation_summary.json`.

Las métricas suplementarias del validador no participan en decisiones:
espera transversal sobre accesos de slaves excluyendo E1/E5/E10,
detenciones observadas en esos tres edges y tiempo desde detección en E1
hasta cruce de J16 para vehículos de origen master con recorrido completo.
El tiempo medio incluye solo vehículos que completan ese recorrido; se
reporta también el número incompleto/censurado y una comparación de IDs
comunes para evitar atribuir diferencias de muestra a mejora de tiempos.

Resultado C1 con la demanda oficial: waiting total 26865 → 26955 (+0.34%),
throughput 604 → 608 (+0.66%), waiting transversal 13521 → 14105 (+4.32%)
y detenciones del corredor 413 → 435 (+5.33%). Se ejecutaron 9/4/7
intervenciones y se redujeron 40/20/31 s en J2/J10/J16, sin reservas
perdidas al ejecutar. Las tres corridas ADVANCE fueron idénticas y las
46 pruebas pasaron; OFF/SHADOW conservaron C0 exactamente. C1 no cumple
el criterio de mejora de waiting total en este escenario: se detiene aquí
el experimento, sin extensión ni nuevas heurísticas. El modo sigue siendo
optativo y el predeterminado continúa en OFF.

## Auditoría diagnóstica de flujos con C1 congelado

```powershell
.\.venv\Scripts\python.exe tests/validate_flow_metrics.py
.\.venv\Scripts\python.exe tests/report_flow_metrics.py
```

Este validador añade únicamente un observador externo de evaluación,
`policy/flow_metrics.py`, después del registro existente de cada paso.
El controlador no importa ese módulo; la interfaz del observador expone
solo getters TraCI. No cambia semillas, tiempos, decisiones, condiciones,
escrituras ni pasos de simulación. Compara OFF/SHADOW/ADVANCE con sus
evaluaciones guardadas y repite OFF y ADVANCE con igualdad exacta.

El movimiento recto se define por ruta e índice actuales: E1→E5 en J2,
E5→E10 en J10 y E10→E13 en J16. Se distinguen el origen J0 verificado
por Fase B, las inserciones directas en E1 y otros orígenes. También se
reportan todos los movimientos de E1/E5/E10, incluidos giros: esa es la
extensión de acceso que sirve la fase receptora actual. Para J16 se entrega
así tanto E10→E13 como el conjunto E10→E11/E12/E13 de origen J0.
El grupo secundario es el complemento del movimiento recto por TLS; incluye
giros y tráfico del boulevard en sentido contrario, además del transversal.

Waiting suma segundos con velocidad <0.1 m/s en carriles de entrada y se
reconcilia exactamente con el waiting existente de cada slave. Una detención
es el inicio de un episodio bajo ese umbral; una primera observación ya
detenida cuenta como parada observada. Los vehículos que aún no cruzan y
los que terminan ruta sobre un acceso se identifican por separado del
throughput local. El porcentaje sin parada usa cruces completos, y los
tiempos desde la liberación de J0 usan solo recorridos completados.

Los remanentes son un subconjunto temporal según el forecast causal vigente,
sin duplicar población. Se comparan población completa, mismos IDs presentes
en ambas condiciones y mismos IDs que cruzan en ambas. En agregados, un ID
puede pertenecer a distintos grupos en distintos TLS; los waiting por paso
son disjuntos, pero los conteos de IDs de los grupos no se deben sumar.

`logs/c1_diagnostic/report.md` contiene tablas por TLS y agregadas.
`comparison.json` conserva todas las métricas y cohortes; `*.flow_metrics.json`
conserva registros por vehículo y `*.flow_events.jsonl` entradas/paradas/cruces.
`validation_summary.json` documenta hashes de congelación y comparaciones.
Esta auditoría describe efectos observados sin elegir el objetivo de tesis
ni clasificar globalmente C1 como bueno o malo.

`tests/report_flow_metrics.py` deriva tablas adicionales de los registros
guardados, sin importar TraCI ni ejecutar otra simulación. Distingue rutas
parciales del corredor de vehículos de origen J0 con ruta completa
E1,E5,E10 y una salida válida de J16. Una ruta que termina sobre E10 no
cuenta como movimiento de cruce en J16. Conserva vistas de todos los IDs,
IDs comunes y viajes completados en ambas condiciones. Para los viajes
completos suma waiting/paradas de los tres accesos por vehículo, además de
reportar cruces sin parada en todo el recorrido y tiempo post-J0 hasta J16.

`logs/c1_diagnostic/overview.md` reúne los resultados. El detalle de rutas
completas está en `full_corridor_trips.md`; la vista local de origen J0 en
`corridor_j0_any_j16_exit.md`; el desglose de cada conexión y origen en
`movement_breakdown.csv`. Las poblaciones y definiciones se identifican
por separado: estos agregados complementarios no se deben sumar entre sí.

## Preparación pre-C2 (sin planificador conjunto)

`run_policy()` mantiene su valor predeterminado histórico `max_green=60` para
llamadas directas que omiten el argumento. La CLI de `main.py` usa `45` y lo
pasa explícitamente, igual que los validadores y el checkpoint oficial
`model_future_v39_yellow_test36_3.pth` (`min_green=5`, `max_green=45`).
Las nuevas llamadas directas deben pasar ambos límites explícitamente. No se
cambió el valor interno heredado ni el mapeo de acciones. La prueba de un
checkpoint con metadatos sigue rechazando límites distintos.

Correcciones técnicas en C1: el ejecutor comprueba el horizonte de la
apertura calculada con la reducción final, después del redondeo; si queda
fuera registra `effective_opening_outside_horizon` y se abstiene. La reserva
activa de shadow se sincroniza con el presupuesto realmente ejecutado. Un
rechazo de TraCI registra `write_failed`; un comando aceptado sin cambio
observado en `getNextSwitch()` registra `write_not_applied`. Ambos liberan
la reserva activa y no activan descanso. El historial de oportunidades C0
permanece en los eventos. Un valor de `getNextSwitch()` inesperado después
de la escritura detiene la corrida para evitar contabilizar una intervención
ambigua. Los vehículos censurados por teleport, desaparición o cambio de
ruta no vuelven a entrar al forecast local aunque reaparezcan en el acceso.

La telemetría nueva vive en `policy/corridor_telemetry.py` y solo la invoca
`tests/validate_pre_c2.py` como observador externo tras el registro existente
de cada paso. El módulo de control no la importa. Su interfaz TraCI permite
únicamente getters; no agrega `simulationStep()` ni modifica decisiones.

```powershell
.\.venv\Scripts\python.exe tests\validate_pre_c2.py
```

La validación exige una referencia previa en `logs/pre_c2/before` y escribe
`logs/pre_c2/after/validation.json`, además de `*.vehicle_steps.jsonl`,
`*.vehicle_events.jsonl`, `*.platoon_steps.jsonl` y `*.corridor.json` por
modo. Las muestras incluyen tiempo, identidad y platoón originales,
TLS/acceso, edge/carril/posición/velocidad, distancia al final del carril,
fase/estado/señal del movimiento, verde receptor, vehículos delante en el
mismo carril, detención, ocupación de los carriles downstream de la conexión
e intervalo libre estimado con distancia y velocidad disponibles en ese
instante. `*.corridor.json` conserva visitas, segmentos J0→J2, J2→J10 y
J10→J16, censura y dispersión temporal de grupos. Los miembros originales
se conservan aunque ya no estén presentes; se registran por paso los
miembros observados y si están distribuidos en edges/carriles distintos.

| Evento | Definición observada | Uso retrospectivo |
|---|---|---|
| `j0_release_observed` | Primera detección verificada en E1 tras J0 | Inicio J0→J2, no instante subsegundo de stop line J0 |
| `approach_enter` / `next_link_enter` | Primera muestra en E1, E5 o E10 | Llegada al acceso; puede estar censurada a izquierda |
| `queue_proxy_enter` | Velocidad <0.1 m/s y otro vehículo por delante en el mismo carril | Proxy reproducible de cola, no cola física confirmada |
| `stop_begin` / `stop_end` | Cruce del umbral 0.1 m/s entre muestras | Episodios detenidos y waiting por segmento |
| `stop_line_cross` | Transición observada del acceso a carril interno o salida prevista | Cruce J2/J10/J16 y ciclo receptor observado |
| `intersection_exit` | Entrada observada en el edge de salida | Inicio efectivo del tramo siguiente |
| `vehicle_censored` | Teleport, desaparición o cambio de ruta | Excluir cruces y grupos incompletos |

Cada segmento guarda llegada/cruce, waiting detenido, inicios de detención
y tiempo libre estimado al inicio. Las muestras por paso permiten separar
tiempo detenido con verde, rojo, amarillo o señal desconocida; el estado
de la señal observado al final del paso no prueba por sí solo la causa del
retraso. Vehículos delante y ocupación downstream son indicios causales
disponibles, no una prueba de bloqueo; no se infiere una causa exclusiva
sin capacidad adicional. Los eventos de platoón ofrecen dispersión en
entrada/cruce/salida, amplitud cabeza-cola, miembros completos y cruces en
ciclos receptores distintos. `compact_exit_percent` usa solo pares
platoón×TLS completos con al menos dos miembros esperados y la definición temporal
original (gaps ≤3 s y ventana ≤15 s); no redefine los platoones del control.
La distribución entre edges o carriles indica separación observada, pero
no basta para afirmar una partición física estable. Los instantes tienen
resolución de un paso y los periodos fuera del horizonte quedan censurados.
`arrival` significa primera muestra en el edge de acceso. En J0→J2 puede
coincidir con `j0_release_observed`, porque E1 comienza justo después de J0;
para tiempo hasta la línea de parada se usa `cross`. La distancia por paso
permite definir después un umbral de proximidad explícito sin alterar C1.
Las ventanas alcanzables/incompatibles con ±5 s pueden evaluarse después
uniendo muestras y eventos C0/C1 con los límites de fase ya registrados;
la telemetría no toma esa decisión ni implementa el predictor encadenado.
