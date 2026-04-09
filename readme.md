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

## Comandos base de trabajo

### Entrenamiento base v39 sin amarillo
```bash
python main.py --policy-train -m model_future_v39 -e 20 -s 2000 --min-green 5 --max-green 45
```

### Test base v39 sin amarillo
```bash
python main.py --policy-test -m model_future_v39 -s 2000 --min-green 5 --max-green 45
```

### Entrenamiento rama v39 con amarillo
```bash
python main.py --policy-train -m model_future_v39_yellow_test -e 32 -s 2000 --min-green 5 --max-green 45
```

### Test rama v39 con amarillo
```bash
python main.py --policy-test -m model_future_v39_yellow_test -s 2000 --min-green 5 --max-green 45
```
