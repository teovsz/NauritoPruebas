# Neurito: experimentos y respuestas

**Configuración base:** red con 3 entradas (hambre, energía, aburrimiento), 6 neuronas ocultas con ReLU y 3 salidas con Softmax (comer, dormir, jugar). Entrenada con 3000 ejemplos sintéticos, 1200 épocas y tasa de aprendizaje 0.1.
**Resultado base:** pérdida = 0.138, precisión de prueba = 99.40%.

---

## Experimentos

### 1. Dos necesidades constantes, variar la tercera

Hambre = 0.5 y aburrimiento = 0.5 fijos; se varía la energía.

| Energía | Cansancio (1 - energía) | Decisión   | Comer / Dormir / Jugar |
| ------- | ----------------------- | ---------- | ---------------------- |
| 0.90    | 0.10                    | comer      | 0.514 / 0.000 / 0.486  |
| 0.70    | 0.30                    | comer      | 0.516 / 0.011 / 0.473  |
| 0.55    | 0.45                    | comer      | 0.441 / 0.164 / 0.395  |
| 0.50    | 0.50                    | comer      | 0.352 / 0.334 / 0.313  |
| 0.45    | 0.55                    | **dormir** | 0.232 / 0.563 / 0.205  |
| 0.30    | 0.70                    | dormir     | 0.023 / 0.956 / 0.020  |
| 0.10    | 0.90                    | dormir     | 0.001 / 0.999 / 0.000  |

La acción cambia de "comer" a "dormir" entre energía 0.50 y 0.45, es decir, cuando el cansancio supera 0.5, el valor de las otras dos necesidades. La probabilidad de "dormir" sube de forma gradual (0.011, 0.164, 0.334, 0.563, 0.956, 0.999), por lo que la red aprendió una frontera suave y no un salto brusco. Con energía alta, "comer" y "jugar" quedan casi empatadas (0.514 contra 0.486), porque hambre y aburrimiento valen lo mismo.

### 2. Prioridades casi empatadas

| Caso                            | Decisión | Comer / Dormir / Jugar |
| ------------------------------- | -------- | ---------------------- |
| hambre 0.80 / cansancio 0.80    | dormir   | 0.486 / 0.514 / 0.000  |
| hambre 0.80 / cansancio 0.81    | dormir   | 0.439 / 0.561 / 0.000  |
| hambre 0.90 / cansancio 0.20    | comer    | 1.000 / 0.000 / 0.000  |
| triple empate (0.5 / 0.5 / 0.5) | comer    | 0.352 / 0.334 / 0.313  |

Con prioridades casi empatadas, Softmax reparte la probabilidad entre las dos candidatas (0.486 contra 0.514). Con una diferencia clara (0.90 contra 0.20), una acción domina con probabilidad cercana a 1. Una diferencia de apenas 0.01 en las necesidades mueve "dormir" de 0.514 a 0.561, así que la salida es sensible cerca de la frontera, pero no salta de 0 a 1. Una probabilidad de 0.51 y una de 1.00 producen la misma acción, pero expresan confianzas muy distintas.

### 3. Cambiar las neuronas ocultas

| Neuronas ocultas | Pérdida | Precisión |
| ---------------- | ------- | --------- |
| 3                | 0.176   | 96.30%    |
| 6                | 0.138   | 99.40%    |
| 10               | 0.141   | 99.10%    |
| 20               | 0.138   | 99.00%    |

Con 3 neuronas la red tiene menos capacidad: más pérdida y menos precisión. Con 6 se alcanza el mejor resultado. Con 10 y 20 no mejora, y las diferencias entre 6, 10 y 20 son de 3 o 4 casos de 1000, es decir, ruido. Para un problema tan simple, 6 neuronas bastan, y más neuronas solo agregan cálculo.

### 4. Cambiar épocas o tasa de aprendizaje

**Épocas (lr = 0.1):**

| Épocas | Pérdida | Precisión |
| ------ | ------- | --------- |
| 100    | 0.819   | 75.10%    |
| 400    | 0.281   | 97.50%    |
| 1200   | 0.138   | 99.40%    |
| 5000   | 0.065   | 99.90%    |

El entrenamiento mejora de forma continua, pero con rendimientos decrecientes. Con 100 épocas hay subajuste. Incluso con 5000 sigue mejorando, solo que más despacio.

**Tasa de aprendizaje (1200 épocas):**

| lr    | Pérdida | Precisión | Veces que la pérdida subió |
| ----- | ------- | --------- | -------------------------- |
| 0.001 | 1.040   | 41.70%    | 0                          |
| 0.01  | 0.752   | 80.10%    | 0                          |
| 0.1   | 0.138   | 99.40%    | 0                          |
| 1     | 0.059   | 98.40%    | 545                        |
| 10    | 3.082   | 33.30%    | 797                        |

- **lr muy baja (0.001 y 0.01):** el entrenamiento es estable pero demasiado lento y no llega a converger en 1200 épocas.
- **lr = 0.1:** estable y con muy buen resultado.
- **lr = 1:** la pérdida final es la más baja, pero la pérdida subió 545 veces durante el entrenamiento y la precisión es menor. Es un entrenamiento inestable, que oscila.
- **lr = 10:** diverge. La precisión (33.3%) equivale a adivinar entre 3 clases.

### 5. ¿Qué datos y etiquetas requeriría una cuarta acción?

Por ejemplo, "bañarse" o "abrazo". Haría falta:

- Una **nueva entrada** que represente su necesidad (por ejemplo `suciedad` o `soledad`), de modo que la entrada pasa de 3 a 4 valores.
- Una **neurona de salida extra**: `W2` pasa de (6, 3) a (6, 4) y `b2` de (1, 3) a (1, 4). `W1` también cambia a (4, 6).
- **Datos etiquetados** donde esa nueva necesidad sea la más alta, con la etiqueta de la nueva acción, y en cantidad balanceada con las otras tres acciones.
- **Reentrenar** la red desde cero con los nuevos datos.
- Nuevos **sprites** para mostrar la acción en pantalla.

---

## Explica tus resultados

### 1. ¿En qué se diferencian los pesos de los sesgos?

Los **pesos** (`W1`, `W2`) multiplican cada entrada y definen cuánta influencia tiene cada necesidad sobre cada neurona. Los **sesgos** (`b1`, `b2`) se suman al final y desplazan el umbral de activación de la neurona, de modo que esta puede activarse aunque todas las entradas valgan 0. Los pesos controlan la importancia de cada entrada y el sesgo controla cuándo se activa la neurona.

### 2. ¿Por qué el cansancio es `1 - energy`?

Porque energía y cansancio son opuestos: energía alta significa poco cansancio. La política que genera las etiquetas es "gana la necesidad más alta", así que las tres cantidades deben medirse igual (más valor = más necesidad). Usar `1 - energía` convierte la energía en cansancio y permite compararla con el hambre y el aburrimiento. En este proyecto la red recibe la energía directa y aprende esa inversión a partir de las etiquetas.

### 3. ¿Qué hacen ReLU y Softmax cada uno?

- **ReLU** (`max(0, z)`) está en la capa oculta. Deja pasar los valores positivos y convierte los negativos en 0. Aporta no linealidad, lo que permite aprender fronteras de decisión que una capa lineal sola no podría.
- **Softmax** está en la capa de salida. Convierte los tres valores crudos en probabilidades que suman 1. La mayor indica la acción elegida y además expresa qué tan segura está la red.

### 4. ¿Por qué la alta precisión puede ser engañosa?

- La precisión global fue 99.4%, pero los 6 errores de 1000 están en los **casi-empates**. En casos claros (brecha >= 0.05) la precisión es 100%, y en los 160 casi-empates baja a 96.25%. El promedio esconde dónde falla la red.
- La precisión no mide la **confianza**: acertar con 34% de probabilidad cuenta igual que acertar con 99%.
- Los datos de prueba salen de la misma regla sintética con la que se entrenó, así que la precisión es "frente a la política sintética", no frente al mundo real.
- La pantalla "neutral" de la aplicación usa una regla aparte que la red no entrenó, y esa parte no entra en la precisión.
- En otros problemas, con clases desbalanceadas, un modelo que siempre responda la clase mayoritaria tendría alta precisión sin aprender nada. Aquí las clases están balanceadas (comer 100%, dormir 99.1%, jugar 99.1%), pero conviene revisar la precisión por clase.
