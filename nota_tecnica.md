# UrbanGroundFloor — Nota técnica | Parte B

**Pregunta de estudio.** ¿Existe evidencia visible de actividad comercial activa en la planta baja del **frente objetivo** de una fotografía urbana?  
**Material.** Trece capturas de Google Street View de una colonia popular de Ciudad de México, de 640 × 640 px, proporcionadas sin etiquetas.  
**Sistema.** `google/siglip2-base-patch16-224` (revisión congelada en `configs/experiment_pinned.yaml`), sin ajuste de pesos (*zero-shot*).  
**Fecha de cierre.** 21 de noviembre de 2026. **Estado de la evidencia:** estudio exploratorio sobre el conjunto utilizado para desarrollar el método.

## 1. Decisión de diseño y pregunta operacional

La pregunta no se resuelve con la dicotomía «tienda/no tienda». Una cortina cerrada puede identificar inequívocamente un local sin revelar si presta servicio; una venta informal puede funcionar sin letrero, personal visible ni vitrina. Por ello se emplearon tres **etiquetas de observabilidad**, no de situación jurídica o permanencia del negocio:

| Clase | Regla de decisión | Evidencia suficiente / insuficiente |
|:--|:--|:--|
| **A — actividad comercial visible** | El frente muestra un comercio abierto o preparado para vender/prestar servicios. | Acceso abierto, productos accesibles, mostrador, puesto instalado, mesas preparadas o herramientas disponibles; **no se exige observar una transacción**. |
| **I — operación indeterminada** | Hay indicios de establecimiento comercial, pero la operación **no se puede verificar en esta fotografía**. | Rótulo y cortina baja, acceso cubierto, ventanilla sin atención comprobable. **I no significa «cerrado definitivamente»**. |
| **N — sin evidencia comercial** | El frente objetivo no muestra señales visuales suficientes de un establecimiento. | Acceso habitacional o cochera sin indicios de venta. **N no demuestra ausencia de negocio oculto**. |

**Unidad de anotación:** el frente previamente seleccionado, no todos los locales que aparecen en la imagen. En inmuebles de uso mixto se evalúa exclusivamente la planta baja; la vivienda en pisos superiores no convierte A en N. Cuando hay varios frentes, solo se considera el objetivo: en F11, los comercios parciales de los extremos no se asignan a la fachada habitacional central. F02 y F13 parecen mostrar dos perspectivas del mismo negocio y se mantienen identificadas como grupo dependiente.

**Procedimiento de etiquetado.** (1) Definir el frente objetivo y conservar la imagen original; (2) delimitar su ROI antes de mirar puntuaciones en evaluaciones nuevas; (3) registrar A/I/N y un hecho visible verificable; (4) asignar I si el carácter comercial está identificado pero la operación no es observable; (5) registrar uso mixto, oclusión y posibles duplicados; (6) revisar dudas por segunda persona cuando sea posible. Las etiquetas de este estudio fueron **confirmadas mediante revisión iterativa**, no mediante anotación ciega independiente: una limitación que se declara explícitamente.

Las 13 decisiones humanas y su justificación están en `anotaciones.csv` y `data/annotations.csv` (**7 A, 5 I, 1 N**).

## 2. Por qué estos textos y cuáles se descartaron

**Razonamiento de diseño.** Partimos de distinguir tres hechos que una imagen estática no permite equiparar: **presencia de establecimiento**, **evidencia visible de operación** y **ausencia de indicios en el frente objetivo**. Por eso A no exige observar una compra, I conserva la incertidumbre cuando el comercio está identificado pero no puede verificarse la atención, y N se restringe a la evidencia visible del frente, sin inferir ausencia permanente de actividad. A continuación construimos **dos bancos de cinco descripciones por clase** (15 textos por banco): uno genérico para representar estas reglas sin detalles de contexto y otro contextual para probar si expresar rasgos comerciales locales cambia la compatibilidad imagen–texto. Los textos completos se conservan en `configs/prompts_generic.json` y `configs/prompts_contextual.json`.

**Banco genérico.** Describe establecimientos abiertos, negocios con estado no verificable y fachadas habitacionales sin comercio. Ejemplos originales utilizados:

- **A:** «Fotografía de una tienda de barrio abierta y preparada para vender».
- **I:** «Fotografía de un negocio identificado por su fachada pero con estado de operación desconocido».
- **N:** «Fotografía de una vivienda sin negocio visible en el frente principal».

**Banco contextual.** Hace explícitas señales habituales en los frentes estudiados: puesto en banqueta, mercancía expuesta, fonda con mesas, taller de servicios, mostrador, cortina metálica y acceso cubierto. Ejemplos originales:

- **A:** «Hay un puesto de venta instalado a nivel de banqueta, con productos listos para vender».
- **I:** «Hay un local con nombre o anuncio comercial, pero la cortina metálica está cerrada».
- **N:** «El frente de una vivienda muestra un portón de acceso privado sin mercancías ni anuncios».

**Señales incorporadas y descartadas.** Se incorporaron accesos abiertos, mercancía expuesta desde la vivienda, mostradores, puestos de banqueta, mesas preparadas, talleres y servicios, rótulos, cortinas metálicas y oclusiones: reflejan formas de comercio mixto o poco señalizado que una descripción estereotípica de «tienda» puede omitir. No se exigieron «personas comprando», «caja registradora», vitrinas modernas ni «letrero formal» para asignar A, porque excluirían ventas informales y comercios abiertos sin clientes visibles. Tampoco se interpretó «cortina cerrada» como N o como cierre permanente; ni la vivienda superior como ausencia de actividad en planta baja. Se descartó inferir horarios, permisos, transacciones efectivas o el estado operativo actual a partir de una sola captura.

El banco contextual contiene subtipos heterogéneos (restaurante, taller, tienda, puesto): su promedio puede **diluir una correspondencia correcta con un subtipo**. La comparación entre bancos examina qué señales lingüísticas y visuales están representadas en la similitud imagen–texto; **no identifica causalmente qué píxeles utiliza la red**.

## 3. Inferencia y diseño experimental

Se ejecutó SigLIP 2 con pesos congelados sobre dos vistas de cada fotografía: **imagen completa** y **ROI manual rectangular**. Las ROI (`data/rois.csv`) se definieron previamente para la planta baja objetivo; al procesarse a la resolución del checkpoint, el recorte cambia **simultáneamente el contexto visible y la escala relativa del frente**. Por tanto, no se interpreta como un experimento causal que aísle una sola variable.

Para una imagen o ROI $x$, banco de prompts $T$ y clase $c\in\{A,I,N\}$, la puntuación fue el promedio de los cinco logits de compatibilidad imagen–texto:

$$
S_c(x)=\frac{1}{5}\sum_{j=1}^{5}\ell_{\mathrm{SigLIP2}}(x,t_{c,j}),\qquad
\widehat y(x)=\operatorname*{arg\,max}_{c\in\{A,I,N\}}S_c(x).
$$

Los logits son puntuaciones relativas **no probabilidades calibradas de operación**. Se conservaron **52 decisiones** ($13\times2$ regiones $\times2$ bancos) y **780 logits** ($52\times15$ prompts). Los parámetros, rutas, revisión de modelo y hashes se registran en los archivos de configuración y manifiestos de las corridas.

## 4. Resultado principal: formulación textual y encuadre

| Imagen | Banco | Aciertos | Accuracy | Macro-F1 | Recall A | Recall I | Recall N |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Completa | Genérico | 2/13 | 0.154 | 0.135 | 0.143 | 0.000 | 1.000 |
| Completa | Contextual | 1/13 | 0.077 | 0.051 | 0.000 | 0.000 | 1.000 |
| **ROI manual** | **Genérico** | **10/13** | **0.769** | **0.704** | **1.000** | **0.400** | **1.000** |
| ROI manual | Contextual | 8/13 | 0.615 | 0.627 | 0.429 | 0.800 | 1.000 |

Con imagen completa predomina la decisión **N**, incluso cuando la anotación identifica un negocio. Concentrar la entrada en el frente objetivo eleva la coincidencia de **2/13 a 10/13** con el banco genérico. No se puede atribuir toda esta diferencia a «eliminar contexto»: también cambia el tamaño relativo del negocio y las ROI se diseñaron a partir del mismo conjunto.

**Efecto aislado de cambiar las descripciones.** Al mantener **idénticas las 13 ROI, las imágenes, el checkpoint y la media de cinco logits por clase** y sustituir solamente el banco genérico por el contextual, **F05 y F09** pasan de A (error) a I (acierto): **dos errores corregidos** al describir explícitamente cortinas cerradas y operación no observable. Sin embargo, **F03, F06, F08 y F12** pasan de A (acierto) a I (error): **cuatro errores nuevos**. Las otras **siete** decisiones no cambian. Así, la exactitud pasa de **10/13 a 8/13**, el recall de I sube de **2/5 a 4/5**, pero el de A baja de **7/7 a 3/7**. El banco contextual mejora la coincidencia con estados visualmente indeterminados, pero su promedio combina subtipos heterogéneos —fonda, taller, puesto, tienda— que pueden penalizar a un comercio activo concreto. Por tanto, **más especificidad textual no implica una mejor decisión agregada**. La evidencia respalda sensibilidad a las descripciones y su agregación; no permite identificar causalmente los píxeles atendidos por el modelo.

En ROI + genéricos, la matriz de confusión tiene A: 7→A; I: 2→A, 2→I, 1→N; N: 1→N. La clase N tiene **solo un caso**: recall N = 1.000 no permite afirmar robustez en fachadas no comerciales.

## 5. Diagnóstico de los desacuerdos: qué información parece utilizar el modelo

La tabla documenta las decisiones de las **13 fotografías**. En las explicaciones, «completa» y «ROI» siempre se refieren al **frente objetivo**, y los mecanismos son hipótesis compatibles con las puntuaciones, **no atribuciones causales de atención neuronal**.

| Foto | Humana | Completa gen. | ROI gen. | ROI context. |
|:--|:--:|:--:|:--:|:--:|
| F01 | A | A | A | A |
| F02 | I | N | I | I |
| F03 | A | N | A | I |
| F04 | A | N | A | A |
| F05 | I | N | **A** | I |
| F06 | A | N | A | I |
| F07 | I | N | **N** | **N** |
| F08 | A | N | A | I |
| F09 | I | N | **A** | I |
| F10 | A | N | A | A |
| F11 | N | N | N | N |
| F12 | A | N | A | I |
| F13 | I | N | I | I |

**F01 (A; contextual/completa→I).** La anotación registra vitrina y productos. El cambio de banco introduce textos de estados no verificables que pueden favorecer I cuando el frente permanece pequeño en la vista completa; el banco genérico y ambas ROI aciertan.

**F02 (I; completa→N).** Se identifica una cafetería con acceso cubierto; N gana con el contexto completo, pero la ROI recupera las señales de establecimiento y la clase I. Esta vista parece corresponder al mismo comercio que F13.

**F03 (A; completa→N; contextual/ROI→I).** El negocio pequeño con exhibidores está a distancia. El recorte genérico recupera A, pero las descripciones contextualizadas de subtipos agregadas dan ventaja a I.

**F04 (A; completa→N).** La anotación identifica toldo azul, entrada abierta y exhibidores. La decisión pasa a A con ambos bancos al recortar el frente, compatible con mayor peso relativo de esas señales visuales.

**F05 (I; completa→N; genérico/ROI→A).** Rótulos con cortina principal cerrada. La descripción genérica puede confundir la presencia del establecimiento con operación; el banco contextual reconoce correctamente la incertidumbre.

**F06 (A; completa→N; contextual/ROI→I).** Tienda de esquina con acceso aparentemente abierto y objetos visibles. La media del banco contextual favorece I, compatible con la penalización de prompts de subtipos no aplicables.

**F07 (I; completa y ROI→N).** Cocina económica con menú y ventanilla, sin atención verificable. Las señales no producen similitud suficiente con «establecimiento comercial», aun después del recorte.

**F08 (A; completa→N; contextual/ROI→I).** Restaurante con mesas y acceso: un prompt específico de A coincide bien, pero al promediarlo con otros subtipos, la media de I termina siendo superior.

**F09 (I; completa→N; genérico/ROI→A).** Minisúper anunciado con cortina cerrada. El banco genérico reconoce comercio y extrapola indebidamente actividad; el contextual sí distingue la operación no verificable.

**F10 (A; completa→N).** Puesto con sombrilla y productos. La descripción de puesto informal y el recorte favorecen A aunque no exista necesariamente una fachada comercial formal.

**F11 (N; sin desacuerdo en E0).** El frente objetivo es el inmueble habitacional central; los comercios parciales de los extremos pertenecen a otros frentes. Sirve como control de la definición de unidad de anotación, no como prueba de sensibilidad general a N.

**F12 (A; completa→N; contextual/ROI→I).** Cerrajería con artículos/herramientas visibles. La mezcla de textos de restaurantes, tiendas y talleres reduce la media A pese a la evidencia de servicio abierto.

**F13 (I; completa→N).** Segunda perspectiva de la cafetería de F02; recortar permite reconocer el establecimiento sin poder verificar operación. No se debe contar como negocio estadísticamente independiente.

Los errores residuales de **ROI genérica** merecen especial atención:

1. **F05, I→A.** $S_A=-0.450$, $S_I=-0.727$: ventaja pequeña de A (**0.277**). El concepto «local comercial» parece sustituir al requisito más estricto de acceso operativo; el banco contextual y E3 sí seleccionan I.
2. **F07, I→N.** $S_N=-0.722$, $S_A=-1.861$, $S_I=-2.891$: ventaja de N sobre el segundo lugar de **1.139**. Menú y ventanilla no producen suficiente similitud agregada con negocio. Es un ejemplo de **subdetección de comercio discreto**, no prueba de ausencia real.
3. **F09, I→A.** $S_A=-1.177$, $S_I=-1.513$: margen **0.336**. El rótulo de minisúper concuerda con comercio, pero no verifica que el frente esté atendiendo; el banco contextual y E3 distinguen correctamente ambos conceptos.

**Mecanismo de agregación observable (F08).** Con ROI y banco contextual, el mejor prompt individual de A alcanza **+1.130**, frente a **+0.397** para el mejor I; sin embargo, las **medias de clase** son $S_A=-2.645$ y $S_I=-2.111$, por lo que gana I. La clasificación depende tanto de *qué subtipo describe cada prompt* como de cómo se agregan los logits; no sería correcto afirmar que el modelo «no reconoce» ninguna señal de restaurante. En F12, el prompt de restaurante/fonda tiene logit **−8.005** y perjudica la media de A para una cerrajería.

## 6. Experimentos complementarios y propuesta de ajuste

| Experimento | Observación verificable | Interpretación acotada |
|:--|:--|:--|
| **E1 — Agregación** | Se evaluaron media, mediana, Top-2, máximo y log-mean-exp sobre los mismos 780 logits. | Sensibilidad a la regla de combinación; **selección post hoc**, no mejora validada. |
| **E2 — Intervenciones espaciales** | Código para imagen completa, ROI y desenfoque en distintas regiones. | Herramienta de diagnóstico; aquí no se afirma un efecto numérico no respaldado por una corrida consolidada. |
| **E3 — Jerarquía presencia→operación** | En ROI **9/13**, macro-F1 **0.778**; presencia comercial frente a N: **13/13** sobre el mismo conjunto. | Separar dos preguntas mejora el recall de I, pero pierde A; tampoco valida generalización. |
| **E4 — Refinamiento de operación** | En ROI **9/13**, macro-F1 **0.771**. F07 se corrige de A a I respecto de E3 y **F08 pasa de A a I por error**. | Un cambio de prompts puede compensar errores sin aumentar aciertos globales. |
| **E5 — ROI automática** | Con banco genérico: manual **10/13** (F1 **0.704**), OWLv2 **9/13** (F1 **0.610**), OpenCV **7/13** (F1 **0.378**). | Localización imperfecta introduce errores; IoU medio contra ROI manual: OWLv2 **0.360**, OpenCV **0.285**. |

El ajuste operativo que se propone es **localización explícita del frente + reconocimiento de establecimiento + evaluación separada de indicios de operación**, con una salida **I** cuando el estado no sea verificable. Para fotografías nuevas, OWLv2 puede sugerir una ROI sometida a **validación humana**; si hay un frente definido, debe confirmarse que corresponde al inmueble objetivo. La evidencia de E3/E4 justifica investigar esta descomposición, **no proclamarla como superior a ROI genérica**: esta última sigue siendo el mejor método exploratorio en *accuracy* sobre las 13 imágenes.

En E5, F05 y F12 tienen IoU **0** con OWLv2 frente a sus ROI manuales, mientras F11 muestra que una ROI estructural equivocada puede convertir N→A. Se separan así **error de localización** y **error de interpretación**: F07 y F09 siguen fallando con ROI manual, lo que evidencia que no todo desacuerdo se resuelve encontrando mejor el frente.

## 7. Límites y reproducibilidad

1. **Muestra pequeña y sesgada por diseño:** 13 fotografías, clases 7/5/1, con posible dependencia entre F02 y F13. Las ROI y los prompts se ajustaron exploratoriamente sobre estas mismas imágenes. Las cifras no estiman generalización ni permiten atribuir un sesgo regional al preentrenamiento sin dominios de control.
2. **Observabilidad restringida:** «A» se refiere a evidencia visual de un frente dispuesto para comercio, no a comprobar que se produjo una venta ni a saber si está abierto ahora. «I» no es predicción de cierre.
3. **Mecanismos inferidos, no comprobados causalmente:** logits y cambios de encuadre/texto sustentan hipótesis de sensibilidad al contexto y a las descripciones, no mapas de atención causales.
4. **Modelo congelado y trazabilidad:** configuración de revisión en `configs/experiment_pinned.yaml`, textos completos en `configs/`, anotaciones y ROI en `data/`, resultados históricos y comparaciones E5 en `reference/`. Las fotografías originales deben copiarse **sin alterar** a `data/images/`; no se incluyen aquí por una decisión de gestión de derechos, que debe resolverse antes de publicar el repositorio definitivo.

**Reproducción mínima desde la raíz del repositorio (Python ≥3.10):**

```bash
python -m pip install -r requirements.txt
python -m pytest -q
bash scripts/run_offline_audit.sh runs/auditoria --plots  # reutiliza CSV reales, sin GPU
# Con las 13 imágenes originales en data/images/ y PyTorch CUDA compatible:
python -m urban_vlm validate --config configs/experiment_pinned.yaml
python -m urban_vlm run --config configs/experiment_pinned.yaml \
    --device auto --output runs/replica_e0
```

La nota se basa en los CSV conservados: **no se ha repetido hoy la inferencia sobre las fotografías originales**. El paquete permite verificar los datos históricos en CPU; la reproducción completa desde cero exige disponer de los originales y descargar los pesos del checkpoint. Las diferencias entre hardware y versiones pueden alterar los logits, por lo que no se promete identidad bit a bit.

**Conclusión.** En las 13 fachadas analizadas, el encuadre del frente objetivo y las palabras que distinguen *establecimiento comercial* de *operación observable* modifican decisivamente las decisiones de SigLIP 2. El procedimiento propuesto combina una delimitación explícita del frente, categorías capaces de representar incertidumbre y un diagnóstico verificable por fotografía; requiere validación externa antes de utilizarse a escala urbana.
