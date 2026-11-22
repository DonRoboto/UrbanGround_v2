# Protocolo experimental (Parte B)

## Pregunta
¿Es posible identificar **evidencia visible** de comercio activo en la planta baja de frentes urbanos de Ciudad de México mediante SigLIP 2? No se pretende confirmar estado real de operación, horarios ni cumplimiento normativo.

## Unidad y etiquetas
Cada **fotografía del frente objetivo** es una observación. `A` = evidencia de atención/venta o establecimiento abierto/preparado; `I` = hay señales de establecimiento pero la operación no es verificable; `N` = no hay comercio visible en el frente **objetivo**. F02 y F13 son perspectivas posiblemente repetidas de una misma unidad comercial. La clase N tiene un solo ejemplo (F11).

Las etiquetas se revisaron iterativamente tras conocer algunas salidas del modelo. No constituyen anotación ciega. La anotación correcta para evidencia ambigua no es necesariamente «cerrado». Si hay vivienda en pisos superiores y comercio activo en planta baja, se marca A.

## Diseño y controles

| ID | Qué cambia | Qué permanece fijo | Salidas |
|---|---|---|---|
| E0 | Imagen completa vs ROI; banco genérico vs contextual | Checkpoint, etiquetas, regla mean | 52 predicciones; 780 logits |
| E1 | Media vs mediana/Top-2/max/log-mean-exp | E0 logits | 260 predicciones recalculadas |
| E2 | Recorte, blur superior, blur interior y blur exterior ROI | Checkpoint y bancos E0 | 130 predicciones, 1950 logits |
| E3 | Clasificación jerárquica: presencia luego actividad | Checkpoint e imágenes/ROI | 26 predicciones, 520 logits |
| E4 | **Solo** textos de operación de E3 | Misma etapa 1, checkpoint, mean, ROI | 26 predicciones, 520 logits archivados/nuevos |

No se entrenan pesos ni se usa una muestra externa. Los resultados son descriptivos del conjunto de desarrollo. Si se obtienen fotografías nuevas, anotarlas **sin acceso a predicciones**, definir el protocolo antes de verlas y separarlas por establecimiento (evitando vistas duplicadas).

## Hipótesis y mecanismo de fallo

**H1 — Contexto visual:** la evidencia comercial pequeña queda mezclada con árboles, autos, viviendas y pisos superiores. Recortar ROI cambia a la vez escala relativa y contexto. Blur exterior conserva escala pero altera la distribución visual. Estas intervenciones **no** demuestran atención causal a un objeto específico.

**H2 — Representación textual:** promediar prompts heterogéneos (puestos, restaurantes, talleres, servicios) puede ocultar una descripción pertinente. El análisis de logits individuales y agregadores es **post hoc**, no un ajuste validado para generalización.

**H3 — Incertidumbre operativa:** identificar un establecimiento no implica observarlo en operación. Los prompts que expresan ausencia de clientes se superponen semánticamente con negocios abiertos fotografiados sin personas. E3 y E4 contrastan estas hipótesis sin inferir estados reales.

## Protocolo del diagnóstico por fotografía
Registrar: imagen/F#, frente objetivo, etiqueta humana, evidencia visible, clase predicha, los dos mejores textos por hipótesis, cambio de ROI, efecto de intervenciones, margen por etapa y explicación condicional («compatible con...», no «el modelo atendió tal píxel»).

## Métricas
Accuracy, macro-F1, recall A/I/N, matrices de confusión con filas humanas/columnas predichas. En E3/E4, informar por separado etapa 1 y etapa 2; nunca comparar logits de etapas diferentes como probabilidades. En una muestra con N=1, una tasa de acierto N=100 % se basa en un único ejemplo.

## Reproducibilidad y límites
Manifiestos con checkpoint/model commit (si disponible), versiones, SHA256 de imágenes y archivos de configuración, prompt exacto y semilla. El modelo se ejecuta en inferencia congelada. Un cambio de ROI/prompt después de conocer los casos es exploratorio. Para afirmar sesgo regional se necesitaría un estudio comparativo de ciudades con condiciones de captura, tipologías y etiquetas equivalentes, que **no** forma parte de estas pruebas.
