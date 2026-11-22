# E5 · Localización automática de frentes como etapa principal de UrbanGroundFloor

## Pregunta y alcance

**Pregunta:** ¿Se conserva la mejora de SigLIP 2 que observamos al utilizar ROI manuales cuando el recorte de planta baja se obtiene automáticamente? La tarea se descompone en **localizar el frente objetivo** y **clasificar la evidencia observable A/I/N**. No debe confundirse una detección de fachada con la afirmación de que existe un negocio abierto.

Las 13 imágenes de 640×640 píxeles constituyen el **conjunto de desarrollo**. Las ROI manuales fueron delimitadas en la exploración previa y se usan como referencia espacial para IoU, no como entrada de OpenCV ni OWLv2. Se conservan las etiquetas humanas, el modelo SigLIP 2, los dos bancos de prompts y la regla de puntuación (`mean` de cinco logits por clase).

## Métodos automáticos implementados

**OpenCV (estructural, sin aprendizaje adicional):** escala para el análisis, suaviza, calcula bordes Canny, gradientes Sobel y saturación; estima una franja plausible a nivel de calle; puntúa ventanas de búsqueda por su estructura interna, tamaño y distancia a un prior espacial del frente objetivo. Puede confundirse con autos, ventanas o comercios vecinos porque no reconoce el significado semántico de un frente.

**OWLv2 (detección zero-shot):** utiliza `google/owlv2-base-patch16-ensemble` y consultas sobre entradas de tienda, comercios pequeños, restaurantes, puestos, talleres y cortinas metálicas cerradas. Filtra detecciones por tamaño, posición y puntuación, las amplía a una ROI del frente y usa OpenCV como respaldo cuando no hay una detección elegible. Sus puntuaciones no constituyen probabilidades calibradas de actividad. Puede favorecer indicios comerciales formales o seleccionar el negocio equivocado.

**Manual:** ROI delimitada por un anotador, sin inferencia de clase; referencia exploratoria, no un estándar geográfico universal.

## Dos objetivos de evaluación

1. **Localización:** IoU entre ROI automática y manual: `intersection_area / union_area`. No es una métrica de detección de actividad comercial.
2. **Clasificación:** accuracy, macro-F1 y recall A/I/N de SigLIP 2 con cada ROI, manteniendo constantes prompts y checkpoint. Las decisiones se comparan por fotografía (no solo sus promedios).

### Localización

| Indicador | OpenCV | OWLv2 |
|---|---:|---:|
| IoU promedio | 0.285 | **0.360** |
| IoU mediana | 0.271 | **0.426** |
| IoU ≥ 0.50 | 2/13 | **5/13** |
| IoU = 0 | 1/13 | 2/13 |
| Marca `manual_review` | 9/13 | 6/13 |

OWLv2 logra mayor IoU en 9 de las 13 fotografías. Esto no implica que cada una de sus ROI represente el frente solicitado: F05 y F12 tienen IoU cero.

### Clasificación

| ROI | Banco de prompts | Aciertos | Accuracy | Macro-F1 | Recall A | Recall I | Recall N |
|---|---|---:|---:|---:|---:|---:|---:|
| Manual | Genérico | **10/13** | **0.769** | **0.704** | 7/7 | 2/5 | 1/1 |
| OpenCV | Genérico | 7/13 | 0.538 | 0.378 | 6/7 | 1/5 | 0/1 |
| OWLv2 | Genérico | **9/13** | **0.692** | **0.610** | 6/7 | 2/5 | 1/1 |
| Manual | Contextual | 8/13 | 0.615 | 0.627 | 3/7 | 4/5 | 1/1 |
| OpenCV | Contextual | 8/13 | 0.615 | 0.556 | 5/7 | 2/5 | 1/1 |
| OWLv2 | Contextual | 7/13 | 0.538 | 0.515 | 3/7 | 3/5 | 1/1 |

Todas las cifras proceden de los CSV de la corrida E5 y describen el mismo conjunto de desarrollo; no debe presentarse como validación generalizable ni optimización de hiperparámetros.

### Diagnóstico cualitativo con prompts genéricos

| Imagen | Anotación | Manual / CV / OWLv2 | IoU CV / OWLv2 | Interpretación |
|---|---|---|---|---|
| F05 | I | A / N / N | 0.264 / 0.000 | OWLv2 localiza una región no correspondiente al local objetivo. |
| F06 | A | A / N / A | 0.000 / 0.193 | OpenCV pierde el objetivo; OWLv2 acierta aunque su IoU sea bajo. |
| F07 | I | N / N / N | 0.533 / 0.324 | Error de interpretación persiste incluso con ROI razonables. |
| F09 | I | A / A / A | 0.190 / 0.061 | El establecimiento se confunde con operación observable. |
| F11 | N | N / A / N | 0.271 / 0.426 | OpenCV incorpora señales ajenas al frente residencial. |
| F12 | A | A / A / N | 0.595 / 0.000 | OWLv2 no recupera la cerrajería objetivo. |
| F13 | I | I / N / I | 0.140 / 0.571 | Mejor localización de OWLv2 y recuperación del estado I. |

La interpretación debe combinar **errores de localización** y **errores semánticos**. Un IoU alto no determina la clase correcta (F07), y un IoU bajo puede retener la evidencia suficiente para acertar (F06).

## Cómo reproducir

1. Instalar dependencias y colocar las 13 imágenes originales sin modificaciones en `data/images/`; conservar `data/annotations.csv` y `data/rois.csv`.
2. Generar ROI automáticas:

```bash
python -m urban_vlm auto-rois --config configs/experiment.yaml \
  --method opencv --output runs/E5_opencv_01
python -m urban_vlm auto-rois --config configs/experiment.yaml \
  --method owlv2 --output runs/E5_owlv2_01
```

3. Validar entradas y ejecutar clasificación pareada con el checkpoint fijado:

```bash
python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv --dry-run

python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv \
  --device auto --output runs/E5_siglip_replicacion
```

4. Revisar `runs/E5_siglip_replicacion/comparison/metrics_summary.csv`, `paired_predictions.csv`, `changes_vs_manual.csv`, `roi_iou.csv` y `comparison_manifest.json` junto con las imágenes `roi_review.png`.

**Condiciones de validez:** los métodos y prompts fueron explorados sobre las 13 fotos; solo existe un caso N y F02/F13 pueden ser vistas del mismo negocio. La ROI manual no es un ground truth exhaustivo de parcela. OWLv2 incorpora pistas semánticas comerciales en su localizador. La revisión humana de ROI se propone para operación práctica, pero no fue aplicada para alterar las predicciones de E5 aquí reportadas.
