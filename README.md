# reproducir todos los experimentos y verificar los resultados

**Inicio rápido:** este repositorio contiene un lanzador que ejecuta E0–E5 sobre las 13 fotografías originales, compara cada predicción con los resultados archivados y entrega un informe de concordancia. 

```bash
python3 -m venv .venv && source .venv/bin/activate
# Si usas GPU, instala previamente PyTorch compatible con CUDA.
python -m pip install -r requirements-reproduction.txt
python scripts/check_entrega.py --require-images
CONFIG=configs/experiment_pinned.yaml ROI_SOURCE=regenerate DEVICE=auto \
bash scripts/reproducir_todos_los_experimentos.sh runs/revisor_01
```

**Resultado principal para comprobar las cifras:** `runs/revisor_01/VERIFICATION/reproduction_report.md`. Los fallos se documentan en `VERIFICATION/differences.csv` y hacen fallar el comando. **Sin los 13 originales exactos en `data/images/` no puede realizarse la reproducción completa**. Los archivos pueden necesitar una entrega privada autorizada.

**Guía detallada:** [`docs/REVIEWER_REPRODUCCION_INTEGRAL.md`](docs/REVIEWER_REPRODUCCION_INTEGRAL.md). Para separar diferencias en detección de ROI de diferencias de SigLIP 2, utilizar `ROI_SOURCE=historical`; este segundo modo no regenera las ROI.

---

# ENTREGA — Parte B: UrbanGroundFloor (cierre 2026-11-21)

**Entregable central:** [`nota_tecnica.pdf`](nota_tecnica.pdf) (**4 páginas; incluye el experimento de ROI automática como eje principal**) y su [fuente LaTeX](nota_tecnica.tex). Responde directamente por qué se formularon las clases y descripciones, qué señales del comercio en CDMX se incorporaron o descartaron y qué reveló cambiar los prompts. Incluye el diagnóstico de F01–F13. [`nota_tecnica.md`](nota_tecnica.md) se conserva como **versión ampliada de apoyo** con mayor detalle experimental.

**Etiquetas solicitadas:** [`anotaciones.csv`](anotaciones.csv), idénticas a `data/annotations.csv`; protocolo detallado en [`docs/ANOTACION.md`](docs/ANOTACION.md).

**Advertencia que debe resolverse antes de publicar:** la consigna exige las 13 fotografías originales en `data/`. Este paquete **no las contiene** porque no fueron adjuntadas en sus 13 archivos originales y la redistribución de imágenes Google Street View requiere revisar permisos. Para completar el repositorio con autorización, copiar los originales a `data/images/` mediante `python scripts/import_originals.py --source /RUTA/A/LAS/13_IMAGENES`, verificando que no se modifican los bytes. No declares la reproducción integral cumplida mientras falten.

## Reproducción integral desde cero (E0–E5)

Con las 13 imágenes originales autorizadas en `data/images/`, ejecuta:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_entrega.py --require-images
CONFIG=configs/experiment_pinned.yaml DEVICE=auto \
  bash scripts/reproducir_todos_los_experimentos.sh runs/entrega_repro_01
```

El script principal corre E0–E5, incluidas **imagen completa (sin ROI), ROI manual, OpenCV y OWLv2**, y deja todas las salidas separadas. El informe E5 reúne las **ocho combinaciones** de entrada espacial y banco de prompts. Guía detallada: [`docs/REPRODUCIR_ENTREGA_TRES_REQUISITOS.md`](docs/REPRODUCIR_ENTREGA_TRES_REQUISITOS.md). La ejecución requiere modelos descargados y puede utilizar GPU.

## Tres comandos de auditoría (CPU)

```bash
python -m pip install -r requirements.txt
python scripts/check_entrega.py
PYTHONPATH=src bash scripts/run_offline_audit.sh runs/auditoria_final --plots
```

Para inferencia real con las 13 fotografías y los pesos congelados:

```bash
python scripts/check_entrega.py --require-images
PYTHONPATH=src python -m urban_vlm run --config configs/experiment_pinned.yaml \
  --device auto --output runs/replica_e0
```

**Datos verificables:** `reference/historical_2026_11_21/` conserva E0, E3, E4 y los resultados E5. `docs/diagnostico_13_fachadas.csv` integra las decisiones y las hipótesis por caso. E1 es una ablation post hoc; E2 está implementado pero no se atribuyen resultados no disponibles. **Los resultados son de desarrollo, no held-out.**

---

# UrbanGroundFloor — Experimental study of ground-floor commerce with SigLIP 2

**Parte B · Prueba técnica · Repositorio reproducible · v1.1 (E5 auto-ROI)**  
**Pregunta:** *¿Hay evidencia visible de actividad comercial activa en la planta baja del frente objetivo?*

Este repositorio **no entrena** ningún VLM. Usa el checkpoint preentrenado `google/siglip2-base-patch16-224` para analizar **13 fotografías de Street View**. Permite contrastar la imagen completa con una ROI manual de planta baja, inspeccionar logits individuales, realizar intervenciones espaciales y estudiar dos variantes jerárquicas de presencia/operación.

> **Advertencia experimental.** Las 13 fotos, las ROI y los prompts ya se utilizaron durante el desarrollo. Las cifras son **exploratorias**, no una evaluación independiente. La clase **I** significa *operación no verificable desde la fotografía*, **no** un negocio cerrado. La foto 02 y la 13 parecen mostrar el mismo comercio desde dos perspectivas. Únicamente hay **una** foto de clase N. Los logits de SigLIP 2 no son probabilidades de operación.

## 1. Qué incluye

| Experimento | Variable estudiada | Dependencia de GPU | Archivos principales |
|---|---|---|---|
| **E0 — Baseline** | Completa/ROI × genéricos/contextuales | Sí, para inferencia | `predictions.csv`, `prompt_logits.csv`, `metrics.json` |
| **E1 — Aggregation** | Mean, median, Top-2, max, log-mean-exp | No | `aggregation_summary.csv`, `changes_vs_mean.csv` |
| **E2 — Spatial interventions** | Completa, ROI, blur superior, blur interior, blur exterior | Sí | `interventions.csv`, `intervention_effects.csv` |
| **E3 — Hierarchical** | Presencia de establecimiento → evidencia de operación | Sí | `hierarchical_predictions.csv`, `hierarchical_logits.csv` |
| **E4 — Operation refinement** | Únicamente nuevos prompts de operación; presencia E3 congelada | Sí | `operation_e4_logits.csv`, `paired_e3_e4.csv` |
| **Study report** | Compara E0–E4, errores y procedencia | **No** | `study_report.md`, `metrics_all.csv`, `cases_13.csv`, gráficas |

Los cinco experimentos están implementados; el **reporte integrado** conserva resultados anteriores y explica los cambios por fotografía. Los pesos del VLM permanecen congelados.

## 2. Instalación (Ubuntu)

Se recomienda **Python 3.10 o 3.11**. La documentación de una ejecución previa registra Python 3.10.22, PyTorch 2.6.0 + CUDA 12.4, Transformers 4.57.6 y RTX 3090; no hay garantía de exactitud bit a bit entre dispositivos.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Con GPU NVIDIA, instala primero un PyTorch compatible con tu versión de CUDA.
python -m pip install -e ".[inference,report,test]"
```

**Para pruebas de CPU sin modelo ni imágenes** basta con:

```bash
python -m pip install -e ".[report,test]"
```

**Importante:** ejecuta los comandos como `python -m urban_vlm`, no con un binario `urban-vlm` posiblemente instalado desde otra versión. Los scripts `.sh` configuran `PYTHONPATH` para utilizar siempre el código de este repositorio.

## 3. Fotografías y anotaciones

Las imágenes **no se redistribuyen** por cuestiones de licencia/atribución. Copia las 13 fotografías originales **640×640** a `data/images/` con nombres `fachada_01.jpg` ... `fachada_13.jpg`. No renombres, recortes ni redimensiones las fotografías antes de utilizar `data/rois.csv`.

- `data/rois.csv`: ROI manual de cada planta baja en coordenadas originales (Pillow: `x1,y1,x2,y2` con fin exclusivo).
- `data/annotations.csv`: etiqueta humana y evidencia visual, confirmadas mediante revisión iterativa (no anotación ciega).
- `configs/prompts_generic.json` y `configs/prompts_contextual.json`: bancos históricos de 15 textos por condición.
- `configs/prompts_hierarchical.json`: 20 textos (dos decisiones de 10 textos).
- `configs/prompts_operation_e4.json`: 10 textos nuevos para la operación; **no altera** la presencia de E3.
- `configs/experiment.yaml`: checkpoint, ROI, rutas, semilla y tamaño esperado.

**Categorías:** `A` evidencia de local abierto o preparado para vender, `I` establecimiento identificado sin operación verificable, `N` sin comercio visible en el *frente objetivo*. Las plantas superiores residenciales no impiden A; tampoco se deben atribuir al inmueble objetivo comercios vecinos (especialmente F11). Consulta `docs/ANOTACION.md`.

```bash
python -m urban_vlm validate --config configs/experiment.yaml
python -m urban_vlm preview-rois --config configs/experiment.yaml --output runs/roi_review.png
```

Estos comandos fallarán explícitamente si las fotos no existen o no corresponden a las dimensiones/ROI almacenadas.

## 4. Prueba inmediata de reproducibilidad (sin GPU ni fotografías)

El proyecto contiene **logits y predicciones históricos** procedentes de una ejecución real previa, separados bajo `reference/historical_2026_11_21/`. No se consideran resultados de evaluación independiente.

```bash
python -m pytest -q
bash scripts/run_offline_audit.sh runs/offline_01 --plots
```

Abre `runs/offline_01/FINAL_REPORT/study_report.md` y las gráficas PNG. Obtendrás las ocho condiciones E0, E3 y E4; la ablación E1 y el diagnóstico textual por fotografía. **E2 requiere la inferencia sobre imágenes reales; no hemos fabricado sus logits históricos a partir de otros experimentos.**

## 5. Primera inferencia: dos fotografías

```bash
python -m urban_vlm run --config configs/experiment.yaml \
  --limit 2 --device auto --output runs/smoke_baseline_2
```

`--limit` es solo para verificar la instalación. **No calcules ni presentes métricas de las dos fotos como resultado final.**

## 6. Corrida integral E0–E4

La forma recomendada para tus nuevos experimentos es:

```bash
DEVICE=auto PLOTS=1 bash scripts/run_full_study.sh runs/study_01
```

La ejecución verifica imágenes y ROI, corre E0, E1, E2, E3, E4, genera un análisis por fotografía y construye el informe final. Si falta GPU o alguna dependencia, se detendrá sin presentar una corrida incompleta como concluida.

**No se sobrescriben corridas previas.** Para una repetición usa `runs/study_02`.

**Para fijar el commit observado en E3 durante toda una corrida nueva:**

```bash
CONFIG=configs/experiment_pinned.yaml DEVICE=auto PLOTS=1 \
  bash scripts/run_full_study.sh runs/study_pinned_01
```

El archivo `experiment_pinned.yaml` reproduce el **ID de checkpoint** observado en el manifiesto de E3; no modifica ni valida por sí solo la escala, las etiquetas o las imágenes. La misma configuración se utiliza en todas las etapas del estudio.


Los archivos clave serán:

```text
runs/study_01/
  execution.log
  roi_review.png
  E0_baseline/{predictions.csv,prompt_logits.csv,metrics.json,manifest.json}
  E1_aggregation/{aggregation_summary.csv,changes_vs_mean.csv}
  E2_spatial/{interventions.csv,intervention_effects.csv,manifest.json}
  E3_hierarchical/{hierarchical_predictions.csv,hierarchical_logits.csv,manifest.json}
  E4_operation/{operation_e4_logits.csv,paired_e3_e4.csv,metrics.json,manifest.json}
  case_analysis/{per_image_analysis.csv,case_report.md}
  FINAL_REPORT/{study_report.md,cases_13.csv,metrics_all.csv,
                stage1_presence_metrics.csv,confusion_all.csv,
                operation_prompt_diagnostics.csv,study_inputs.json,
                accuracy_comparison.png,confusion_*.png}
```

La primera etapa E3 se reutiliza en E4 sin volver a inferir sus 10 textos. El chequeo E4 compara hashes de entradas y el commit del checkpoint para evitar mezclar experimentos.

## 7. Ejecución por partes

Puedes ejecutar cada etapa independientemente:

```bash
# E0 — Baseline
python -m urban_vlm run --config configs/experiment.yaml --device auto --output runs/E0

# E1 — Reagregación offline de logits de E0
python -m urban_vlm ablate --logits runs/E0/prompt_logits.csv \
  --predictions runs/E0/predictions.csv --output runs/E1

# E2 — Intervenciones espaciales
python -m urban_vlm diagnose --config configs/experiment.yaml --device auto --output runs/E2

# E3 — Jerarquía presencia / operación
python -m urban_vlm hierarchical --config configs/experiment.yaml \
  --prompts configs/prompts_hierarchical.json --region both \
  --aggregation mean --device auto --output runs/E3

# E4 — Mismos logits de presencia E3, nuevas descripciones de operación
python -m urban_vlm operation-e4 --config configs/experiment.yaml \
  --e3-dir runs/E3 --region both --aggregation mean --device auto --output runs/E4

# Informe científico a partir de CSV (CPU)
python -m urban_vlm report --baseline runs/E0/predictions.csv \
  --e3 runs/E3/hierarchical_predictions.csv \
  --e4-paired runs/E4/paired_e3_e4.csv \
  --e4-logits runs/E4/operation_e4_logits.csv \
  --spatial-effects runs/E2/intervention_effects.csv \
  --annotations data/annotations.csv --plots --output runs/FINAL_REPORT
```

Para la comprobación de E4 sin GPU, añade `--dry-run` a `operation-e4` con `--e3-dir` válido. No modifica los datos.

## 8. Resultados de referencia (exploratorios, no de generalización)

| Método | Región | Aciertos | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| E0 genérico | Imagen completa | 2/13 | 0.154 | 0.135 |
| E0 genérico | ROI | **10/13** | **0.769** | 0.704 |
| E0 contextual | ROI | 8/13 | 0.615 | 0.627 |
| E3 jerárquico | ROI | 9/13 | 0.692 | **0.778** |
| E4 operación | ROI | 9/13 | 0.692 | 0.771 |

E3 mejora la clase I respecto de E0, pero pierde sensibilidad de A. E4 corrige F07 e introduce un error en F08; la exactitud queda en 9/13. El valor de macro-F1 debe interpretarse con cautela porque N tiene un único caso. Los nuevos prompts y la agregación Top-2 son **post hoc**.

## 9. Protocolo científico y documentación

Lee primero:

1. `docs/RESEARCH_PROTOCOL.md`: hipótesis, variables, diseño, mecanismos y controles.
2. `docs/ANOTACION.md`: criterios A/I/N, uso mixto y casos ambiguos.
3. `docs/REPRODUCIBILITY.md`: modelo, hashes, entorno y pruebas.
4. `docs/experiment_log_template.md`: registro previo y posterior de cada nueva corrida.
5. `docs/RELEASE_CHECKLIST.md`: verificación antes de publicar en GitHub.

**Para cada experimento nuevo:** escribe la hipótesis antes de cambiar prompts o ROI; conserva una corrida de referencia; guarda el manifiesto y los logits; reporta tanto correcciones como errores nuevos. Nunca elijas retrospectivamente el mejor agregador/prompt y lo presentes como rendimiento externo.

## 10. Alcance y licencia

El **código** se publica bajo MIT. Las imágenes de Google Street View no están cubiertas por esta licencia y **no se incluyen** en el repositorio. Comprueba las condiciones aplicables antes de compartirlas. Las puntuaciones incluidas se facilitan como resultados históricos de desarrollo; no implican autorización para redistribuir las imágenes originales.

Este es un prototipo de análisis urbano, no un detector validado de horarios, licencias comerciales, apertura actual ni sesgo regional del preentrenamiento.

## E5 — Automatic candidate ground-floor ROIs (new, experimental)

This extension proposes new ROI boxes **without using the manually annotated ROI coordinates as inputs**. It provides a CPU OpenCV structural method and an optional OWLv2 zero-shot cue detector (with OpenCV fallback). The output is **only a localization proposal**; it does not decide whether commercial activity is present. Review the 13 boxes before evaluating them. See [`docs/E5_AUTOMATIC_ROI.md`](docs/E5_AUTOMATIC_ROI.md).

```bash
python -m pip install -e ".[auto-roi,inference,test]"

# CPU localization, unchanged manual data/rois.csv
python -m urban_vlm auto-rois --config configs/experiment.yaml \
    --method opencv --output runs/E5_auto_01

# Review the output and run a separate experimental SigLIP 2 baseline
python -m urban_vlm run --config configs/experiment.yaml \
    --rois-file runs/E5_auto_01/rois_auto.csv --device auto \
    --output runs/E5_auto_01/E5_inference
```

Optional OWLv2 proposal: `--method owlv2` (downloads `google/owlv2-base-patch16-ensemble`). Automatic ROI data and the previous manual E0–E4 conditions are never overwritten. **Image-only boxes do not identify the intended target frontage in every scene**; the facade 11 is particularly ambiguous. No results on the user's 13 photographs have been computed for E5 in the construction environment.

## E5 — Comparación SigLIP 2 sin ROI / manual / OpenCV / OWLv2

Para evaluar el impacto de la localización automática en la clasificación de las 13 fachadas, utiliza el nuevo script:

```bash
python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv \
  --device auto --output runs/E5_siglip_01
```

Este comando usa el código local del repositorio (no `urban-vlm` global), verifica las huellas de los insumos, genera las tres corridas por tipo de ROI y resume además la condición de imagen completa como control, para un total de ocho combinaciones. Para revisar antes de descargar modelos, usa `--dry-run`. Documentación: [`docs/E5_CLASSIFICATION.md`](docs/E5_CLASSIFICATION.md).

**Análisis académico integrado E5:** [`docs/E5_ROI_EXPERIMENTO_INTEGRAL.md`](docs/E5_ROI_EXPERIMENTO_INTEGRAL.md), incluyendo comparación espacial (IoU), clasificación pareada con los dos bancos y diagnóstico por fotografía. Es un componente principal de [`nota_tecnica.pdf`](nota_tecnica.pdf), no solamente una prueba opcional.
