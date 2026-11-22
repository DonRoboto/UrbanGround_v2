# Reproducción desde cero — tres requisitos de la convocatoria

## 1. README

El archivo [`README.md`](../README.md) describe la pregunta, clases A/I/N, formulaciones de prompts, ROI manuales y automáticas, resultados esperados y limitaciones. El presente documento es la ruta breve para un evaluador.

## 2. Dependencias

Requiere Python 3.10 u 3.11 (Ubuntu recomendado); GPU NVIDIA opcional pero recomendable para inferencia. Instala PyTorch compatible con tu driver/CUDA **antes** de instalar el resto si deseas GPU. En la raíz:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

`requirements.txt` instala el proyecto local editable con los extras `inference`, `auto-roi`, `report` y `test`. Sus rangos admiten actualizaciones: **los resultados bit a bit no están garantizados** entre máquinas y versiones. Para la inferencia se recomienda `configs/experiment_pinned.yaml`, que fija el commit de SigLIP 2 observado en las corridas históricas. El checkpoint de OWLv2 usado en E5 se identifica por nombre, pero no por commit; hay que registrar la revisión descargada si se requiere repetibilidad estricta del localizador.

## 3. Datos originales

Colocar las 13 imágenes **sin alterar sus bytes** en `data/images/`: `fachada_01.jpg` a `fachada_13.jpg`, todas 640 × 640. El script de importación opcional es:

```bash
python scripts/import_originals.py --source /ruta/a/fotos_originales
python scripts/check_entrega.py --require-images
```

`data/annotations.csv` y `anotaciones.csv` incluyen A/I/N y evidencia humana; `data/rois.csv` fija las 13 ROI manuales. Los originales **no están distribuidos en el ZIP** y debe resolverse el permiso correspondiente antes de publicarlos en un repositorio abierto; en su ausencia, no puede declararse reproducida la inferencia desde cero.

## 4. Script principal: todas las formulaciones

```bash
CONFIG=configs/experiment_pinned.yaml DEVICE=auto OWL_DEVICE=auto PLOTS=1 \
  bash scripts/reproducir_todos_los_experimentos.sh runs/entrega_repro_01
```

El lanzador aborta ante errores, **nunca sobreescribe una corrida existente** y organiza resultados:

- `E0_E4/E0_baseline`: cuatro configuraciones, imagen completa/ROI × genéricos/contextualizados.
- `E0_E4/E1_aggregation`: cinco reglas para combinar los logits de los prompts.
- `E0_E4/E2_spatial`: intervenciones espaciales sobre imágenes.
- `E0_E4/E3_hierarchical`: presencia comercial → actividad verificable.
- `E0_E4/E4_operation`: refinamiento de prompts de operación.
- `E0_E4/FINAL_REPORT`: informe y métricas de las formulaciones E0–E4.
- `E5/roi_opencv` y `E5/roi_owlv2`: propuestas automáticas y diagnósticos de región.
- `E5/clasificacion`: tres métodos ROI × dos bancos textuales, predicciones pareadas, IoU e informe.

Todos los experimentos se ejecutan sin entrenamiento del VLM. **El comando puede descargar checkpoints en la primera ejecución**. La comparación de E5 usa las ROI manuales como referencia de IoU y NO para orientar los detectores automáticos.

## 5. Alternativa rápida sin imágenes ni GPU

Si el evaluador todavía no dispone de los 13 originales, es posible regenerar un informe a partir de puntuaciones históricas, lo cual **no equivale a repetir la inferencia**:

```bash
bash scripts/run_offline_audit.sh runs/auditoria_sin_gpu --plots
```

## 6. Archivos a revisar

- `runs/entrega_repro_01/reproducir_todos.log` — registro de ejecución.
- `runs/entrega_repro_01/E0_E4/FINAL_REPORT/study_report.md` — análisis principal.
- `runs/entrega_repro_01/E5/clasificacion/comparison/E5_comparison_report.md` — análisis automático.
- `runs/entrega_repro_01/E5/clasificacion/comparison/metrics_summary.csv` — métricas de las seis condiciones.
- `configs/prompts_generic.json`, `configs/prompts_contextual.json`, `configs/prompts_hierarchical.json`, `configs/prompts_operation_e4.json` — formulaciones exactas.

**Limitación:** 13 ejemplos empleados durante el desarrollo, solo una etiqueta N, y posible dependencia entre F02 y F13. Estas métricas no representan desempeño en un conjunto held-out.
