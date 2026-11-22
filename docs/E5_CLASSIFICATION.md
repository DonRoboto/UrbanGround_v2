# E5 — Clasificación SigLIP 2: imagen completa, ROI manual, OpenCV y OWLv2

El objetivo es medir **clasificación** por separado de la calidad de **localización**. No se entrenan pesos nuevos.

## Requisitos

- UrbanGroundFloor **v1.1** con `auto-rois` y `run` operativos.
- Las 13 imágenes originales en `data/images/`, de **640×640 píxeles**.
- Las ROI manuales originales en `data/rois.csv`.
- ROI automáticas de E5 en `runs/E5_opencv_01/rois_auto.csv` y `runs/E5_owlv2_01/rois_auto.csv` junto con su respectivo `manifest.json` (obligatorio para auditar los hashes).
- Modelo `google/siglip2-base-patch16-224` y dependencias de inferencia instaladas.

## Instalación del parche

Extrae el ZIP **desde la raíz del repositorio**; no borra `data/` ni `runs/`. Desde la raíz:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m urban_vlm run --help
```

El lanzador usa siempre `sys.executable` y el código local, evitando un ejecutable `urban-vlm` antiguo instalado en el entorno.

## 1. Verificación sin GPU

```bash
python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv \
  --dry-run
```

Verifica las 13 fotografías, sus dimensiones, 3 conjuntos de ROI y hashes de los manifiestos de E5. **Las ROI manuales se emplean como referencia de IoU, no para seleccionar ROIs automáticas.** Si las fotografías cambiaron desde E5, es necesario ejecutar E5 otra vez, no editar hashes a mano.

## 2. Primera inferencia: prueba de humo con dos imágenes

```bash
python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv \
  --limit 2 --device auto --output runs/E5_siglip_smoke_01
```

La prueba de humo no es un resultado estadístico de 13 imágenes.

## 3. Inferencia completa sobre 13 imágenes

```bash
python scripts/run_e5_siglip2.py \
  --config configs/experiment_pinned.yaml \
  --opencv-rois runs/E5_opencv_01/rois_auto.csv \
  --owlv2-rois runs/E5_owlv2_01/rois_auto.csv \
  --device auto --output runs/E5_siglip_01
```

La corrida vuelve a evaluar con el **mismo modelo, prompts e imágenes** cuatro condiciones espaciales: imagen completa (**sin ROI**), ROI manual, ROI OpenCV y ROI OWLv2. Cada método de ROI produce también predicciones para la imagen completa; el informe toma las del experimento manual como control y compara las otras dos para detectar inconsistencias. No es necesario repetir la inferencia solo para incluir el control en la tabla.

Para reutilizar una corrida manual previa **comprobable mediante su manifiesto**, agrega, por ejemplo, `--manual-run runs/study_pinned_01/E0_baseline`. Si su checkpoint, imágenes o bancos de prompts difieren de los de las corridas nuevas, la comparación rechazará los datos. Sin ese argumento, el lanzador vuelve a inferir las ROI manuales y elimina esa posible incompatibilidad.

## Resultados

```text
runs/E5_siglip_01/
  manual/  predictions.csv, prompt_logits.csv, metrics.json, manifest.json, ...
  opencv/  ...
  owlv2/  ...
  comparison/
    metrics_summary.csv           4 condiciones espaciales × 2 bancos (8 filas)
    paired_predictions.csv        13 fotografías × 2 bancos, incluye full_prediction
    changes_vs_full.csv           Cambios de cada ROI frente a imagen completa
    changes_vs_manual.csv         Mejoras, deterioros y cambios por fotografía
    roi_iou.csv                   IoU (auto vs manual) por fotografía
    comparison_manifest.json      Hashes y revisión de procedencia
    E5_comparison_report.md       Resumen académico
```

**Condiciones de interpretación.** El conjunto original fue empleado en el desarrollo de prompts y ROI. Estas cifras son **exploratorias**, no generalización. `I` significa *operación no verificable*, no negocio cerrado. Las escenas F02 y F13 pueden mostrar el mismo establecimiento. F11 es el único negativo; no interpretes su recall como estimación estable. OWLv2 utiliza señales semánticas; su detector no es neutral con respecto a la pregunta de comercio.

## Diagnóstico de incidencias

- **Falta `auto-rois`**: usa `python -m urban_vlm ...` y confirma que el parche v1.1 está instalado en el repositorio actual. El script E5 en sí utiliza solo el comando `run` ya disponible.
- **No encuentro las ROI E5**: copia las carpetas `E5_opencv_01` y `E5_owlv2_01` en `runs/`, incluyendo `rois_auto.csv` y `manifest.json`. También puedes indicar rutas absolutas.
- **Hashes de imágenes no coinciden**: volviste a recortar/redimensionar o sustituir imágenes después de E5. No puedes comparar directamente; vuelve a generar esas ROI sobre las imágenes actuales.
- **Checkpoint distinto**: utiliza `configs/experiment_pinned.yaml` de inicio a fin para evitar variaciones de versión.
- **La carpeta ya existe**: el programa nunca sobrescribe; selecciona otra mediante `--output`.
