# Metodología y trazabilidad experimental

## 1. Alcance
Modelo congelado `google/siglip2-base-patch16-224`, cero actualizaciones de pesos; 13 vistas originales.
Comparación factorial: región (full/roi) × descripciones (generic/contextual). ROI manual precisa para planta baja.

## 2. Función de decisión
Sean ℓ(I,t) los logits de compatibilidad imagen-texto, y cinco textos por clase c∈{A,I,N}.

`S_c(I) = (1/5) · Σ_j ℓ(I, t_cj)`; `ŷ(I) = argmax_c S_c(I)`.

Los logits no son probabilidades; no usar `softmax` de clases como incertidumbre calibrada.
Se guardan 15 logits por decisión y una diferencia entre clase ganadora y segunda.

## 3. Baselines
- full + generic — línea base genérica sin recorte.
- full + contextual — ablación de formulaciones textuales.
- roi + generic — efecto de ROI manteniendo los textos.
- roi + contextual — interacción ROI×texto.

## 4. Mecanismos examinados
- **Interferencia espacial/escala:** el recorte cambia simultáneamente contexto y tamaño relativo.
- **Comercio vs. actividad:** cortina cerrada con anuncio puede parecer comercio sin probar actividad.
- **Prompts heterogéneos:** promediar restaurante, ferretería y puesto puede penalizar subtipos.
- **Fuga del frente objetivo:** tiendas vecinas no deben etiquetar un frente central residencial.

Las explicaciones son hipótesis apoyadas por señales y logits, **no mapas de atención causal**.

## 5. Perturbaciones diagnósticas (opcionales)
Blur de mitad superior, blur dentro de ROI, blur exterior de ROI, recorte. No confundir resultados
multiclase del repositorio con márgenes de ensayos binarios previos.

## 6. Amenazas a la validez
Selección de ROI y prompts post hoc, n=13, N=1, F02/F13 no independientes; no hay validación
externa ni certeza de operación real de los negocios. No atribuir error a sesgo regional demostrado.

## 7. Reproducibilidad
El manifiesto guarda huellas SHA256 de imágenes, configuración, anotaciones y bancos de prompts;
registra checkpoint solicitado y commit devuelto cuando esté disponible. Para replicación exacta:
fijar versiones del entorno, checkpoint por hash, GPU y algoritmos deterministas según compatibilidad.
La semilla por sí sola no garantiza identidad de floats entre plataformas.
