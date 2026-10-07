# 

Cómo usarlos:
- Pégalos en Claude Code (terminal o pestaña **Code** de Claude Desktop) desde la raíz del repo.
- Los prompts están en español; todos piden que el código, los comentarios y la documentación del
  repo se escriban en inglés.
- Para tareas grandes, empieza en **modo Plan** (Shift+Tab), revisa el plan y luego cambia a
  **Accept edits**.
- Una tarea = una rama = un PR. Crea la rama antes de pegar el prompt:
  `git switch main && git pull && git switch -c <ID>-<slug>`.
- Máquina: Mac mini Apple M2 Pro, 16 GB, MPS (sin CUDA). Ya está incluido en los prompts que lo
  necesitan.

Puntos de control: G1 jue 8 oct mediodía (pipeline de punta a punta) · G2 vie 9 · G3 mié 14 ·
G4 vie 16 · G5 mié 21 (entrega).

---

## Fase 0 · Arranque

### F0-01 · Repositorio
```
/bootstrap-repo
```

### F0-03 · Protocolo y contratos congelados
```
Tarea F0-03. Reescribe docs/protocol.md en inglés como el protocolo experimental congelado.
Usa exactamente estos valores; no agregues ni cambies hiperparámetros.

Secciones:
1. Preguntas de investigación: (a) cuál de ResNet-50, EfficientNet-B0 y ViT-S/16 rinde mejor en
   EuroSAT y a qué costo; (b) cuánto se degrada cada uno en Aguascalientes sin ajuste; (c) cuánto
   recupera con ajuste fino ligero; (d) RGB vs. 13 bandas; (e) pesos SSL4EO-S12 vs. ImageNet.
2. Datos: EuroSAT multiespectral de TorchGeo con su partición oficial train/val/test; RGB = B04,
   B03, B02 tomadas de los mismos archivos. Aguascalientes: ~400 parches de abril 2024, partición
   espacial por bloques de 5x5 km, 50 % ajuste fino / 50 % test, estratificada por clase, ningún
   bloque en ambas particiones.
3. Clases: las 10 de EuroSAT en el orden de índices de TorchGeo (AnnualCrop, Forest,
   HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River,
   SeaLake). Las clases con menos de 10 parches de test en Aguascalientes se excluyen de su
   macro-F1 y se reportan aparte.
4. Modelos: timm resnet50, efficientnet_b0 y vit_small_patch16_224 con pesos ImageNet. Con 13
   bandas, copia los pesos RGB de la primera capa a B04/B03/B02 y su promedio al resto. Variante
   SSL (Debería): ResNet50_Weights.SENTINEL2_ALL_MOCO y ViTSmall16_Weights.SENTINEL2_ALL_DINO de
   TorchGeo, solo 13 bandas.
5. Configuración común en tabla: entrada de 64 redimensionada a 224 (bilineal); normalización
   por banda del train de EuroSAT (la de TorchGeo para pesos SSL4EO); volteos horizontal y
   vertical + rotaciones de 90°; AdamW lr 1e-4, weight decay 0.05; 1 época de warmup + cosine;
   batch efectivo 64 = 2 micro-lotes de 32 con acumulación de gradiente; máximo 15 épocas; early
   stopping con paciencia 3 sobre macro-F1 de val; entropía cruzada (ponderada por clase en
   Aguascalientes); semilla 0; Mac mini Apple M2 Pro con MPS en fp32 para todas las corridas.
6. Matriz: E1 3 modelos x {RGB, 13} con ImageNet = 6 corridas; E2 SSL = 2 corridas; E3
   evaluación sin ajuste de los 8 checkpoints; E4 ajuste fino de los 8 checkpoints en modos head
   y last_block, 10 épocas fijas = 16 corridas.
7. Métricas: accuracy, macro-F1 (principal), F1 por clase, matriz de confusión, parámetros,
   tiempo total y por época, memoria pico (driver de MPS), parches/s, épocas hasta el 95 % del
   mejor macro-F1 de val. Caída = F1_EuroSAT - F1_Ags_sin_ajuste; Recuperación =
   (F1_Ags_ajustado - F1_Ags_sin_ajuste) / Caída.
8. Regla de selección: mayor macro-F1 en Aguascalientes tras el ajuste last_block; si dos modelos
   quedan a menos de 1 punto, gana el de más parches/s.
9. Bitácora de cambios: tabla (fecha, cambio, razón, aplica a todos los modelos: sí) con la
   primera fila "2026-10-07, protocolo congelado".

Después haz que configs/base.yaml coincida exactamente con la sección 5.

Luego reescribe docs/contracts.md en inglés con estos 4 contratos (ruta, formato, campos y
tipos, quién lo escribe y quién lo lee):
1. Parche: data/ags/patches/{cell_id}.tif, GeoTIFF 13x64x64 uint16, EPSG:32613, orden de bandas
   de EuroSAT en TorchGeo (B01..B12, B8A al final), valores = reflectancia TOA x 10000.
2. Etiquetas: data/ags/labels.csv con cell_id, label (uno de los 10 nombres de clase),
   annotator, block_id, split (finetune|test).
3. Checkpoint: checkpoints/{model}_{bands}_{init}.pt más un .yaml con modelo, bandas, init,
   estadísticas de normalización, lista de clases y versión del protocolo.
4. Resultado: experiments/results/{run_id}.json con run_id, model, bands, init, split, accuracy,
   macro_f1, f1_per_class, confusion, params, train_time_s, peak_mem_mb, epochs_to_95,
   patches_per_s, device, micro_batch, accum_steps, protocol_version, git_commit.
Agrega src/hydroscope/contracts.py con validate_result(dict) que falle si falta un campo o tiene
el tipo equivocado, más una prueba con un ejemplo válido y uno inválido. Corre las pruebas.
Muéstrame el plan antes de editar.
```
Al terminar: `/protocol-check G1`, commit y `git tag protocol-v1`.

### F0-04 · Earth Engine
Se hace a mano en el navegador (registro del proyecto no comercial). Después, en tu terminal:
`earthengine authenticate` y `export EE_PROJECT=<tu-id>` en `~/.zshrc`.

### F0-05 · Plantilla IEEE
```
Tarea F0-05. Crea el esqueleto del artículo IEEE de conferencia en paper/, en inglés:
- paper/main.tex con \documentclass[conference]{IEEEtran}, basado en la estructura oficial de
  bare_conf.tex: título provisional "CNNs vs. Vision Transformers under Geographic Domain Shift:
  From EuroSAT to Semi-Arid Aguascalientes, Mexico", bloque de autores, abstract y keywords
  provisionales, y un \input por sección.
- paper/sections/{introduction,related,data,methodology,results,discussion,conclusions}.tex,
  cada uno solo con su \section{} y un \todo{} que describa lo que contendrá. Sin contenido
  inventado y sin números.
- paper/references.bib vacío salvo un comentario; no inventes referencias.
- Paquetes: graphicx, booktabs, amsmath, siunitx, cite, xcolor, url; una macro \todo que
  imprima texto rojo.
- paper/tables/ y paper/figures/ con .gitkeep.
- Target `paper` en el Makefile que ejecute `tectonic paper/main.tex`, y agrega los archivos
  auxiliares de LaTeX (*.aux, *.log, *.bbl, *.blg, *.out) a .gitignore.
Compila con `make paper` y corrige hasta que paper/main.pdf se genere.
```

---

## Fase 1 · Literatura (solo 3 referencias clave)

### F1-01 · Fichas y .bib
```
Tarea F1-01. Crea docs/literature.md y paper/references.bib en inglés con exactamente estas tres
referencias: 10.1109/JSTARS.2019.2918242, 10.3390/rs13030516, 10.1109/MGRS.2023.3281651.
1. Obtén cada entrada BibTeX con:
   curl -sLH "Accept: application/x-bibtex" https://doi.org/<DOI>
   Llaves: helber2019eurosat, bazi2021vit, wang2023ssl4eo. Nunca escribas BibTeX a mano.
2. En docs/literature.md, una tabla: llave, modelos, datos, resultado principal reportado (con
   su origen: abstract o tabla N, o "check PDF"), hallazgo relevante para nuestro proyecto y
   sección del artículo donde lo citamos.
3. Confirma que el .bib compila con `make paper` (\nocite{*} temporal y luego quítalo).
```

### F1-02 · Related Work
```
/paper-section related
```

### F1-03 · Introducción y motivación
```
/paper-section introduction
```
Antes de correrlo, deja en `docs/literature.md` las fuentes oficiales que quieras usar sobre el
agua en Aguascalientes (CONAGUA, INEGI). El agente no debe inventar cifras: lo que no tenga
fuente queda como `\todo{}`.

---

## Fase 2 · Datos

### F2-01 · EuroSAT (incluye ajustes para el M2 Pro)
```
Usa el subagente ml-engineer. Tarea F2-01, siguiendo docs/protocol.md y docs/contracts.md.
Escribe el código, los comentarios y la documentación en inglés.

Máquina: Mac mini con Apple M2 Pro, 16 GB de memoria unificada, backend MPS (sin CUDA).
Todo lo siguiente aplica a los tres modelos por igual, nunca a uno solo.

1. src/hydroscope/data/eurosat.py: envoltura de torchgeo.datasets.EuroSAT con
   root="data/eurosat", la partición oficial train/val/test de TorchGeo y download=True.
   Opción `bands`: "ms13" = las 13 bandas en el orden de TorchGeo (B01..B12, B8A al final) y
   "rgb" = ("B04","B03","B02") tomadas de los mismos archivos. Expón CLASSES en el orden de
   índices de TorchGeo.
2. Función que calcule la media y la desviación por banda sobre TODOS los píxeles SOLO del
   split de train (en streaming, float64) y las guarde en configs/stats/eurosat_train_stats.json
   con el nombre de cada banda como llave. No uses val ni test para las estadísticas.
3. scripts/prepare_eurosat.py: descarga el dataset, guarda en docs/data_eurosat.md el conteo de
   imágenes por clase y por split, escribe el archivo de estadísticas y guarda cada split como
   arreglos uint16 en data/eurosat/cache/{split}_x.npy y {split}_y.npy. La clase del dataset
   debe leer del caché cuando exista, abriéndolo con np.load(..., mmap_mode="r").
4. src/hydroscope/utils/device.py: elige cuda, si no mps, si no cpu. En MPS usa fp32 (sin
   autocast) y PYTORCH_ENABLE_MPS_FALLBACK=1, con un aviso en el log si alguna operación cae a
   CPU. Memoria pico: torch.mps.driver_allocated_memory() en MPS y
   torch.cuda.max_memory_allocated en CUDA. No modifiques PYTORCH_MPS_HIGH_WATERMARK_RATIO.
5. Preparación para F3-03 (solo deja las utilidades listas):
   - Batch efectivo 64 = 2 micro-lotes de 32 con acumulación de gradiente.
   - Redimensionar de 64 a 224 en el dispositivo, por lote, con
     torch.nn.functional.interpolate (bilinear), no imagen por imagen en CPU.
   - DataLoader con num_workers=2, persistent_workers=True y pin_memory=False.
   - Opción --max-steps para que F3-04 pueda medir 100 pasos por modelo.
6. Agrega en docs/contracts.md los campos "device", "micro_batch" y "accum_steps" al JSON de
   resultados (si F0-03 no los incluyó).
7. Agrega una fila a la bitácora de cambios de docs/protocol.md:
   "2026-10-07 | Hardware: Apple M2 Pro MPS, fp32; batch efectivo 64 = 2x32 con acumulación
   (BatchNorm ve 32 muestras); memoria pico = memoria del driver MPS; datos en caché .npy;
   resize en el dispositivo | única máquina disponible | aplica a todos los modelos: sí".
8. Pruebas con torchgeo.datasets.EuroSAT100 (pequeño y rápido): forma (13,64,64) en ms13,
   forma (3,64,64) en rgb, los canales rgb iguales a las bandas B04, B03 y B02 de ms13, etiquetas
   en [0,9], y que una muestra leída del caché sea idéntica a la original.
9. Ejecuta las pruebas y dame el comando para correr la preparación completa; no la corras tú.
```
Verifica: train + val + test = 27,000; 10 clases; 13 bandas con B8A al final.

### F2-02 · Compuesto de abril 2024 en Earth Engine
```
Usa el subagente geo-data-engineer. Tarea F2-02. Código y documentación en inglés.
1. src/hydroscope/geo/gee_export.py: inicializa Earth Engine con
   ee.Initialize(project=os.environ["EE_PROJECT"]) (nunca escribas el ID en el código).
2. Límite estatal de Aguascalientes desde FAO/GAUL/2015/level1 (ADM1_NAME == "Aguascalientes"),
   o desde un archivo del INEGI si existe en data/ref/.
3. Colección COPERNICUS/S2_HARMONIZED, abril 2024, nubes enmascaradas con los bits 10 y 11 de
   QA60, mediana, las 13 bandas en el orden de EuroSAT en TorchGeo (B1..B12 y B8A al final),
   tipo uint16, escala 10 m, CRS EPSG:32613, recortado al estado.
4. Exporta a Google Drive con Export.image.toDrive (fileDimensions para partir en mosaicos si
   hace falta, maxPixels 1e10) y además una banda extra con el número de observaciones válidas
   por píxel.
5. Script que, ya descargados los GeoTIFF en data/ags/raw/, compare los histogramas por banda
   contra configs/stats/eurosat_train_stats.json y escriba docs/data_ags_check.md.
Dame los comandos para lanzar la exportación y para la comparación; no los corras tú.
```

### F2-03 · Malla de 640 m y parches
```
Usa el subagente geo-data-engineer. Tarea F2-03. Código y documentación en inglés.
1. src/hydroscope/geo/grid.py: malla fija de celdas de 640 m en EPSG:32613 que cubra el estado,
   con cell_id estable (fila_columna) y bloque de 5x5 km (block_id); guarda data/ref/grid.gpkg.
2. src/hydroscope/geo/tiling.py: corta un parche 13x64x64 uint16 por celda desde el raster de
   abril 2024 y lo guarda según el contrato; calcula la fracción de píxeles válidos; guarda
   también una vista previa RGB en PNG (B04, B03, B02, estiramiento 2-98 %) en
   data/ags/previews/{cell_id}.png.
3. Omite las celdas con menos del 80 % dentro del estado o con fracción válida < 0.6.
4. Pruebas: cada parche coincide con su celda y tiene forma (13,64,64).
Dame el comando para generar todos los parches.
```

### F2-04 · Guía y herramienta de etiquetado
```
Usa el subagente geo-data-engineer. Tarea F2-04. Escribe en inglés.
1. labeling/GUIDE.md de una página: para cada una de las 10 clases de EuroSAT, cómo se ve en el
   paisaje semiárido de Aguascalientes (por ejemplo: SeaLake = presas y bordos; PermanentCrop =
   huertas de guayaba y vid; HerbaceousVegetation = matorral y pastizal natural), y la regla:
   solo se etiquetan celdas con al menos 70 % de una sola clase; si no, "skip".
2. labeling/label.ipynb con ipywidgets: muestra el PNG de la celda grande, un mapa con imagen de
   alta resolución centrado en la celda (ipyleaflet con Esri World Imagery), 10 botones de clase
   más "skip", y guarda cada respuesta al instante en data/ags/labels.csv (cell_id, label,
   annotator, block_id; split se llena después). Debe poder retomar donde se quedó.
```

### F2-05 · Selección de celdas candidatas
```
Usa el subagente geo-data-engineer. Tarea F2-05 (preparación). Crea
scripts/select_candidates.py que genere data/ags/candidates.csv con unas 800 celdas: muestra
aleatoria estratificada por bloque y búsqueda dirigida de clases raras usando índices simples
(NDWI > 0 para agua, NDVI alto en la Sierra Fría para bosque). El orden de etiquetado debe ser
aleatorio. No asignes etiquetas: el etiquetado lo hace una persona.
```
El etiquetado de F2-05 (unos 400 parches) lo haces tú en `labeling/label.ipynb`.

### F2-06 · Partición espacial y Data Card
```
Usa el subagente geo-data-engineer. Tarea F2-06. Código y documentación en inglés.
1. scripts/split_ags.py: asigna split por block_id (50 % finetune / 50 % test), estratificado
   por clase en lo posible, con semilla fija; ningún bloque en ambas particiones. Escribe el
   split en data/ags/labels.csv.
2. Prueba que falle si un block_id o un cell_id aparece en las dos particiones.
3. docs/data_card.md: fuente, fecha, procesamiento, conteo por clase y por split, clases con
   menos de 10 parches en test, regla de etiquetado y limitaciones.
4. Si hay 50 parches etiquetados por dos personas, calcula el kappa de Cohen y agrégalo a la
   Data Card.
```

---

## Fase 3 · Entrenamiento en EuroSAT

### F3-01 y F3-02 · Fábrica de modelos y transformaciones
```
Usa el subagente ml-engineer. Tareas F3-01 y F3-02. Código y documentación en inglés.
1. src/hydroscope/models/factory.py: build_model(name, bands, init) para resnet50,
   efficientnet_b0 y vit_small_patch16_224 de timm con 10 clases. Con 13 bandas, copia los pesos
   RGB de la primera capa a B04/B03/B02 y su promedio al resto. Con init="ssl4eo", carga
   ResNet50_Weights.SENTINEL2_ALL_MOCO o ViTSmall16_Weights.SENTINEL2_ALL_DINO de TorchGeo
   (solo 13 bandas; cualquier otra combinación debe fallar con un mensaje claro).
2. src/hydroscope/data/transforms.py: normalización por banda con
   configs/stats/eurosat_train_stats.json (o la de TorchGeo para pesos SSL4EO), volteos
   horizontal y vertical y rotaciones de 90°, idénticas para los tres modelos.
3. Pruebas: forward con salida (2,10) en las 8 combinaciones válidas sobre MPS, y que las
   transformaciones den el mismo resultado con la misma semilla.
```

### F3-03 · Ciclo de entrenamiento y evaluación
```
Usa el subagente ml-engineer. Tarea F3-03. Código y documentación en inglés.
1. src/hydroscope/training/train.py: un solo ciclo para los tres modelos leído de un YAML que
   extiende configs/base.yaml. AdamW, 1 época de warmup + cosine, batch efectivo 64 = 2x32 con
   acumulación, resize 64->224 en el dispositivo por lote, máximo 15 épocas, early stopping con
   paciencia 3 sobre macro-F1 de val, semilla 0, MPS en fp32.
2. Registra por época: pérdida de train, macro-F1 de val y tiempo. Al final: parámetros, tiempo
   total, memoria pico, parches/s en inferencia y épocas hasta el 95 % del mejor macro-F1 de val.
   Guarda el mejor checkpoint (.pt + .yaml según el contrato) y las curvas por época en
   experiments/logs/{run_id}.csv.
3. src/hydroscope/training/evaluate.py: accuracy, macro-F1, F1 por clase, matriz de confusión y
   parches/s; escribe el JSON del contrato y valídalo con validate_result.
4. configs/smoke.yaml con EuroSAT100, 1 época, que escriba en una carpeta temporal (nunca en
   experiments/results/).
5. Corre la prueba de humo en MPS y muéstrame el JSON resultante.
```
Con esto se cumple **G1**.

### F3-04 · Medir el presupuesto de cómputo
```
Usa el subagente ml-engineer. Tarea F3-04. Para cada combinación de E1 (3 modelos x {rgb, ms13})
corre 100 pasos de entrenamiento con --max-steps 100 en MPS y mide segundos por paso. Con eso
estima en docs/compute.md el tiempo por época y por corrida completa (15 épocas, más
validación) y el total de las 6 corridas de E1 y las 2 de E2.
Si las 6 corridas de E1 superan 8 horas, propón entrada 128x128 para los TRES modelos (timm debe
interpolar los position embeddings del ViT) y estima de nuevo. No cambies nada del protocolo sin
mi aprobación; si lo apruebo, regístralo en la bitácora de docs/protocol.md.
```

### F3-05 · Corridas principales (E1)
```
/new-experiment resnet50 rgb imagenet
/new-experiment resnet50 ms13 imagenet
/new-experiment efficientnet_b0 rgb imagenet
/new-experiment efficientnet_b0 ms13 imagenet
/new-experiment vit_small rgb imagenet
/new-experiment vit_small ms13 imagenet
```
Luego:
```
Crea scripts/run_e1.sh que ejecute las 6 configs de E1 una tras otra con
caffeinate -i python -m hydroscope.training.train --config <config>, guarde el log de cada una en
experiments/logs/ y siga con la siguiente si una falla. No lo ejecutes; dame el comando.
```
Antes de lanzarlo en la noche: cierra apps pesadas y conecta el Mac a la corriente.

### F3-06 · Variante autosupervisada (E2, Debería)
```
/new-experiment resnet50 ms13 ssl4eo
/new-experiment vit_small ms13 ssl4eo
```

---

## Fase 4 · Generalización a Aguascalientes

### F4-01 · Evaluación sin ajuste
```
Usa el subagente ml-engineer. Tarea F4-01. Agrega a evaluate.py el split ags_test, que lea los
parches de data/ags/patches/ con split == test, aplique la misma normalización del checkpoint y
excluya del macro-F1 las clases con menos de 10 parches de test (reportándolas aparte). Evalúa
los checkpoints disponibles y dime si alguno falta.
```

### F4-02 · Ajuste fino ligero
```
Usa el subagente ml-engineer. Tarea F4-02. src/hydroscope/training/finetune.py con dos modos:
head (backbone congelado) y last_block (último bloque + cabeza), 10 épocas fijas, pérdida
ponderada por clase, usando solo split == finetune, misma configuración para los tres modelos.
Después evalúa cada resultado en ags_test. Dame los comandos de las corridas que tarden más de
10 minutos para lanzarlas yo.
```

### F4-03 · Caída y recuperación
```
/results-tables
```
```
Usa el subagente ml-engineer. Tarea F4-03. Crea notebooks/03_generalization.ipynb que lea solo
experiments/results/*.json y calcule la caída y la recuperación por modelo y variante, más el F1
por clase en Aguascalientes, con una figura PDF en paper/figures/. Señala qué clases confunde
más cada modelo.
```
Después: `/protocol-check G3`.

---

## Fase 5 · Análisis e interpretabilidad

### F5-01 y F5-02 · Costo y convergencia
```
/results-tables
```

### F5-03 · Grad-CAM
```
Usa el subagente ml-engineer. Tarea F5-03. src/hydroscope/explain/gradcam.py con
pytorch-grad-cam: última capa convolucional en ResNet-50 y EfficientNet-B0; en ViT-S/16 usa un
reshape_transform que quite el token CLS y reacomode los 196 tokens en 14x14. Genera en
paper/figures/gradcam.pdf una cuadrícula con 6 ejemplos por modelo: EuroSAT y Aguascalientes,
aciertos y errores, antes y después del ajuste fino.
```

### F5-04 · Selección del modelo
```
Aplica la regla de selección de docs/protocol.md con los JSON de experiments/results/ y escribe
docs/model_card.md en inglés: modelo elegido, datos, métricas (generadas por script), costo,
limitaciones y uso previsto. No escribas ninguna cifra a mano.
```

### F5-05 · Mapa del estado (Debería)
```
Usa el subagente geo-data-engineer. Tarea F5-05. Clasifica todas las celdas de abril 2024 con el
modelo de docs/model_card.md y genera paper/figures/ags_map_2024_04.pdf con una leyenda de
clases y el contorno estatal.
```

### F5-06 · Discusión
```
/paper-section discussion
```

---

## Fase 6 · Entregables

```
/paper-section data
/paper-section methodology
/paper-section results
/paper-section conclusions
```
```
/protocol-check G4
```
```
Revisa README.md en inglés como alguien que nunca vio el proyecto: instalación del entorno,
descarga de datos, cómo reproducir cada tabla del artículo y cómo citar. Sigue los pasos en un
entorno limpio y corrige lo que falle o sea ambiguo.
```
```
Con base en paper/ y paper/figures/, propón el guion de 12 diapositivas en inglés para la
presentación final: una idea por diapositiva y qué figura usa cada una. Ninguna cifra a mano.
```
```
Prepara el release v1.0: checkpoint del modelo elegido, data/ags/labels.csv con sus parches y los
JSON de resultados. Dame los comandos de gh release create; no publiques sin mi confirmación.
```
```
/protocol-check G5
```

---

## Diario
```
/sprint-status
```

---

## Segunda etapa (en paralelo, rama stage2/*)
```
Crea un worktree en la rama stage2/app. Código, comentarios y documentación en inglés.
Primero genera data/mock con la malla, predicciones e indicadores falsos de los 11 municipios ×
117 meses. Luego usa el subagente api-builder para construir el backend con esos datos y, cuando
exporte backend/openapi.json, usa el subagente frontend-builder para el frontend bilingüe
"Hydroscope Aguascalientes". Abre un PR por componente. No toques archivos fuera de backend/,
frontend/ y data/mock/.
```
