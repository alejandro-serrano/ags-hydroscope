# Prompts listos por tarea

Cópialos en Claude Code (terminal o pestaña Code de Claude Desktop) desde la raíz del repo.
Para tareas grandes, empieza en modo Plan (Shift+Tab en la terminal o el selector de modo en Desktop),
revisa el plan y luego cambia a "Accept edits".

## Sprint 1 (5–9 oct)

**F0-01 · Repositorio**
```
/bootstrap-repo
```

**F2-01 + F3-01/02 · EuroSAT, modelos y transformaciones**
```
Usa el subagente ml-engineer. Implementa F2-01, F3-01 y F3-02 del plan: dataset EuroSAT de TorchGeo
con su partición oficial y derivación RGB (B04, B03, B02), stats.json por banda calculado solo con
train, la fábrica de modelos para 3 o 13 bandas y las transformaciones comunes. Agrega pruebas de
forma y de determinismo. No lances entrenamientos largos.
```

**F3-03 · Ciclo de entrenamiento y evaluación**
```
Usa el subagente ml-engineer. Implementa F3-03: un solo ciclo de entrenamiento y evaluación con
config YAML que respete docs/protocol.md y escriba el JSON del contrato. Verifícalo con
configs/smoke.yaml en CPU y muéstrame el JSON resultante.
```

**F3-04/05 · Corridas**
```
/new-experiment resnet50 rgb imagenet
/new-experiment resnet50 ms13 imagenet
/new-experiment efficientnet_b0 rgb imagenet
/new-experiment efficientnet_b0 ms13 imagenet
/new-experiment vit_small rgb imagenet
/new-experiment vit_small ms13 imagenet
```

**F2-02/03 · Earth Engine y parches**
```
Usa el subagente geo-data-engineer. Implementa F2-02 y F2-03: exportación de la mediana de abril 2024
de COPERNICUS/S2_HARMONIZED para Aguascalientes, malla de 640 m en EPSG:32613 y corte de parches
13×64×64 con vista previa RGB. Compara los histogramas por banda contra EuroSAT y repórtame el
resultado.
```

**F2-04 · Herramienta de etiquetado**
```
Usa el subagente geo-data-engineer. Crea labeling/GUIDE.md (en inglés, una página, equivalencias
de las 10 clases de EuroSAT en el paisaje semiárido) y labeling/label.ipynb con ipywidgets que
muestre el PNG y 10 botones y guarde labels.csv según el contrato.
```

**F1 · Literatura**
```
Busca 8–10 trabajos sobre CNN y Vision Transformers en EuroSAT/Sentinel-2, SSL4EO-S12 y cambio de
dominio geográfico. Para cada uno: referencia BibTeX verificada, modelo, datos, métrica y hallazgo.
Escribe docs/literature.md y agrega las entradas a paper/references.bib. No inventes referencias:
si no puedes verificar un DOI, márcalo.
```

## Sprint 2 (12–16 oct)

**F4 · Generalización**
```
Usa el subagente ml-engineer. Evalúa sin ajuste los 8 checkpoints en el test de Aguascalientes y
luego ejecuta el ajuste fino en modos head y last_block (10 épocas, pérdida ponderada). Dame los
comandos de las corridas de más de 10 minutos para lanzarlas yo.
```

**F5 · Análisis y Grad-CAM**
```
/results-tables
```
```
Usa el subagente ml-engineer. Genera la figura de Grad-CAM (F5-03): 6 ejemplos por modelo,
EuroSAT y Aguascalientes, aciertos y errores, antes y después del ajuste fino.
```

**G3 · Resultados congelados**
```
/protocol-check G3
```

**F6 · Artículo**
```
/paper-section methodology
/paper-section results
/paper-section discussion
```

## Sprint 3 (19–21 oct)

```
/protocol-check G5
```
```
Revisa README.md como si fueras un compañero que nunca vio el proyecto: sigue las instrucciones paso
a paso en un entorno limpio y corrige todo lo que falle o sea ambiguo.
```
```
Con base en paper/ y paper/figures/, propón el guion de 12 diapositivas en inglés para la
presentación final, una idea por diapositiva y qué figura usa cada una.
```

## Diario

```
/sprint-status
```

## Segunda etapa (en paralelo, rama stage2/*)

```
Crea un worktree en la rama stage2/app. Primero genera data/mock con la malla, predicciones e
indicadores falsos de los 11 municipios × 117 meses, siguiendo los contratos de la segunda etapa del
plan. Luego usa el subagente api-builder para construir el backend con esos datos y, cuando exporte
openapi.json, usa el subagente frontend-builder para el
frontend bilingüe. Abre un PR por componente. No toques archivos fuera de backend/, frontend/ y
data/mock/.
```
