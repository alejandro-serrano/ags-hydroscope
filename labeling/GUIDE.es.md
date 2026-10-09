# Guía de etiquetado (v2, 2026-10-09)

Traducción al español de [`GUIDE.md`](GUIDE.md). Si las dos versiones difieren, manda la versión
en inglés. Los nombres de las clases y de los botones (**skip**, **back**) se dejan en inglés
porque son los que aparecen en el notebook y en `labels.csv`.

**Cómo abrirlo:** `conda activate hydroscope && jupyter lab labeling/label.ipynb`, ejecuta todas
las celdas, escribe tu id de anotador y presiona Start. Cada clic se guarda al instante; Start
retoma donde te quedaste. Las etiquetas hechas con la guía v1 están archivadas en
`data/ags/labels_v1_2026-10-09.csv`.

## Contexto: cómo se ve Aguascalientes en abril

- Clima semiárido, unos 510–530 mm de lluvia al año, concentrada de junio a octubre.
  **Abril es lo más fuerte de la temporada seca**, así que casi todo el estado se ve beige o café
  en la vista previa.
- **Verde en abril significa agua:** cultivos de riego (alfalfa, maíz forrajero, avena),
  huertas, el bosque de encino y pino de las sierras del poniente, o vegetación junto a canales y
  presas.
- Alrededor de 70 000 ha de temporal están **desnudas o en descanso en abril**; se siembran hasta
  que empiezan las lluvias.
- La cubierta natural es sobre todo pastizal seco y matorral xerófilo (matorral crasicaule,
  huizache, nopal silvestre) en llanos y cerros, y bosque de encino/pino en la Sierra Fría y las
  sierras del poniente.
- **La vista previa (abril de 2024) es la evidencia principal.** El mapa base de Esri es imagen
  reciente, muchas veces de otra temporada: úsalo solo para reconocer objetos (surcos, cercas,
  techos, caminos).

## Las 10 clases (nombres de EuroSAT, sin cambios)

| Clase | Etiquétala cuando veas | No es esta clase |
| --- | --- | --- |
| AnnualCrop | Un patrón de parcelas: linderos rectos, surcos, tonos uniformes por parcela. Incluye **parcelas de temporal desnudas o en descanso**, maíz forrajero y avena de riego, **alfalfa** y pivotes centrales. | Hileras de árboles o vides (PermanentCrop). |
| PermanentCrop | **Solo huertas y viñedos en hileras**: guayaba (sobre todo Calvillo), uva, durazno, y nopal o maguey plantados en hileras. Puntos regulares de copas de árboles. | Parcelas verdes uniformes sin hileras visibles (AnnualCrop). |
| Pasture | Pasto **manejado para pastoreo**: potreros cercados, cubierta de pasto uniforme, formas regulares, a menudo con bebedero o bordo, sin surcos. | Pasto sin señales de manejo (HerbaceousVegetation). |
| HerbaceousVegetation | Pastizal seco natural, **matorral xerófilo** (crasicaule, huizache, nopal silvestre), encinar abierto donde el pasto cubre la mayor parte del suelo, incluidos cerros pedregosos con matorral ralo. Textura irregular, en manchones. | Parcelas con linderos rectos (AnnualCrop o Pasture). |
| Forest | **Dosel continuo** de encino o pino en al menos 70 % de la celda (Sierra Fría, sierras del poniente). Verde oscuro incluso en abril. | Árboles dispersos sobre pasto (HerbaceousVegetation). |
| Residential | Casas y techos pequeños densos con traza de calles, incluidos fraccionamientos nuevos con las calles ya trazadas. | Techos y patios grandes (Industrial). |
| Industrial | Naves y techos de fábricas grandes, parques industriales, patios grandes de estacionamiento y carga. | Casas (Residential). |
| Highway | Una carretera principal (federales, libramientos, anillos periféricos) que cruza la celda y domina la imagen. | Calles urbanas dentro de una colonia (Residential). |
| River | Un cauce que domina la celda, con o sin agua, reconocible por su vegetación ribereña o su lecho arenoso. Poco común. | Una zanja o canal pequeño entre parcelas (AnnualCrop). |
| SeaLake | **Agua que cubre al menos 70 % de la celda** en la vista previa de abril: presas y bordos. | Una presa cuya agua se retiró: etiqueta la cubierta visible, o skip. |

## Reglas

1. Etiqueta una celda solo si **al menos 70 %** de ella es de una sola clase. Excepción: Highway y
   River se etiquetan por el elemento dominante, como en EuroSAT.
2. Desempates para las confusiones frecuentes del semiárido:
   - Patrón de parcelas (linderos rectos, surcos), verde o desnudo → **AnnualCrop**.
   - Árboles o vides **en hileras** → **PermanentCrop**; nada más es PermanentCrop.
   - Pasto con señales claras de manejo para pastoreo → **Pasture**; sin ellas →
     **HerbaceousVegetation**.
   - Dosel continuo en más de 70 % de la celda → **Forest**; si no →
     **HerbaceousVegetation**.
3. **Cerros y terreno pedregoso:** el relieve no es una clase; etiqueta la cubierta que tiene
   encima. Matorral o pasto con roca o suelo visible entre las plantas →
   **HerbaceousVegetation**; dosel continuo de encino/pino → **Forest**; **skip** solo cuando la
   roca o el suelo desnudo cubren la mayor parte de la celda (≥ 70 %), o en canteras y bancos de
   material.
4. Presiona **skip** para cubiertas que no tienen clase en EuroSAT: invernaderos, canteras y
   minas, aeropuertos, campos de golf, rellenos sanitarios, superficies casi sin vegetación
   (afloramientos de roca desnuda, cárcavas, suelo desnudo sin patrón de parcelas), y cualquier
   celda en la que dudes.
5. **back** regresa a la celda anterior; hacer clic de nuevo en una clase reemplaza tu etiqueta
   anterior.

## Ejemplos (llénalos mientras etiquetas)

| Clase | cell_id 1 | cell_id 2 |
| --- | --- | --- |
| AnnualCrop | | |
| Forest | | |
| HerbaceousVegetation | | |
| Highway | | |
| Industrial | | |
| Pasture | | |
| PermanentCrop | | |
| Residential | | |
| River | | |
| SeaLake | | |
