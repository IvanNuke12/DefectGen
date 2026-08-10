# Recorta · Etiqueta · Fusiona — HMI por proyectos y ventanas

Interfaz web local para seleccionar parches sin defecto, inspeccionar
etiquetas y fusionar crops editados. Sin popups de OpenCV: pantalla de
proyectos, galería de miniaturas, tres ventanas (tabs) y registro de
recortes exportable (CSV/JSON).

## 1. Instalación

Necesitas Python 3.9+ instalado en Windows. Abre una terminal (CMD o PowerShell)
en esta carpeta y ejecuta:

```bash
pip install -r requirements.txt
```

## 2. Ejecución

```bash
python app.py
```

Se abrirá automáticamente tu navegador en `http://127.0.0.1:5000`. Si no se abre
solo, entra a esa dirección manualmente. La ventana de la terminal debe quedar
abierta mientras usas la herramienta; ciérrala (Ctrl+C) para detener el servidor.

### Ejecución con Docker

```bash
docker build -t defect-crop-hmi .
docker run -d -p 5000:5000 \
  -e STATE_DIR=/app/state -e PROJECTS_ROOTS="/data/projects" \
  -v crop-hmi-state:/app/state \
  -v /ruta/a/tus/proyectos:/data/projects \
  defect-crop-hmi
```

- `STATE_DIR`: dónde se guardan `config.json`, `projects_recent.json` y `.thumb_cache/` (persistir con un volumen).
- `PROJECTS_ROOTS`: raíces visibles en el navegador de carpetas (separadas por `;`).
- Variables de entorno: `HOST` (por defecto `0.0.0.0` en el contenedor), `PORT` (5000), `OPEN_BROWSER` (`0` en el contenedor; solo abre navegador en localhost).

> El selector de carpetas ahora es 100% web (`/api/browse`): ya no depende de
> diálogos tkinter en el servidor, por lo que funciona en contenedores headless.

## 3. Proyectos

Al arrancar verás la **pantalla de proyectos**. La aplicación siempre pregunta
qué proyecto quieres usar; no recuerda el último automáticamente.

Un **proyecto** es una carpeta con esta estructura:

```
mi_proyecto/
├── images/                # tus imágenes de entrada (se crea automáticamente)
├── labels/                # tus labels (YOLO .txt, se crea automáticamente)
├── *.json                 # anotaciones COCO, si existen (opcional)
└── crops/                 # todos los recortes del proyecto
    ├── crops_log.csv
    ├── images/            # los parches <imagen>_crop_<n>.<ext>
    └── masks/             # las máscaras <imagen>_mask_<n>.png
```

- **Crear**: se crea una carpeta nueva con sus subcarpetas `images/`, `labels/`
  y `crops/` automáticamente. Puedes indicar (opcional) una **carpeta de origen
  de imágenes** y una **carpeta de origen de labels**: sus archivos se copian al
  proyecto en el momento de crearlo (`images/` copia solo imágenes soportadas;
  `labels/` copia `.txt`/`.json`).
- **Abrir**: selecciona una carpeta de proyecto existente. Se detectan las
  anotaciones automáticamente: un JSON COCO en la raíz (`annotations.json`,
  `instances_default.json`, etc.) o la subcarpeta `labels/` como YOLO.
- **Recientes**: los últimos proyectos abiertos quedan a un clic.
- Cuando hay un proyecto activo, el HMI muestra su nombre y el botón
  **Proyectos** (arriba a la derecha) para cerrarlo y volver a la selección.
- Dentro del HMI también puedes **importar** carpetas de origen en cualquier
  momento: botón **Importar imágenes…** (panel Recortar) y **Importar
  labels…** (panel Etiquetado). Los archivos se copian a `images/` / `labels/`
  del proyecto activo y la galería/anotaciones se refrescan.

Todos los recortes de un proyecto se acumulan en su carpeta `crops/`, sin
separar por fechas: el registro `crops_log.csv` contiene todo el historial de
recortes de ese proyecto.

## 4. Uso

### Ventana Recortar (panel lateral: Proyecto + Tamaño + Recortes)

1. **Proyecto**: muestra el nombre y la carpeta `images/` activa (no editable).
2. **Tamaño de recorte**: ajusta el slider o escribe el valor directamente en el
   campo numérico (cuadrado centrado en el clic). El máximo se adapta
   automáticamente al tamaño de cada imagen.
3. Mueve el cursor sobre la imagen: el recuadro azul con esquinas de mira sigue
   al puntero y muestra posición y tamaño en tiempo real. Haz **clic** para
   guardar el recorte y pasar a la siguiente imagen.
4. Atajos: `→`/`←` navegar, `espacio` saltar imagen.
5. **Exportar CSV / JSON**: descarga el registro de recortes del proyecto.

### Ventana Etiquetado (panel lateral: Anotaciones + Visualización)

Exclusiva para cargar anotaciones y ver máscaras. Nada relacionado con el crop.

1. El formato se detecta solo (YOLO si el proyecto tiene `labels/`, COCO si hay
   JSON en la raíz). Puedes forzar formato (auto / COCO JSON / YOLO detection) o
   importar una carpeta de labels con **Importar labels…**.
2. Pulsa **Aplicar y recargar**. La máscara aparece en rojo sobre la imagen
   seleccionada en la galería; puedes ocultarla o ajustar su opacidad.
3. Si el etiquetado está activo, los recortes de la ventana **Recortar**
   también guardan la máscara binaria alineada en `salida/masks`.

### Ventana Fusionar (a ancho completo, sin panel lateral)

Incrusta un crop editado externamente en la imagen original:

1. Selecciona en el desplegable el crop registrado (se cargan desde el log
   del proyecto).
2. Sube el archivo editado (PNG/JPG/BMP/WEBP). Si sus dimensiones no coinciden
   con el crop original, se redimensiona automáticamente al tamaño registrado.
3. Pulsa **Insertar**. Se genera `<imagen>_merged_<crop><ext>` en la carpeta de
   entrada (sin sobrescribir el original) y se muestra el resultado, con enlace
   de descarga.

## 6. Notas técnicas

- Los recortes se guardan con nombre único `<imagen>_crop_<n>.<ext>` (n
  incremental por imagen y proyecto). Nunca se sobrescriben crops anteriores.
- El registro del proyecto está en `<proyecto>/crops/crops_log.csv`, con
  archivo, `crop_name`, `crop_index`, coordenadas (`x1,y1,x2,y2`), tamaño y hora.
- Las miniaturas se generan bajo demanda y se cachean en `.thumb_cache/`
  (puedes borrar esa carpeta sin problema; se regenera sola).
- `config.json` guarda los defaults de UI (tamaño de recorte, etiquetado y
  opacidad). Las carpetas de entrada/salida ya no viven ahí: las gestiona el
  proyecto activo.
- COCO admite polígonos, RLE y `bbox` como respaldo. YOLO detection convierte
  cada bounding box normalizada en una región blanca de la máscara.
- Cuando el etiquetado está activo, incluso las imágenes sin objetos generan
  una máscara negra; así pueden usarse como muestras negativas de segmentación.
- El fusionado es pixel-perfect: si el crop editado ya tiene el tamaño correcto
  se pega sin redimensionar (resample solo con NEAREST), evitando cualquier
  desplazamiento de subpíxel.
- Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

## 7. Estructura del proyecto

```
crop-hmi/
├── app.py                 # Backend Flask (API + servidor + gestión de proyectos)
├── annotation_masks.py    # Conversión COCO/YOLO a máscaras binarias
├── requirements.txt
├── config.json            # Defaults de UI (tamaño, etiquetado, opacidad)
├── projects_recent.json   # Recientes de proyectos (se crea automáticamente)
├── templates/
│   ├── projects.html      # Pantalla de selección/creación de proyectos
│   └── index.html         # HMI principal (Recortar · Etiquetado · Fusionar)
├── static/
│   ├── css/
│   │   ├── projects.css   # Estilos de la pantalla de proyectos
│   │   └── style.css      # Estilos del HMI principal
│   └── js/
│       ├── common.js        # Utilidades compartidas + bus de eventos
│       ├── projects.js      # Abrir/crear proyecto + recientes
│       ├── app.js           # Orquestador: proyecto, config, tabs
│       ├── crop_app.js      # Ventana Recortar (galería + retícula + crop)
│       ├── labels.js        # Ventana Etiquetado (preview de máscara)
│       └── merge.js         # Ventana Fusionar (subir crop editado + incrustar)
└── tests/
    └── test_annotation_flow.py
```
