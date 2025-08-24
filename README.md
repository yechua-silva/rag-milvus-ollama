# 🤖 Sistema RAG con Arquitectura Limpia, Milvus y Aceleración por GPU

Este es un sistema de chat inteligente y escalable que permite hacer preguntas sobre documentos PDF. Utiliza el patrón de **Arquitectura Limpia** para un diseño modular y soporta **aceleración por GPU** para optimizar el rendimiento del proceso de ingesta de datos.

---

## ✨ Características Principales

- ✅ **Procesamiento de Documentos Flexible**: Extracción de texto de múltiples archivos PDF en una carpeta dedicada.
- ✅ **Chunking Adaptativo**: División inteligente del texto que preserva el contexto estructural del documento.
- ✅ **Base de Datos Vectorial Robusta**: Almacenamiento y búsqueda eficiente de embeddings con **Milvus**.
- ✅ **Procesamiento Paralelo de Embeddings**: Generación de embeddings de forma acelerada usando la **GPU** (vía ONNX Runtime + DirectML/CUDA) o recurriendo a la **CPU** con **Ollama**.
- ✅ **Arquitectura Limpia**: Diseño modular y escalable siguiendo principios de la Programación Orientada a Objetos (POO), ideal para proyectos de nivel profesional.
- ✅ **Chat Interactivo**: Interfaz de línea de comandos para consultas en lenguaje natural sobre el contenido de los documentos.

---

## 🚀 Instalación Rápida

### Prerrequisitos

- **Python 3.8+**
- **Docker** y **Docker Compose** (necesarios para Milvus)
- **Git**

### 1. Clonar el repositorio

````bash
git clone [https://github.com/tu-usuario/rag-milvus-ollama.git](https://github.com/tu-usuario/rag-milvus-ollama.git)
cd rag-milvus-ollama
### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
````

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Ollama

#### Windows

1. Descargar desde [ollama.ai](https://ollama.ai/download)
2. Ejecutar el instalador
3. Verificar instalación: `ollama --version`

#### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### macOS

```bash
brew install ollama
```

### 5. Descargar modelos necesarios

```bash
# Modelo de embeddings (requerido)
ollama pull mxbai-embed-large

# Modelo de lenguaje para chat (requerido)
ollama pull qwen2.5:3b

# Modelos alternativos (opcional)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 6. Iniciar Milvus

```bash
docker-compose up -d
```

Verificar que Milvus esté funcionando:

```bash
curl http://localhost:19530/health
```

## 📖 Uso

### Configuración

Modifica el archivo `.env` (opcional) o modifica `config.py`:

**Configuracion de Acelarion (GPU/CPU)**
El sistema está diseñado para detectar y usar automáticamente tu GPU.

- Para tarjetas AMD (DirectML) / NVIDIA (CUDA): El sistema intentará usar la GPU por defecto. Asegúrate de tener los drivers más recientes instalados.

- Para forzar el uso de la CPU: En config.py, cambia la siguiente línea:

```py
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
```

```env
# Rutas (usar DOCS_FOLDER para múltiples archivos)
DOCS_FOLDER=./docs

# Milvus
MILVUS_URI=http://127.0.0.1:19530
COLLECTION_NAME=knowledge_base

# Modelos
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=qwen2.5:3b

# Procesamiento de texto
CHUNK_SIZE=800
CHUNK_OVERLAP=100
SEARCH_TOP_K=5
```

### Ingesta de documentos

Coloca tu archivo PDF en el directorio del proyecto y ejecuta:

```bash
python main.py --ingest
```

### Iniciar chat interactivo

```bash
python main.py
```

### Ejemplo de uso

```
Sistema de Chat RAG listo. Escribe 'salir' para terminar.

Pregunta: ¿Quién es el autor del libro y de qué se trata la lógica de programación?

Respuesta:
El autor del libro "Lógica de Programación" es Omar Iván Trejos Buriticá. La lógica de programación es la unión de conceptos sencillos para el diseño de soluciones lógicas, que nos permiten diseñar soluciones a problemas que pueden ser implementados a través de un computador. Se basa en un conjunto de normas técnicas que permiten desarrollar un algoritmo entendible para la solución de un problema.
```

## 🛠️ Desarrollo

### Estructura del proyecto

```
/tu_proyecto
|-- src/
|   |-- domain/                         # Modelos de datos puros (e.g., DocumentChunk)
|   |-- application/                    # Lógica de negocio y orquestación
|   |   |-- chat_service.py
|   |   ├── ingestion_orchestrator.py   # Orquestador para ingesta paralela con GPU
|   |   ├── orchestrator.py             # Orquestador general (ingesta/chat)
|   |   └── interfaces.py               # Definición de interfaces (contratos)
|   ├── infrastructure/                 # Implementaciones concretas y dependencias
|   │   ├── document_loader.py          # Carga de PDFs
|   │   ├── embedding_gpu.py            # Generador de embeddings con GPU (ONNX)
|   │   ├── embedding_manager.py        # Generador de embeddings con CPU (Ollama)
|   │   ├── text_processor.py           # Lógica de chunking y limpieza de texto
|   │   └── vector_store_manager.py     # Lógica de Milvus
|   └── main.py                         # Punto de entrada y configuración
├── docs/                               # Carpeta para documentos PDF
├── models/                             # Modelos ONNX generados automáticamente
├── config.py                           # Configuración centralizada
├── docker-compose.yml                  # Configuración de Docker para Milvus
└── requirements.txt                    # Dependencias del proyecto
```

### Arquitectura Clean

El proyecto sigue principios de **Clean Architecture**:

- **Infrastructure**: Adaptadores externos (Milvus, Ollama, archivos)
- **Application**: Casos de uso y lógica de negocio
- **Domain**: Modelos y entidades (models.py)

### Añadir nuevos tipos de documentos

Para soportar otros formatos:

1. Crear nuevo loader en `infrastructure/`
2. Implementar interfaz similar a `PdfDocumentLoader`
3. Registrar en `main.py`

## 📊 Monitoreo

### Ver estadísticas de Milvus

```python
from src.infrastructure.vector_store_manager import MilvusManager

vector_store = MilvusManager("http://localhost:19530", "collection_name", 1024)
stats = vector_store.get_stats()
print(stats)
```

### Interfaz web de Milvus

Accede a `http://localhost:9001` (MinIO) para gestión de archivos.

## 🔧 Solución de Problemas

### Error: "GPU no detectada" o DmlExecutionProvider not found

**Causa Principal:** onnxruntime no puede comunicarse con tu GPU.

**Solución:**

1. Actualiza los drivers de tu GPU a la última versión disponible (Adrenalin para AMD, Game Ready/Studio para NVIDIA).

2. Realiza una reinstalación limpia de las librerías de ONNX:

```bash
pip uninstall onnx onnxruntime onnxruntime-directml -y
pip install onnx onnxruntime-directml
```

### Error: "Ollama not found"

```bash
# Verificar instalación
ollama --version

# Iniciar servicio (Linux/macOS)
sudo systemctl start ollama
```

### Error: "Connection refused" (Milvus)

```bash
# Verificar contenedores
docker-compose ps

# Reiniciar servicios
docker-compose down && docker-compose up -d
```

### Error: "Modelo no encontrado"

```bash
# Listar modelos instalados
ollama list

# Instalar modelo faltante
ollama pull mxbai-embed-large
```

### Error: "No se encontraron archivos PDF"

Asegurate de que la carpeta `docs` exista y contenga archivos PDF:

```bash
mkdir docs
```

##

## 🚀 Mejoras Futuras

- [ ] Implementar un HierarchicalChunker más avanzado.

- [ ] Incorporar un Reranker para mejorar la precisión de las respuestas.

- [ ] Soportar múltiples tipos de documentos (HTML, Word, etc.).

- [ ] Añadir una interfaz web con Streamlit o Gradio.

- [ ] Desarrollar una API REST con FastAPI.

- [ ] Integrar métricas de evaluación de relevancia.

- [ ] Desarrollar un sistema de memoria de conversación.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Milvus](https://milvus.io/) - Base de datos vectorial
- [Ollama](https://ollama.ai/) - Modelos LLM locales
- [PyMuPDF](https://pymupdf.readthedocs.io/) - Procesamiento de PDFs

## 📞 Contacto

**[Yechua Silva]** - [yechua_silva@outlook.cl]

**[Yechua Linkedin]** - [[Linkedin](https://www.linkedin.com/in/yechua-silva/)]

Proyecto: [Github Rag Milvus Ollama](https://github.com/yechua-silva/rag-milvus-ollama)

---

⭐ Si este proyecto te resulta útil, ¡considera darle una estrella!
