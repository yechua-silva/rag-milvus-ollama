# 🤖 Sistema RAG (Retrieval-Augmented Generation) con Milvus y Ollama

Un sistema de chat inteligente que permite hacer preguntas sobre documentos PDF utilizando técnicas de recuperación aumentada de generación (RAG). El sistema procesa documentos PDF, los divide en chunks inteligentes, genera embeddings y permite consultas en lenguaje natural.

## ✨ Características

- **Procesamiento inteligente de PDFs**: Extracción automática de texto de todos los PDFs en la carpeta 'docs'
- **Chunking adaptativo**: División inteligente del texto preservando el contexto
- **Base de datos vectorial**: Almacenamiento eficiente usando Milvus
- **Embeddings locales**: Generación de embeddings usando Ollama
- **Chat interactivo**: Interfaz de línea de comandos para consultas
- **Configuración flexible**: Variables de entorno para personalización

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.8+
- Docker y Docker Compose
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/rag-milvus-ollama.git
cd rag-milvus-ollama
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

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

Pregunta: ¿Qué es la programación orientada a objetos?

Respuesta:
La programación orientada a objetos es un paradigma de programación que se basa en el concepto de "objetos", los cuales pueden contener datos (atributos) y código (métodos). Este paradigma permite crear software más modular, reutilizable y fácil de mantener...

[Fuente: logica_programacion.pdf, Pagina: 15]
```

## 🛠️ Desarrollo

### Estructura del proyecto

```
├── src/
│   ├── infrastructure/
│   │   ├── document_loader.py      # Carga de múltiples PDFs
│   │   ├── text_processor.py       # Limpieza y chunking
│   │   ├── embedding_manager.py    # Generación de embeddings
│   │   └── vector_store_manager.py # Gestión de Milvus
│   └── application/
│       ├── ingestion_service.py    # Orquestación de ingesta
│       └── chat_service.py         # Servicio de chat
├── docs/                           # Carpeta para documentos PDF
├── config.py                       # Configuración
├── main.py                         # Punto de entrada
├── docker-compose.yml              # Servicios de Docker
└── requirements.txt                # Dependencias Python
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

## 🚀 Mejoras Futuras

- [ ] Soporte para múltiples tipos de documentos (Word, HTML, etc.)
- [ ] Interfaz web con Streamlit/Gradio
- [ ] API REST con FastAPI
- [ ] Integración con bases de datos relacionales
- [ ] Sistema de autenticación
- [ ] Métricas de relevancia y evaluación
- [ ] Cache de embeddings
- [ ] Procesamiento distribuido

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
