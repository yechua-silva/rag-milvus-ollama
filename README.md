# 🤖 Sistema RAG con Milvus y Aceleración GPU

Sistema de chat inteligente que permite consultas sobre documentos PDF utilizando arquitectura limpia, base de datos vectorial Milvus y aceleración por GPU.

## ✨ Características

- **Procesamiento GPU/CPU**: Aceleración automática con DirectML (AMD) o CUDA (NVIDIA)
- **Base de Datos Vectorial**: Búsqueda eficiente con Milvus
- **Arquitectura Limpia**: Diseño modular y escalable siguiendo principios SOLID
- **Chunking Inteligente**: División contextual que preserva la coherencia del documento
- **Chat Interactivo**: Interfaz CLI para consultas en lenguaje natural

## 🚀 Instalación

### Prerrequisitos

- Python 3.8+
- Docker y Docker Compose
- Git

### Configuración

```bash
# Clonar repositorio
git clone https://github.com/yechua-silva/rag-milvus-ollama.git
cd rag-milvus-ollama

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh  # Linux
# Descargar desde https://ollama.ai para Windows/macOS

# Descargar modelos
ollama pull mxbai-embed-large
ollama pull qwen2.5:3b

# Iniciar Milvus
docker-compose up -d
```

## 📖 Uso

### Configuración (Opcional)

Modifica `config.py` o usa variables de entorno:

```python
# Aceleración GPU (automática)
USE_GPU = True  # False para forzar CPU

# Rutas y modelos
DOCS_FOLDER = "./docs"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "qwen2.5:3b"

# Parámetros de procesamiento
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEARCH_TOP_K = 5
```

### Ingesta de Documentos

```bash
# Colocar archivos PDF en ./docs/
python main.py --ingest
```

### Chat Interactivo

```bash
python main.py
```

### Ejemplo de Uso

```
Pregunta: ¿Cuál es el tema principal del documento?

Respuesta: El documento trata sobre lógica de programación,
escrito por Omar Iván Trejos Buriticá. Se enfoca en conceptos
fundamentales para el diseño de soluciones algorítmicas...
[Fuente: logica_programacion.pdf, Página: 1]
```

## 🏗️ Arquitectura

```
src/
├── domain/          # Modelos de datos
├── application/     # Lógica de negocio
├── infrastructure/  # Implementaciones concretas
└── main.py         # Punto de entrada
```

### Componentes Principales

- **DocumentLoader**: Extracción de texto de PDFs
- **TextProcessor**: Limpieza y chunking inteligente
- **EmbeddingGenerator**: GPU (ONNX) o CPU (Ollama)
- **VectorStore**: Gestión de Milvus
- **Orchestrator**: Coordinación del flujo completo

## 🔧 Solución de Problemas

### GPU no detectada

```bash
# Actualizar drivers y reinstalar ONNX
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-directml
```

### Milvus no conecta

```bash
# Verificar contenedores
docker-compose ps
curl http://localhost:19530/health

# Reiniciar servicios
docker-compose down && docker-compose up -d
```

### Modelos no encontrados

```bash
ollama list  # Verificar modelos instalados
ollama pull mxbai-embed-large  # Reinstalar si es necesario
```

## ⚙️ Configuración Avanzada

### Modelos Alternativos

```bash
# Embeddings
ollama pull nomic-embed-text

# LLM
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

### Variables de Entorno

```env
MILVUS_URI=http://127.0.0.1:19530
COLLECTION_NAME=knowledge_base
EMBEDDING_BATCH_SIZE=64
NUM_WORKERS=4
```

## 📊 Rendimiento

- **GPU**: ~500 embeddings/segundo (AMD RX 6600)
- **CPU**: ~50 embeddings/segundo (Ollama)
- **Memoria**: ~2GB para modelos 3B
- **Almacenamiento**: ~1GB por cada 1000 páginas

## 🛠️ Desarrollo

### Extensiones

Para añadir nuevos formatos de documento:

1. Implementar nueva clase en `infrastructure/`
2. Extender interfaz `DocumentLoader`
3. Registrar en `main.py`

### Testing

```bash
python -m pytest tests/
```

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## 🙏 Créditos

- [Milvus](https://milvus.io/) - Base de datos vectorial
- [Ollama](https://ollama.ai/) - LLMs locales
- [ONNX Runtime](https://onnxruntime.ai/) - Inferencia optimizada

## 📞 Contacto

**Yechua Silva**  
📧 yechua_silva@outlook.cl  
💼 [LinkedIn](https://www.linkedin.com/in/yechua-silva/)  
🐙 [GitHub](https://github.com/yechua-silva/rag-milvus-ollama)

---

⭐ **¿Te resulta útil?** ¡Dale una estrella al proyecto!
