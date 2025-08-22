class IngestionService:
    """Orquesta el flujo de ingesta: Cargar -> Chunkear -> Embeber -> Guardar."""

    def __init__(self, loader, chunker, embedder, vector_store):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self):
        """Ejecuta el proceso completo de ingesta"""
        # carga
        pages = self.loader.load()
        print(f"Páginas cargadas: {len(pages)}")

        # chunkear
        chunks = self.chunker.chunk(pages)
        print(f"Chunks creados: {len(chunks)}")

        # configurar DB
        self.vector_store.set_collection()

        # Preparar textos para embeddings
        texts = [chunk["text"] for chunk in chunks]

        # Generar embeddings en lotes (ya tiene su propia barra de progreso)
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=15)

        # Preparar datos para inserción
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append(
                {
                    "id": chunk["id"],
                    "vector": embeddings[i],
                    "text": chunk["text"][:1000],  # Limitar tamaño para evitar problemas
                    "page": chunk["page"],
                    "source": chunk["source"],
                }
            )

        # Insertar en Milvus con lotes
        self.vector_store.insert(data_to_insert, batch_size=100)

        # Obtener estadísticas
        stats = self.vector_store.get_stats()
        print(f"Estadísticas de la colección: {stats}")
