from src.application.interfaces import DocumentLoader, TextProcessor, Chunker, Embedder, VectorStore


class Orchestrator:
    """Coordina el flujo de trabajo entre todos los componentes del sistema RAG"""

    def __init__(
        self,
        loader: DocumentLoader,
        text_processor: TextProcessor,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        llm_model: str,
        search_top_k: int,
    ):
        self.loader = loader
        self.text_processor = text_processor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.search_top_k = search_top_k

    def ingest_documents(self):
        """Ejecuta el proceso completo de ingesta de documentos"""
        pages = self.loader.load()
        print(f"Páginas cargadas: {len(pages)}")

        chunks = self.chunker.chunk(pages)
        print(f"Chunks creados: {len(chunks)}")

        self.vector_store.set_collection()

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=15)

        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append(
                {
                    "id": chunk["id"],
                    "vector": embeddings[i],
                    "text": chunk["text"][:1000],
                    "page": chunk["page"],
                    "source": chunk["source"],
                }
            )

        self.vector_store.insert(data_to_insert, batch_size=100)

        stats = self.vector_store.get_stats()
        print(f"Estadísticas de la colección: {stats}")

    def ask_question(self, question: str) -> str:
        """Procesa una pregunta y genera una respuesta"""
        question_embedding = self.embedder.get_embedding(question)

        results = self.vector_store.search(question_embedding, self.search_top_k)

        # Generar respuesta
        if not results:
            return "No encontré información relevante."

        # Formatear contexto
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Fuente: {result['metadata']['source']}, Página: {result['metadata']['page']}]\n"
                f"{result['text']}\n"
            )

        context = "\n---\n".join(context_parts)

        # Generar prompt y respuesta
        import ollama

        prompt = f"""Eres un asistente especializado que responde preguntas
        basándose únicamente en el contexto proporcionado.

        CONTEXTO:
        {context}

        PREGUNTA: {question}

        INSTRUCCIONES:
        1. Responde únicamente con información del contexto proporcionado
        2. Si no hay información suficiente, di que no sabes
        3. Cita las fuentes (documento y página) cuando sea relevante
        4. Sé preciso y conciso

        RESPUESTA:"""

        response = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
