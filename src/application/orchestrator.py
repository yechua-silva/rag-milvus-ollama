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
        print("1. Generando embedding para la pregunta...")
        question_embedding = self.embedder.get_embedding(question)

        print("2. Buscando en la base de conocimiento...")
        results = self.vector_store.search(question_embedding, self.search_top_k)

        if not results:
            return "No encontré información relevante en los documentos para responder a esta pregunta."

        # Formatear el contexto de una manera muy clara para el LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "desconocida")
            page = result["metadata"].get("page", "?")
            context_parts.append(f"--- Fuente {i} (Documento: {source}, Página: {page}) ---\n" f"{result['text']}\n")
        context = "\n".join(context_parts)

        # **PROMPT MEJORADO Y MÁS ESTRICTO**
        prompt = f"""
        **Tu Tarea:** Eres un asistente experto que responde preguntas basándose
        EXCLUSIVAMENTE en el contexto proporcionado de los documentos.

        **REGLAS ESTRICTAS E INQUEBRANTABLES:**
        1.  **NO PUEDES** usar ningún conocimiento externo.
            Tu única fuente de verdad es el texto en la sección "CONTEXTO DE LOS DOCUMENTOS".
        2.  Lee el CONTEXTO cuidadosamente y extrae de él la información necesaria para responder
            la PREGUNTA DEL USUARIO.
        3.  Si la respuesta se encuentra en el contexto, formúlala con tus propias palabras,
            siendo claro y conciso.
        4.  **Cita tus fuentes OBLIGATORIAMENTE.** Después de cada pieza de información,
            debes añadir la cita correspondiente, por ejemplo: [Fuente: nombre_del_archivo.pdf, Página: X].
        5.  Si después de leer todo el contexto, la información para responder la pregunta no se encuentra,
            debes responder **EXACTAMENTE** con la frase:
            "La información necesaria para responder a esta pregunta no se encuentra en
            los documentos proporcionados." No intentes adivinar.

        **CONTEXTO DE LOS DOCUMENTOS:**
        {context}

        **PREGUNTA DEL USUARIO:**
        {question}

        **RESPUESTA (basada únicamente en el contexto y citando las fuentes):**
        """

        print("3. Generando respuesta con el LLM...")
        import ollama

        response = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
