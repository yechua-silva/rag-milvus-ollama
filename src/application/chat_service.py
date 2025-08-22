import ollama
from src.application.interfaces import ResponseGenerator


class ChatService(ResponseGenerator):
    """Orquestar flujo de chat: Buscar -> Generar respuestas"""

    def __init__(self, embedder, vector_store, llm_model, top_k):
        """_summary_

        Args:
            embedder (_type_): _description_
            vector_store (_type_): _description_
            llm_moder_ (_type_): _description_
            top_k (_type_): _description_
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k

    def _generate_prompt(self, context: str, question: str) -> str:
        return f"""
        Eres un asistente especializado que SOLO puede responder basándose en la
        información proporcionada en los documentos.

        REGLAS ESTRICTAS:
        1. Responde ÚNICAMENTE con información encontrada en el contexto proporcionado
        2. Si la información no está en el contexto, di EXACTAMENTE:
        "No tengo información sobre esto en los documentos disponibles"
        3. NO inventes nombres, autores, títulos o cualquier información que no esté explícitamente en el contexto
        4. Cuando sea posible, cita la fuente específica (documento y página) de tu información
        5. Sé preciso y conciso en tus respuestas

        CONTEXTO DE LOS DOCUMENTOS:
        {context}

        PREGUNTA DEL USUARIO:
        {question}

        RESPUESTA BASADA EXCLUSIVAMENTE EN LOS DOCUMENTOS:"""

    def ask(self, question: str):
        """_summary_

        Args:
            question (str): _description_
        """

        question_embedding = self.embedder.get_embedding(question)
        results = self.vector_store.search(question_embedding, self.top_k)
        if not results:
            return "No encontré información relevante."
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Fuente: {result['metadata']['source']}, Página: {result['metadata']['page']}]\n"
                f"{result['text']}\n"
            )

        context = "\n---\n".join(context_parts)

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
