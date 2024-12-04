from service import PromptEngine, ChatEngine


class Orchestrator:
    def __init__(self, metadata_chunks):
        """
        Method to initialize orchestrator
        """
        self.__chat_engine = ChatEngine(metadata_chunks)
        self.__prompt_engine = None

    def __get_prompt_engine(self):
        """
        Method to get prompt engine
        """
        if self.__prompt_engine is None:
            self.__prompt_engine = PromptEngine()
        return self.__prompt_engine

    def get_llm_response(self, query, enable_prompt_layer):
        """
        Method to get response from LLM
        """
        if enable_prompt_layer:
            prompt_engine = self.__get_prompt_engine()
            query = prompt_engine.generate_ai_assisted_query(query)
        response = self.__chat_engine.get_llm_response(query)
        return response
