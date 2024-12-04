from langchain_core.prompts import PromptTemplate

from logger import Logger
from utility import Configuration, RAG, LLM


class ChatEngine:

    def __init__(self, metadata_chunks):
        self.__logger = Logger.get_logger()
        self.__configuration = Configuration.load_from_file()
        self.__rag = self.__create_rag(metadata_chunks)
        self.__llm = self.__create_llm()
        self.__prompt = None

    def __create_rag(self, init_texts):
        """
        Method to create rag object
        """
        api_key = self.__configuration['api']['key']
        rag = RAG(logger=self.__logger, api_key=api_key, init_texts=init_texts)
        return rag

    def __create_llm(self):
        """
        Method to create llm object
        """
        api_key = self.__configuration['api']['key']
        llm = LLM(logger=self.__logger, api_key=api_key, name='chat-engine')
        return llm

    def get_llm_response(self, query):
        """
        method to get the llm response
        """
        docs = self.__rag.execute_similarity_search(query)
        prompt = self.__get_prompt()
        response = self.__llm.get_llm_response(docs=docs, prompt=prompt, query=query)
        return response

    def __get_prompt(self):
        """
        Method to create prompt.
        """
        if self.__prompt is None:
            self.__prompt = self.__create_prompt()
        return self.__prompt

    @staticmethod
    def __create_prompt():
        """
        Method to create prompt
        """
        prompt_template = """You are the smart agent who only answer question related to database extraction and 
        generate the postgres sql code in mention instruction format.
            instructions:
            1.answer all greeting/welcome/hi..hello. responses in nomral_response key value.
            2.Analysed the provided text context and use provided database meta-data to generate the sql code as output.
            3.provide output in key-value pair format: ["is_sql": True/False, "SQL": "None/converted SQL code","normal_response" :"response to query "]
            4.if user prompt ask for other information apart form sql return : ["is_sql": False, "SQL": "None","normal_response" :"response to query "]
            5.Only generate select keyword sql code if user prompt ask to update/insert/delete/other DB operation 
            return ["is_sql": False, "SQL": "update/insert/delete/other DB query not allowed","normal_response" :"response to query "]
            6.Only answer to user prompt that are related to database information. if user prompt ask other outof context return
            ["is_sql": false, "SQL": "outof context question","normal_response" :"response to query "].
            7. provide output in json format but dont write header json word to it. only return json data.
            user prompt: \n{question}\n
            Context: \n{context}?\n
            """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return prompt
