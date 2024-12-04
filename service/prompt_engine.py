from langchain_core.prompts import PromptTemplate

from logger import Logger
from utility import Configuration, LLM


class PromptEngine:

    def __init__(self, metadata_chunks):
        self.__logger = Logger.get_logger()
        self.__configuration = Configuration.load_from_file()
        self.__llm = self.__create_llm()
        self.__prompt = None

    def __create_llm(self):
        """
        Method to create llm object
        """
        api_key = self.__configuration['api']['key']
        llm = LLM(logger=self.__logger, api_key=api_key, name='chat-engine')
        return llm

    def generate_ai_assisted_query(self, query):
        """
        method to get the llm response
        """
        prompt = self.__get_prompt()
        response = self.__llm.get_llm_response(docs=list(), prompt=prompt, query=query)
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
        prompt_template = """You are the smart agent who generate question to ask another smart agent.
            We are working with a report and if the question is about adding, modifying, deleting or removing certain
            columns or rows, answer me how to ask another smart agent who work with a database and who is capable of 
            generating sql scripts. Else, you can give me the question asked as my answer
            user prompt: \n{question}\n
            """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        return prompt
