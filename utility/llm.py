import json

import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain


class LLM:
    def __init__(self, logger, api_key, name='default'):
        self.name = name
        self.logger = logger
        self.api_key = api_key
        self.model = self.__create_llm_model(name, api_key)

    def __create_llm_model(self, api_key, name):
        """
        Method to create llm model
        """
        self.logger.info(f'creating new generative ai engine: {name}')
        genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, api_key=api_key)
        self.logger.info(f'created engine: {name}')
        return model

    def __get_conversational_chain(self, prompt):
        """
        Method to get conversational chain
        """
        self.logger.info(f'getting conversational chain from generative ai engine: {self.name}')
        self.logger.info(f'given prompt: {prompt}')
        chain = load_qa_chain(self.model, chain_type="stuff", prompt=prompt)
        self.logger.info('obtained conversational chain')
        return chain

    def __cleanup_raw_response(self, raw_response):
        """
        Method to extract relevant text from response object
        """
        self.logger.info('parsing raw response...')
        self.logger.info("Reply: ", raw_response["output_text"])
        response: str = raw_response["output_text"]
        clean_response = json.loads(response[8:-3])
        return clean_response

    def get_llm_response(self, docs, prompt, query):
        """
        Method to get the llm response
        """
        chain = self.__get_conversational_chain(prompt)
        raw_response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        clean_response = self.__cleanup_raw_response(raw_response)
        return clean_response
