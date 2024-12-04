from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RAG:
    def __init__(self, logger, api_key, init_texts):
        self.__logger = logger
        self.__embeddings = self.__create_generative_ai_embeddings(api_key)
        self.__database = self.__create_vector_database(init_texts)

    def __create_generative_ai_embeddings(self, api_key):
        """
        Method to create generative AI embeddings
        """
        self.__logger.info('creating generative ai embeddings')
        print(api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.__logger.info('created generative ai embeddings')
        return embeddings


    def __create_vector_database(self, init_texts):
        """
        Method to create vector database
        """
        vector_store = FAISS.from_texts(init_texts, embedding=self.__embeddings)
        vector_store.save_local("faiss_index")
        database = FAISS.load_local("faiss_index", self.__embeddings, allow_dangerous_deserialization=True)
        return database

    def execute_similarity_search(self, query):
        """
        Method to do similarity search in vector database
        """
        self.__logger.info('executing similarity search in vector database...')
        self.__logger.info(f'query: {query}')
        docs = self.__database.similarity_search(query)
        self.__logger.info('completed similarity search')
        return docs
