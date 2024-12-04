import pandas as pd

from utility import Database
from orchestrator import Orchestrator
from service import Features, Metadata

class Backend:
    __orchestrator = None

    @classmethod
    def initialize_app_orchestrator(cls, host, port, database, username, password):
        """
        Main method to initialize a query ease application session
            1) resets the database details
            2) get database metadata chunks
            3) initialize orchestrator
            4) execute prompts
        """
        Database.reset(host, port, database, username, password)
        metadata_chunks = cls.__get_metadata_chunks()
        cls.__orchestrator = Orchestrator(metadata_chunks)

    @classmethod
    def execute_prompt(cls, query, enable_prompt_layer):
        """
        Main method to send prompt to AI engine and get the select query and execute
            1) Send prompt to AI engine and to generate the select query
            2) Run validate, execute loop for a maximum of given retries
        """
        status, message, result = False, 'AI failed to generate valid query', list()
        sql_chat = False
        response  = cls.__orchestrator.get_llm_response(query=query, enable_prompt_layer=enable_prompt_layer)
        if response['is_sql']:
            status , error_msg = Database.validate_query(response['SQL'])
            if status:
                df  = pd.read_sql(response['SQL'],Database.get_connection())
                sql_chat = True
                return sql_chat,df
            else:
                return sql_chat,result
        else :
            return sql_chat,response['normal_response']

    @classmethod
    def __get_metadata_chunks(cls):
        """
        Method to create database metadata chunks
        """
        filename = 'data/metadata/database.txt'
        Metadata.export_database_metadata(filename)
        metadata_chunks = Metadata.get_chunks_of_metadata(filename)
        return metadata_chunks

    @classmethod
    def get_excel_from_df(cls, df):
        """
        Method to convert dataframes to excel
        """
        return Features.convert_df_to_excel(df)
