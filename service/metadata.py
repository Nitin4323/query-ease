import re

from logger import Logger
from utility import Database, File


class Metadata:
    __logger = Logger.get_logger()

    @classmethod
    def __get_schema_names(cls):
        """
        Method to fetch all schema names from given database
        """
        cls.__logger.info('fetching all schema names from database...')
        schema_names = list()
        query = """SELECT schema_name 
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema','pg_toast');"""
        _, result,_ = Database.execute_query(query)
        for item in result:
            schema_names.append(item[0])
        cls.__logger.info(f'fetched all schema names. number of schema names: {len(schema_names)}')
        return schema_names

    @classmethod
    def __get_table_names(cls, schema_name):
        """
        Method to fetch all table names of given schema name
        """
        table_names = list()
        cls.__logger.info(f'fetching all the table names of schema: {schema_name}...')
        query = f"""SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema_name}';"""
        _, result,_ = Database.execute_query(query)
        for item in result:
            table_names.append(item[0])
        cls.__logger.info(f'fetched all table names. number of tables: {len(table_names)}')
        return table_names

    @classmethod
    def __get_column_details(cls, schema_name, table_name):
        """
        Method to get column details of a given table of a given schema
        """
        cls.__logger.info(f'fetching column details of table: {table_name}')
        query = f"""SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_schema = '{schema_name}' AND table_name = '{table_name}';"""
        _, result,_ = Database.execute_query(query)
        cls.__logger.info(f'fetched column details. number of columns: {len(result)}')
        return result

    @classmethod
    def __get_relationship_details(cls, schema_name, table_name):
        """
        Method to get relationship details of a given table of a given schema
        """
        cls.__logger.info(f'fetching relationship details of table name: {table_name}')
        query = f"""SELECT 
                        tc.table_name AS foreign_table, 
                        kcu.column_name AS foreign_column,
                        ccu.table_name AS referenced_table,
                        ccu.column_name AS referenced_column
                FROM 
                    information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' AND 
                tc.table_schema = '{schema_name}' AND tc.table_name = '{table_name}';"""
        _, result,_ = Database.execute_query(query)
        cls.__logger.info(f'fetched relationship details. number of relationships: {len(result)}')
        return result

    @classmethod
    def __write_schema_details_to_file(cls, schema_name, file):
        """
        Method to write schema details to given file object
        """
        file.write(f"Schema: {schema_name}\n")
        table_names = cls.__get_table_names(schema_name)
        for table_name in table_names:
            cls.__write_table_details_to_file(schema_name, table_name, file)

    @classmethod
    def __write_table_details_to_file(cls, schema_name, table_name, file):
        """
        Method to write table details to the given file object
        """
        file.write(f"  Table: {table_name}\n")
        column_details = cls.__get_column_details(schema_name, table_name)
        cls.__write_column_details_to_file(column_details, file)
        relationship_details = cls.__get_relationship_details(schema_name, table_name)
        cls.__write_relationship_details_to_file(table_name, relationship_details, file)

    @classmethod
    def __write_column_details_to_file(cls, column_details, file):
        """
        Method to write column details to the given file object
        """
        file.write(f"    Columns:\n")
        for column_detail in column_details:
            column_name, data_type, character_maximum_length = column_detail
            if data_type == 'integer':
                file.write(f"      {column_name}: {data_type}\n")
            else:
                file.write(f"      {column_name}: {data_type} {character_maximum_length}\n")

    @classmethod
    def __write_relationship_details_to_file(cls, table_name, relationship_details, file):
        """
        Method to write relationship details to the given file object
        """
        file.write(f"    Relations:\n")
        if relationship_details:
            for relationship_detail in relationship_details:
                foreign_table, foreign_column, ref_table, ref_column = relationship_detail
                file.write(f"      {foreign_column} in {table_name} references {ref_column} in {ref_table}\n")
        else:
            file.write(f"      No relations found.\n")

    @classmethod
    def export_database_metadata(cls, filename):
        """
        Method to create database metadata file
        """
        cls.__logger.info(f'obtaining details to create database metadata file: {filename}')
        schema_names = cls.__get_schema_names()
        with open(filename, "w") as file:
            for schema_name in schema_names:
                cls.__write_schema_details_to_file(schema_name, file)
        cls.__logger.info(f'database metadata written to {filename}')

    @classmethod
    def get_chunks_of_metadata(cls, filename):
        """
        Method to create the chunks from metadata file
        """
        content = File.read_file(filename)
        table_name_pattern = r'\bTable\b'
        split_data = re.split(table_name_pattern, content)
        chunks = []
        for i, chunk in enumerate(split_data):
            if i > 0:
                chunk = f'Table{chunk}'
            chunks.append(chunk)
        return chunks
