import pyodbc
import configparser
import mysql.connector
import os
class Config_db():
    def __init__(self):
        self.file_config = os.path.abspath("database.ini")
        # self.load_credentials()
        self.list_db = ["MSSQL, MySQL"]
    def connect(self):
        config = configparser.ConfigParser()
        config.read(self.file_config)
        if 'Database' in config:
            self.selected_database = config['Database']['Db_type']
            self.database_host = config['Database']['db_host']
            self.database_port = config['Database']['port']
            self.database_name = config['Database']['database_name']
            self.database_username = config['Database']['username']
            self.database_password = config['Database']['password']
            if(self.selected_database == "MSSQL"):
                self.connect_driver = "DRIVER={Devart ODBC Driver for SQL Server};"
                self.connect_infor = 'Server={0};Database={1};Port={2};User ID={3};Password={4}'.format(self.database_host, self.database_name,self.database_port,self.database_username,self.database_password)
                self.connect_string = self.connect_driver + self.connect_infor
                try:
                    self.cnxn = pyodbc.connect(self.connect_string)
                except:
                    print("Error connecting to database")
            if(self.selected_database == "MySQL"):
                config = {
                    'user':self.database_username,
                    'password': self.database_password,
                    'host': self.database_host,
                    'database': self.database_name,
                    'raise_on_warnings': True
                }
                cnx = mysql.connector.connect(**config)