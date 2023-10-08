import tkinter as tk
from tkinter import ttk
import configparser
from tkinter import StringVar
from tkinter import messagebox
from connect_db import Config_db
import os
class UIConect_db(tk.Toplevel):
        def __init__(self,parent):
            self.conectdb = Config_db()
            self.form_db = tk.Toplevel(parent)
            self.form_db.title("Nhập thông tin database")
            self.form_db.geometry("450x450")
            self.form_db.resizable(False, False)
            self.file_config = os.path.abspath('database.ini')
            # frame chứa các input
            self.input_frame_server = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_server.pack(side=tk.TOP, padx=10)
            # self.input_frame_server.configure(background="red")
            self.input_frame_server.pack_propagate(False)
            
            self.input_frame_host = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_host.pack(side=tk.TOP, padx=10)
            # self.input_frame_host.configure(background="blue")
            self.input_frame_host.pack_propagate(False)
            
            self.input_frame_port = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_port.pack(side=tk.TOP, padx=10)
            # self.input_frame_port.configure(background="blue")
            self.input_frame_port.pack_propagate(False)
            
            self.input_frame_db_name = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_db_name.pack(side=tk.TOP, padx=10)
            # self.input_frame_db_name.configure(background="blue")
            self.input_frame_db_name.pack_propagate(False)
            
            self.input_frame_db_username = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_db_username.pack(side=tk.TOP, padx=10)
            # self.input_frame_db_username.configure(background="blue")
            self.input_frame_db_username.pack_propagate(False)
            
            self.input_frame_db_password = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_db_password.pack(side=tk.TOP, padx=10)
            # self.input_frame_db_password.configure(background="blue")
            self.input_frame_db_password.pack_propagate(False)
            
            #Chứ các button
            self.input_frame_db_button = tk.Frame(self.form_db, width=600, height=60)
            self.input_frame_db_button.pack(side=tk.TOP, padx=10)
            # self.input_frame_db_button.configure(background="blue")
            self.input_frame_db_button.pack_propagate(False)
            #Combo Box
            self.id_label = tk.Label(self.input_frame_server, text="Loại database:",  width=20)
            self.id_label.pack(side = tk.LEFT, padx=10, pady=10)
            self.list_database =  ['MSSQL', "MySQL"]
            self.selected_database = tk.StringVar()
            self.selected_database.set(self.list_database[1])
            self.db_now = tk.Label(self.input_frame_server, textvariable=self.selected_database, width=20)
            self.db_now.pack(side = tk.LEFT, padx=10, pady=10)
            self.db_now.pack_forget()
            self.db_combobox = ttk.Combobox(self.input_frame_server, textvariable=self.selected_database)
            self.db_combobox['values'] =  self.list_database
            self.db_combobox.pack(side = tk.LEFT, padx=10, pady=10)
            self.db_combobox.current(1)
            
            # Label và Entry cho tên database
            self.lb_host = tk.Label(self.input_frame_host, text="Database Host", width=20)
            self.lb_host.pack(side=tk.LEFT, padx=10, pady=10)
            self.database_host_var = StringVar()
            self.database_host_var.set("123456")
            self.host_entry = tk.Entry(self.input_frame_host, textvariable=self.database_host_var, validate="key", width=24)
            self.host_entry.pack(side=tk.LEFT, padx=10, pady=10)
            
            # Dữ liệu cho port 
            self.port_label = tk.Label(self.input_frame_port, text="Port", width=20)
            self.port_label.pack(side=tk.LEFT, padx=10, pady=10)
            self.database_port_var = StringVar()
            self.database_port_var.set("1433")
            self.port_entry = tk.Entry(self.input_frame_port, textvariable=self.database_port_var, validate="key", width=24)
            self.port_entry.pack(side=tk.LEFT, padx=10, pady=10)
            
            # Dữ liệu Database name
            self.database_name_label = tk.Label(self.input_frame_db_name, text="Database Name", width=20)
            self.database_name_label.pack(side=tk.LEFT, padx=10, pady=10)
            self.database_name_var = StringVar()
            self.database_name_var.set("mydatabase")
            self.database_name_entry = tk.Entry(self.input_frame_db_name, textvariable=self.database_name_var, validate="key", width=24)
            self.database_name_entry.pack(side=tk.LEFT, padx=10, pady=10)
            
            # Dữ liệu cho username
            self.database_username_label = tk.Label(self.input_frame_db_username, text="User Name", width=20)
            self.database_username_label.pack(side=tk.LEFT, padx=10, pady=10)
            self.database_username_var = StringVar()
            self.database_username_var.set("1234556789")
            self.database_username_entry = tk.Entry(self.input_frame_db_username, textvariable=self.database_name_var, validate="key", width=24)
            self.database_username_entry.pack(side=tk.LEFT, padx=10, pady=10)
            
                        
            # Dữ liệu cho password
            self.database_password_label = tk.Label(self.input_frame_db_password, text="PassWord", width=20)
            self.database_password_label.pack(side=tk.LEFT, padx=10, pady=10)
            self.database_password_var = StringVar()
            self.database_password_var.set("1234556789")
            self.database_password_entry = tk.Entry(self.input_frame_db_password, textvariable=self.database_name_var, validate="key", width=24,show="*")
            self.database_password_entry.pack(side=tk.LEFT, padx=10, pady=10)
            
            #Các nút
            self.save_button = tk.Button(self.input_frame_db_button, text="Lưu đường dẫn", command=self.save_path, height = 3)
            self.save_button.grid(row=3, column=2, columnspan=2 ,sticky=tk.W, padx=20, pady=5)
            self.load_credentials()
        def save_path(self):
            try:
                config = configparser.ConfigParser()
                config.read(self.file_config)
                config['Database'] = {
                    "Db_type" : self.selected_database.get(),
                    "db_host" : self.database_host_var.get(),
                    "port" : self.database_port_var.get(),
                    "database_name" : self.database_name_var.get(),
                    "username" : self.database_username_var.get(),
                    "password" : self.database_password_var.get()
                }
                with open(self.file_config, 'w') as configfile:
                    config.write(configfile)
                messagebox.showinfo(title="Thành công",message = "Lưu đường dẫn thành công")
                try:
                    self.conectdb.connect()
                    messagebox.showinfo(title="Thành công",message = "Kết nối thành công")
                except:
                    messagebox.showerror(title="Lỗi",message = "Bạn đã gặp lỗi kết nối database")
            except Exception as e:
                messagebox.showerror(title="Lỗi",message = "Bạn đã gặp lỗi")
                print(e)
                
        def load_credentials(self):
            config = configparser.ConfigParser()
            config.read(self.file_config)
            if 'Database' in config:
                for x in config['Database']:
                    self.selected_database.set(config['Database']['Db_type'])
                    self.database_host_var.set(config['Database']['db_host'])
                    self.database_port_var.set(config['Database']['port'])
                    self.database_name_var.set(config['Database']['database_name'])
                    self.database_username_var.set(config['Database']['username'])
                    self.database_password_var.set(config['Database']['password'])
