import tkinter as tk
from tkinter import messagebox
import configparser
from PIL import Image, ImageTk
from ui_recognition import UIRecognition
import os
image =  os.path.abspath("background-vang.png")
save_password = os.path.abspath('credentials.ini')
class UILogin():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Login")
        self.root.geometry("530x172")
        self.root.configure(bg="#FFFF66")
        self.root.resizable(False, False)

        # Load thông tin đăng nhập từ file .ini
        self.saved_username, self.saved_password = self.load_credentials()

        # Load ảnh đăng nhập
        self.image = Image.open(image)
        self.image = self.image.resize((300, 168), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(self.image)
       
        # Ảnh đăng nhập bên trái
        self.image_label = tk.Label(self.root, image=self.image)
        self.image_label.grid(row=0, column=0, rowspan=4)

        # Username label và entry
        self.username_label = tk.Label(self.root, text="Tài khoản:", bg="white", fg="black")
        self.username_label.grid(row=0, column=1, pady=5)
        self.username_entry = tk.Entry(self.root, bg="white", fg="black", bd=3, relief=tk.SOLID)
        self.username_entry.grid(row=0, column=2, padx=10, pady=5)
        self.username_entry.insert(0, self.saved_username)

        # Password label và entry
        self.password_label = tk.Label(self.root, text="Mật khẩu:", bg="white", fg="black")
        self.password_label.grid(row=1, column=1, pady=5)
        self.password_entry = tk.Entry(self.root, show="*", bg="white", fg="black", bd=3)
        self.password_entry.grid(row=1, column=2, padx=10, pady=5)
        self.password_entry.insert(0, self.saved_password)

        # Remember checkbox
        self.remember_var = tk.IntVar()
        self.remember_checkbox = tk.Checkbutton(self.root, text="Nhớ mật khẩu", variable=self.remember_var, bg="white", fg="black", selectcolor="#000000")
        self.remember_checkbox.grid(row=2, column=2, sticky=tk.W, pady=5)


        # Đăng nhập button
        self.login_button = tk.Button(self.root, text="Đăng nhập", command=self.login, bg="white", fg="black", bd=0)
        self.login_button.grid(row=3, column=2, columnspan=2 ,sticky=tk.W, padx=20, pady=5)

    def save_credentials(self, username, password):
        config = configparser.ConfigParser()
        config.read(save_password)
        config['Credentials'] = {
            'username': username,
            'password': password
        }
        with open(save_password, 'w') as configfile:
            config.write(configfile)

    def load_credentials(self):
        config = configparser.ConfigParser()
        config.read(save_password)
        if 'Credentials' in config:
            return config['Credentials']['username'], config['Credentials']['password']
        else:
            return '', ''

    def login(self):
        entered_username = self.username_entry.get()
        entered_password = self.password_entry.get()

        if self.remember_var.get():
            self.save_credentials(entered_username, entered_password)

        # Kiểm tra đăng nhập
        if entered_username == "admin" and entered_password == "123456":
            self.root.destroy()
            UIRecognition().run()
        else:
            messagebox.showerror("Lỗi", "Đăng nhập thất bại")
            
    def run(self):
        self.root.mainloop()

# UILogin = UILogin()
# UILogin.run()