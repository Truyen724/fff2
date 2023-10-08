import tkinter as tk
from tkinter import ttk, Menu
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import StringVar
import io
import os
import cv2
import face_recognition
from face_recognition import FaceRecognition
from ui_conect_db import UIConect_db
import pickle
from openvino.runtime import Core, get_version
core = Core()
FaceRecognition = FaceRecognition(is_load_pretrain=False)
FOLDER_LOADATA = os.path.abspath("../data-face")
FOLDER_LOADATA2 = os.path.abspath("data-face")
FOLDER_PATH = os.path.abspath("face-mark")
# for i,  x in enumerate(FaceRecognition.frame_processor.faces_database):
#     print(x.label)
#     print(x.descriptors)
    # if x.label == "11234_truyen":
    #     FaceRecognition.frame_processor.faces_database.database.remove(x)
class UIRecognition():
    def __init__(self):
        # Tạo cửa sổ tkinter
        self.root = tk.Tk()
        self.root.title("Face recognition")
        self.root.resizable(False, False)
        self.root.geometry("820x480")

        # Tạo frame chứa ảnh
        self.image_frame = tk.Frame(self.root, width=450, height=300)
        self.image_frame.pack(side=tk.LEFT, padx=0, pady=0)

        # Tạo frame bên trái
        self.left_frame = tk.Frame(self.root, width=300)
        self.left_frame.pack(side=tk.TOP, padx=0, pady=0)
        # Tạo frame camera
        self.open_camera_frame = tk.Frame(self.left_frame, width=300)
        self.open_camera_frame.pack(side=tk.TOP, padx=0, pady=20)
        # Tạo frame bên trái
        self.bottom_frame = tk.Frame(self.left_frame, width=300)
        self.bottom_frame.pack_forget()

        # Đường dẫn đến video hoặc webcam
        self.video_path = 0  # Sử dụng webcam
        #video_path = "video.mp4"  # Sử dụng file video


        # Tạo Combobox để chọn ID của camera
        self.camera_list = self.get_available_cameras()
        self.selected_camera = tk.StringVar()
        self.camera_combobox = ttk.Combobox(self.open_camera_frame, textvariable=self.selected_camera, values=self.camera_list, state="readonly")
        self.camera_combobox.pack(fill=tk.X, padx=0, pady=0)
        self.camera_combobox.current(0)  # Chọn camera đầu tiên trong danh sách


        # Tạo nút "Mở camera"
        self.open_button = tk.Button(self.open_camera_frame, text="Mở camera", command=self.open_camera)
        self.open_button.pack(fill=tk.X, padx=0, pady=0)
        # Mở video hoặc webcam
        self.cap = cv2.VideoCapture(self.video_path)
        # Label để hiển thị ảnh
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Gọi hàm update_frame để cập nhật ảnh lên label
        self.update_frame()

        # Tạo các nút
        self.action_button= tk.Button(self.open_camera_frame, text="Action", command=self.show_button)
        self.action_button.pack(fill=tk.X, padx=0, pady=0)
        
        
        self.add_button = tk.Button(self.bottom_frame, text="Thêm người dùng", command=self.add_user)
        self.add_button.pack(fill=tk.X, padx=0, pady=0)

        self.edit_button = tk.Button(self.bottom_frame, text="Sửa người dùng", command=self.edit_user)
        self.edit_button.pack(fill=tk.X, padx=0, pady=0)

        self.delete_button = tk.Button(self.bottom_frame, text="Xoá người dùng", command=self.delete_user)
        self.delete_button.pack(fill=tk.X, padx=0, pady=(0,20))

        self.connect_db_button = tk.Button(self.bottom_frame, text="Chỉnh sửa kết nối database", command=self.confg_db)
        self.connect_db_button.pack(fill=tk.X, padx=0, pady=0)

        self.get_data_button = tk.Button(self.bottom_frame, text="Lấy dữ liệu từ trước", command=self.get_data)
        self.get_data_button.pack(fill=tk.X, padx=0, pady=0)
        self.reload_db_button = tk.Button(self.bottom_frame, text="Reload Data", command=self.reload_db)
        self.reload_db_button.pack(fill=tk.X, padx=0, pady=0)
        # Tạo menu
        menuBar = Menu(self.root)
        self.root.config(menu=menuBar)
    def run(self):
        # Chạy giao diện
        self.root.mainloop()

        # Khi kết thúc, giải phóng tài nguyên
        face_recognition.client_socket.close()
        self.cap.release()
        cv2.destroyAllWindows()
    def show_button(self):
        if self.bottom_frame.winfo_viewable():
            self.bottom_frame.pack_forget()
        else:
            self.bottom_frame.pack(side=tk.BOTTOM, padx=0, pady=0)
        pass
    def get_data(self):
        # folder_selected = tk.filedialog.askdirectory(title = "Chọn folder chứa ảnh")
        folder_selected = FOLDER_LOADATA
        try:
            messagebox.showinfo("Thông báo", "Đang load dữ liệu!, vui lòng chờ")
            print("Đang load dữ liệu")
            FaceRecognition.frame_processor.save_to_gallery(folder_selected)
            self.reload_db()
        except Exception as  e:
            print(e) 
    # Lấy danh sách các camera hiện có
    def get_available_cameras(self):
        camera_list = []
        for i in range(10):  # Giới hạn tìm kiếm trong khoảng 0-9
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                camera_list.append(i)
                self.cap.release()
            else:
                break
        return camera_list
    # Mở video từ camera đã chọn
    def open_camera(self):
        selected_camera_id = int(self.selected_camera.get())
        self.cap = cv2.VideoCapture(selected_camera_id)
    # Hàm update frame
    def check_update(self):
        file = os.listdir(FOLDER_LOADATA)
        if(len(file)>0):
            FaceRecognition.frame_processor.save_to_update_face(FOLDER_LOADATA)
            FaceRecognition.frame_processor.faces_database.save_pretrain()
        file2 = os.listdir(FOLDER_LOADATA2)
        if(len(file2)>0):
            FaceRecognition.frame_processor.save_to_update_face(FOLDER_LOADATA2)
            FaceRecognition.frame_processor.faces_database.save_pretrain()
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi frame từ BGR sang RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.check_update()
            # Resize frame
            # frame_resized = cv2.resize(frame_rgb, (450, 300))

            # Nhận diện khuôn mặt
            frame_detect = FaceRecognition.detect(frame_rgb)
            # print(len(FaceRecognition._list_users))
            # Tạo ảnh PIL từ frame
            image = Image.fromarray(frame_detect)
            
            # Tạo đối tượng PhotoImage từ ảnh
            photo = ImageTk.PhotoImage(image)
            
            # Cập nhật ảnh trên label
            self.image_label.config(image=photo)
            self.image_label.image = photo

        # Gọi lại hàm update_frame sau 1ms
        self.image_label.after(1, self.update_frame)

    def logout(self):
        self.root.destroy()
    def add_user(self):
        UIAddUser(self.root)

    def edit_user(self):
        UIEditUser(self.root)
    def confg_db(self):
        UIConect_db(self.root)
    def reload_db(self):
        global FaceRecognition
        FaceRecognition.is_load_pretrain = False
        FaceRecognition.reload_data()
    def delete_user(self):
        # Xử lý khi nhấn nút "Xoá người dùng"
        UIDeletetUser(self.root)
        print("Xoá người dùng")

class UIAddUser(tk.Toplevel):
    def __init__(self, parent):
        self.folder_path = FOLDER_PATH
        self. users_dict = {}
        self.read_list_users()
        self.list_images = FaceRecognition._list_users
        self.status_list_images = 1
        self.name_user = ""
        self.form_adduser = tk.Toplevel(parent)
        self.form_adduser.title("Nhập thông tin người dùng")
        self.form_adduser.geometry("600x300")
        self.form_adduser.resizable(False, False)

        # # Đọc hình ảnh mặc định từ OpenCV
        # self.default_image_path = "background_image.png"
        # self.cv_image = cv2.imread(self.default_image_path)
        # self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.cv_image = cv2.cvtColor(self.list_images[0], cv2.COLOR_BGR2RGB)

        # Tạo buffer cho hình ảnh
        self.image_buffer = io.BytesIO()
        Image.fromarray(self.list_images[0]).save(self.image_buffer, format='PNG')
        self.image_buffer.seek(0)

        # Tạo hình ảnh từ buffer
        self.pil_image = Image.open(self.image_buffer)
        self.pil_image = self.pil_image.resize((150, 200), Image.LANCZOS)

        # Chuyển đổi PIL Image sang định dạng hỗ trợ bởi Tkinter
        self.tk_image = ImageTk.PhotoImage(self.pil_image)

        self.prev_button = tk.Button(self.form_adduser, text="◄", font=("Arial", 20, "bold"), command=self.prev_image, bd=0)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.prev_button.config(state="disabled")
        # Frame hình ảnh
        self.image_frame = tk.Frame(self.form_adduser)
        self.image_frame.pack(side=tk.LEFT, padx=0)

        self.next_button = tk.Button(self.form_adduser, text="►", font=("Arial", 20, "bold"), command=self.next_image, bd=0)
        self.next_button.pack(side=tk.LEFT, padx=5)

        if len(self.list_images) <= 1:
            self.next_button.config(state="disabled")
            

        # Hiển thị hình ảnh mặc định bên trái
        self.image_label = tk.Label(self.image_frame, image=self.tk_image)
        self.image_label.pack(side=tk.TOP, padx=10, pady=10)

        # Button Browse
        self.browse_button = tk.Button(self.image_frame, text=" Tải ảnh lên ", command=self.browse_image, bd=1)
        self.browse_button.pack(side=tk.BOTTOM,pady=10)

        # Frame chứa các trường nhập liệu
        self.input_frame_top = tk.Frame(self.form_adduser, width=150, height=300)
        self.input_frame_top.pack(side=tk.TOP, padx=10)

        self.input_frame_bottom = tk.Frame(self.form_adduser, width=150, height=300)
        self.input_frame_bottom.pack(side=tk.TOP, padx=10)

        # Label và Entry cho ID
        self.id_label = tk.Label(self.input_frame_top, text="ID:             ")
        self.id_label.pack(side=tk.LEFT, padx=10, pady=(30, 10))
        self.sv = StringVar()
        self.sv.trace("w", lambda name, index, mode, sv=self.sv: self.on_entry_change(sv))
        self.validate_numeric_input = self.form_adduser.register(self.validate_input)
        self.id_entry = tk.Entry(self.input_frame_top, textvariable=self.sv, validate="key", validatecommand=(self.validate_numeric_input, "%P"))
        self.id_entry.pack(side=tk.LEFT, padx=10, pady=(30, 10))


        # Label và Entry cho Họ và tên
        self.name_label = tk.Label(self.input_frame_bottom, text="Họ và tên:")
        self.name_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.name_entry = tk.Entry(self.input_frame_bottom)
        self.name_entry.pack(side=tk.LEFT, padx=10, pady=10)

        # Button Submit
        self.submit_button = tk.Button(self.form_adduser, text="  Xác nhận  ", command=self.submit, bd=1)
        self.submit_button.pack(pady=10)


    def read_list_users(self):
        # Duyệt qua các file và thư mục trong thư mục
        for item in os.listdir(self.folder_path):
            item_path = os.path.join(self.folder_path, item)
            
            # Kiểm tra xem item là file hay thư mục
            if os.path.isfile(item_path):
                # Tách tên file thành phần trước và sau dấu "_"
                file_name = os.path.splitext(item)[0]
                key, value = file_name.split("_", 1)
                
                # Kiểm tra xem key đã tồn tại trong users_dict hay chưa
                if key in self.users_dict:
                    # Trường hợp key đã tồn tại, thực hiện phân tích và cộng giá trị
                    last_value = self.users_dict[key].split("-")[-1]
                    incremented_value = int(last_value) + 1
                    new_value = value.rsplit("-", 1)[0] + "-" + str(incremented_value)
                    self.users_dict[key] = new_value
                else:
                    # Trường hợp key chưa tồn tại, gán giá trị mới
                    last_value = value.rsplit("-", 1)[-1]
                    incremented_value = int(last_value) + 1
                    new_value = value.rsplit("-", 1)[0] + "-" + str(incremented_value)
                    self.users_dict[key] = new_value

        print(self.users_dict)

    
    def on_entry_change(self, sv):
        # Xử lý sự kiện thay đổi giá trị trường Entry
        new_text = sv.get()
        for key, value in self.users_dict.items():
            if str(new_text) == key:
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(tk.END, value.rsplit("-", 1)[0])
                self.name_user = value
                self.name_entry.configure(state="disabled")
                break
            else:
                self.id_user = key
                self.name_user = ""
                self.name_entry.configure(state="normal")
                self.name_entry.delete(0, tk.END)
        # print("Giá trị mới:", new_text)
        
    def validate_input(self, new_text):
        if new_text.isdigit() or new_text == "":
            return True
        else:
            return False

    def image_OpenCV2PIL(self, cv_image):
        self.cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_buffer = io.BytesIO()
        # Tạo buffer cho hình ảnh
        Image.fromarray(cv_image).save(image_buffer, format='PNG')
        image_buffer.seek(0)

        # Tạo hình ảnh từ buffer
        pil_image = Image.open(image_buffer)
        pil_image = pil_image.resize((150, 200), Image.LANCZOS)

        # Chuyển đổi PIL Image sang định dạng hỗ trợ bởi Tkinter
        tk_image = ImageTk.PhotoImage(pil_image)
        return tk_image

    def next_image(self):
        self.status_list_images += 1
        if self.status_list_images >= len(self.list_images):
            self.status_list_images = len(self.list_images)
            self.next_button.config(state="disabled")
        else:
            self.next_button.config(state="normal")
        if self.status_list_images > 1:
            image_tk = self.image_OpenCV2PIL(self.list_images[self.status_list_images-1])
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk
            self.prev_button.config(state="normal")

    def prev_image(self):
        self.status_list_images -= 1
        if self.status_list_images <= 1:
            self.status_list_images = 1
            self.prev_button.config(state="disabled")
        else:
            self.prev_button.config(state="normal")
        if self.status_list_images < len(self.list_images):
            image_tk = self.image_OpenCV2PIL(self.list_images[self.status_list_images-1])
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk
            self.next_button.config(state="normal")
    def browse_image(self):
        # Hiển thị hộp thoại chọn tệp tin
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Đọc hình ảnh từ tệp tin
            self.cv_image = cv2.imread(file_path)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.list_images.append(self.cv_image)
            self.status_list_images = len(self.list_images)
            self.prev_button.config(state="normal")
            self.next_button.config(state="disabled")
            # Tạo buffer cho hình ảnh mới
            self.image_buffer = io.BytesIO()
            Image.fromarray(self.cv_image).save(self.image_buffer, format='PNG')
            self.image_buffer.seek(0)

            # Tạo hình ảnh từ buffer
            self.pil_image = Image.open(self.image_buffer)
            self.pil_image = self.pil_image.resize((150, 200), Image.ANTIALIAS)

            # Chuyển đổi PIL Image sang định dạng hỗ trợ bởi Tkinter
            self.tk_image = ImageTk.PhotoImage(self.pil_image)

            # Cập nhật hình ảnh bên trái
            self.image_label.configure(image=self.tk_image)

    def submit(self):
        user_id = self.id_entry.get()
        full_name = self.name_entry.get()
        if not user_id or not full_name:
            # Kiểm tra xem dữ liệu có bị để trống hay không
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin")
            return
        
        if self.name_user == "":
            full_name = self.name_entry.get() +"-0"
        else:
            full_name = self.name_user

        full_idname = user_id+"_"+full_name+".jpg"
        file_name = r"{}/{}".format(FOLDER_PATH, full_idname)
        cv2.imwrite(file_name, self.cv_image)
        FaceRecognition.frame_processor.faces_database.add_face_to_database(file_name)
        print(file_name)
        print("ID:", user_id)
        print("Họ và tên:", full_name)

        # Xóa dữ liệu nhập trên các trường input
        self.id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)

        FaceRecognition.frame_processor.faces_database.load_pretrain()
        # Load lại mô hình
        # FaceRecognition.reload_data()
        print(len(FaceRecognition.frame_processor.faces_database.database))
        messagebox.showinfo("Thông báo", "Thêm người dùng thành công!")
        # Trả giá trị về và xoá cửa sổ
        self.form_adduser.destroy()

class UIEditUser():
    def __init__(self, parent):
        self.data_dict = {}
        self.temp_arr = []
        self.read_filenames()
        
        # Tạo cửa sổ chính
        self.form_edituser = tk.Toplevel(parent)
        self.form_edituser.title("Chỉnh sửa thông tin")
        self.form_edituser.geometry("350x200")

        # Frame chứa các trường nhập liệu
        self.input_frame_top = tk.Frame(self.form_edituser)
        self.input_frame_top.pack(side=tk.TOP, padx=10, pady=10)
        
        self.input_frame_body = tk.Frame(self.form_edituser)
        self.input_frame_body.pack(side=tk.TOP, padx=10, pady=10)

        self.input_frame_bottom = tk.Frame(self.form_edituser)
        self.input_frame_bottom.pack(side=tk.TOP, padx=10, pady=10)

        # Label và Entry cho tìm kiếm ID
        self.find_id_label = tk.Label(self.input_frame_top, text="Tìm Kiếm theo ID:")
        self.find_id_label.pack(side=tk.LEFT, padx=10)
        self.sv = StringVar()
        self.sv.trace("w", lambda name, index, mode, sv=self.sv: self.on_entry_change(sv))
        self.validate_numeric_input = self.form_edituser.register(self.validate_input)
        self.find_id_entry = tk.Entry(self.input_frame_top, width=30, textvariable=self.sv, validate="key", validatecommand=(self.validate_numeric_input, "%P"))
        self.find_id_entry.pack(side=tk.LEFT, padx=10)
        # Label và Entry cho ID
        self.id_label = tk.Label(self.input_frame_body, text="Sửa ID:                   ")
        self.id_label.pack(side=tk.LEFT, padx=10)

        self.id_entry = tk.Entry(self.input_frame_body, width=30)
        self.id_entry.pack(side=tk.LEFT, padx=10)
        self.id_entry.configure(state="disabled") 
        # Label và Entry cho Họ và tên
        self.name_label = tk.Label(self.input_frame_bottom, text="Sửa tên:                 ")
        self.name_label.pack(side=tk.LEFT, padx=10)

        self.name_entry = tk.Entry(self.input_frame_bottom, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=10)
        self.name_entry.configure(state="disabled")

        # Button Submit
        self.submit_button = tk.Button(self.form_edituser, text="Sửa thông tin User", command=self.submit)
        self.submit_button.pack(pady=10)

    def add_data_temp_arr(self):
        for value in self.data_dict[self.find_id_entry.get()]:
            self.temp_arr.append(self.name_entry.get() + "-" + value.rsplit("-", 1)[-1])
        
    
    def rename_files(self):
        for index, def_value in enumerate(self.data_dict[self.find_id_entry.get()]):
            default_name_path = FOLDER_PATH + "/" + self.find_id_entry.get() + "_" + def_value
            new_name_path = FOLDER_PATH + "/" + self.find_id_entry.get() + "_"+ self.temp_arr[index]
            os.rename(default_name_path, new_name_path)
            print(default_name_path, new_name_path)
            
    def config_embeded_db(self):
        new_lable = self.find_id_entry.get() +'_'+self.name_entry.get()
        FaceRecognition.frame_processor.faces_database.rename_user(self.find_id_entry.get(),new_lable)

    def submit(self):
        user_id = self.id_entry.get()
        full_name = self.name_entry.get()
        if not user_id or user_id == "ID không tồn tại" or not full_name:
            # Kiểm tra xem dữ liệu có bị để trống hay không
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin")
            return
        self.add_data_temp_arr()
        self.rename_files()
        
        # Xử lý dữ liệu nhập vào ở đây
        print("ID:", user_id)
        print("Tên:", full_name)
        self.config_embeded_db()
        # Xóa dữ liệu nhập trên các trường input
        self.id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)

        # Load lại mô hình
        # FaceRecognition.reload_data()
        
        messagebox.showinfo("Thông báo", "Sửa người dùng thành công!")
        # Trả giá trị về và xoá cửa sổ
        self.form_edituser.destroy()

    def on_entry_change(self, sv):
        # Xử lý sự kiện thay đổi giá trị trường Entry
        new_text = sv.get()
        for key, value in self.data_dict.items():
            if str(new_text) == key:
                self.id_entry.configure(state="normal")
                self.id_entry.delete(0, tk.END)
                self.id_entry.insert(tk.END, key)
                self.name_entry.configure(state="normal")
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(tk.END, value[0].rsplit("-", 1)[0])
                break
            else:
                self.id_user = key
                self.id_entry.delete(0, tk.END)
                self.id_entry.insert(tk.END, "ID không tồn tại")
                self.id_entry.configure(state="disabled")
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(tk.END, "Người dùng không tồn tại")
                self.name_entry.configure(state="disabled")
        # print("Giá trị mới:", new_text)

    def validate_input(self, new_text):
        if new_text.isdigit() or new_text == "":
            return True
        else:
            return False
    def read_filenames(self):
        filenames = os.listdir(FOLDER_PATH)
        for filename in filenames:
            # Tách tên file từ dấu "_"
            name_parts = filename.split("_",1)
            key = name_parts[0]  # Lấy phần đầu tiên làm key

            # Lấy các chuỗi sau dấu "_" và lưu vào mảng
            value = name_parts[1:]
            
            if key in self.data_dict:
                self.data_dict[key].extend(value)
            else:
                self.data_dict[key] = value
class UIDeletetUser():
    def __init__(self, parent):
        self.data_dict = {}
        self.temp_arr = []
        self.read_filenames()
        
        # Tạo cửa sổ chính
        self.form_edituser = tk.Toplevel(parent)
        self.form_edituser.title("Xóa User")
        self.form_edituser.geometry("350x200")

        # Frame chứa các trường nhập liệu
        self.input_frame_top = tk.Frame(self.form_edituser)
        self.input_frame_top.pack(side=tk.TOP, padx=10, pady=10)
        
        self.input_frame_body = tk.Frame(self.form_edituser)
        self.input_frame_body.pack(side=tk.TOP, padx=10, pady=10)

        self.input_frame_bottom = tk.Frame(self.form_edituser)
        self.input_frame_bottom.pack(side=tk.TOP, padx=10, pady=10)

        # Label và Entry cho tìm kiếm ID
        self.find_id_label = tk.Label(self.input_frame_top, text="Tìm Kiếm theo ID:")
        self.find_id_label.pack(side=tk.LEFT, padx=10)
        self.sv = StringVar()
        self.sv.trace("w", lambda name, index, mode, sv=self.sv: self.on_entry_change(sv))
        self.validate_numeric_input = self.form_edituser.register(self.validate_input)
        self.find_id_entry = tk.Entry(self.input_frame_top, width=30, textvariable=self.sv, validate="key", validatecommand=(self.validate_numeric_input, "%P"))
        self.find_id_entry.pack(side=tk.LEFT, padx=10)
        # Label và Entry cho ID
        self.id_label = tk.Label(self.input_frame_body, text="ID:                   ")
        self.id_label.pack(side=tk.LEFT, padx=10)

        self.id_entry = tk.Entry(self.input_frame_body, width=30)
        self.id_entry.pack(side=tk.LEFT, padx=10)
        self.id_entry.configure(state="disabled") 
        # Label và Entry cho Họ và tên
        self.name_label = tk.Label(self.input_frame_bottom, text="Tên:                 ")
        self.name_label.pack(side=tk.LEFT, padx=10)

        self.name_entry = tk.Entry(self.input_frame_bottom, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=10)
        self.name_entry.configure(state="disabled")

        # Button Submit
        self.submit_button = tk.Button(self.form_edituser, text="Xóa User", command=self.submit)
        self.submit_button.pack(pady=10)
            
    def delete_on_embeded_db(self, id_user):
        FaceRecognition.frame_processor.faces_database.delete_user(id_user)
    def submit(self):
        user_id = self.id_entry.get()
        full_name = self.name_entry.get()
        # self.delete_on_embeded_db("99999")
        if not user_id or user_id == "ID không tồn tại" or not full_name:
            # Kiểm tra xem dữ liệu có bị để trống hay không
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin")
            return
        # self.delete_user_by_id(self.id_entry.get())
        # self.delete_on_embeded_db(self.id_entry.get())
        self.delete_on_embeded_db(user_id)
        self.delete_image(user_id)
        # Xử lý dữ liệu nhập vào ở đây
        print("ID:", user_id)
        print("Tên:", full_name)
        
        # Xóa dữ liệu nhập trên các trường input
        self.id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        
        messagebox.showinfo("Thông báo", "Sửa người dùng thành công!")
        # Trả giá trị về và xoá cửa sổ
        self.form_edituser.destroy()
    def delete_image(self, id_user):
        for name in os.listdir(FOLDER_PATH):
            if(name.split("_")[0]==id_user):
                os.remove(FOLDER_PATH+"/"+name)
            
        
    def on_entry_change(self, sv):
        # Xử lý sự kiện thay đổi giá trị trường Entry
        new_text = sv.get()
        for key, value in self.data_dict.items():
            if str(new_text) == key:
                self.id_entry.configure(state="normal")
                self.id_entry.delete(0, tk.END)
                self.id_entry.insert(tk.END, key)
                self.name_entry.configure(state="normal")
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(tk.END, value[0].rsplit("-", 1)[0])
                break
            else:
                self.id_user = key
                self.id_entry.delete(0, tk.END)
                self.id_entry.insert(tk.END, "ID không tồn tại")
                self.id_entry.configure(state="disabled")
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(tk.END, "Người dùng không tồn tại")
                self.name_entry.configure(state="disabled")
        # print("Giá trị mới:", new_text)

    def validate_input(self, new_text):
        if new_text.isdigit() or new_text == "":
            return True
        else:
            return False
    def delete_user_by_id(self, id):
        folder = os.path.abspath(FOLDER_PATH)
        for file in os.listdir(folder):
            if(file.split("_")[0]==id):
                os.remove(os.path.join(folder, file))
    def read_filenames(self):
        filenames = os.listdir(FOLDER_PATH)
        for filename in filenames:
            # Tách tên file từ dấu "_"
            name_parts = filename.split("_",1)
            key = name_parts[0]  # Lấy phần đầu tiên làm key

            # Lấy các chuỗi sau dấu "_" và lưu vào mảng
            value = name_parts[1:]
            
            if key in self.data_dict:
                self.data_dict[key].extend(value)
            else:
                self.data_dict[key] = value
# UIRecognition = UIRecognition()
# UIRecognition.run()