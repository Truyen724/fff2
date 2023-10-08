from ui_login import UILogin
import tkinter as tk
from tkinter import messagebox
import configparser
from PIL import Image, ImageTk
from ui_recognition import UIRecognition
import pyodbc
import configparser
import mysql.connector
import numpy as np
from ie_module import Module
from utils import resize_input
from openvino.runtime import PartialShape
import cv2
import numpy as np
import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import time
import cv2
import numpy as np
from openvino.runtime import Core, get_version
import pickle 
# Sử dụng socket để gửi thông tin
import socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))
from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
import monitors
from pathlib import Path
from setuptools import setup, find_packages
from helpers import resolution
from images_capture import open_images_capture
import os
import sys
from pathlib import Path
import copy
from time import perf_counter
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.models.utils import resize_image
import time
from utils import cut_rois, resize_input
from ie_module import Module
import logging as log
import os
from io import StringIO
import sys
original_stdout = sys.stdout
import os.path as osp
import requests
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import pickle
from face_detector import FaceDetector
import logging as log
from openvino.runtime import AsyncInferQueue
import numpy as np
from utils import cut_rois, resize_input
from ie_module import Module
import tkinter as tk
from tkinter import ttk
import configparser
from tkinter import StringVar
from tkinter import messagebox
from connect_db import Config_db
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import StringVar
import io
import os
import cv2
from face_recognition import FaceRecognition
from ui_conect_db import UIConect_db
import pickle
from openvino.runtime import Core, get_version
import numpy as np
from model_api.models.utils import resize_image
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import StringVar
import io
import os
import cv2
from face_recognition import FaceRecognition
from ui_conect_db import UIConect_db
import pickle
from openvino.runtime import Core, get_version
core = Core()
import random
random_number = random.randint(3, 28)
import os
import logging as log
import datetime
import hashlib
import configparser

UILogin = UILogin()
UILogin.run()