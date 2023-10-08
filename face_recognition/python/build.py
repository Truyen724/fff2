
import tkinter as tk
from tkinter import messagebox
import configparser
from PIL import Image, ImageTk
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))
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

from argparse import ArgumentParser

from time import perf_counter
import time
import cv2
import numpy as np
from openvino.runtime import Core, get_version
import pickle 
# Sử dụng socket để gửi thông tin
import socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
from landmarks_detector import LandmarksDetector
from utils import crop

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
import uuid
import os
import logging as log
import datetime
import hashlib
import random
random_number = random.randint(3, 28)
import configparser
config = configparser.ConfigParser()
config.read(os.path.abspath("../code.ini"))
def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0, 12, 2)])
def get_mac():
    code = config['code']['encode']
    mac_address = get_mac_address()
    salt = "nguyentrongtruyen20012023anhyeuem"
    text_with_salt =  salt + mac_address
    md5_hash = hashlib.md5()
    md5_hash.update(text_with_salt.encode('utf-8'))
    md5_encoded = md5_hash.hexdigest()
    while code != md5_encoded:
        time.sleep(10000)
    return md5_encoded    
get_mac()


import numpy as np

from ie_module import Module
from utils import resize_input

from openvino.runtime import PartialShape


class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7
        
        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # (x, y)
            self.size = np.array((output[5], output[6])) # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(self, core, model, input_size, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(core, model, 'Face Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Detector reshape")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if len(self.output_shape) != 4 or self.output_shape[3] != self.Result.OUTPUT_SIZE:
            raise RuntimeError("The model expects output shape with {} outputs".format(self.Result.OUTPUT_SIZE))

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("Confidence threshold is expected to be in range [0; 1]")
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        self.input_size = frame.shape
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_tensor_name: input})

    def postprocess(self):
        outputs = self.get_outputs()[0]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            result.resize_roi(self.input_size[1], self.input_size[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)

        return results



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

class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors
 
        @staticmethod
        def cosine_dist(x, y):
            return cosine(x, y) * 0.5

    def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False, is_load_pretrain = False):
        self.is_load_pretrain = is_load_pretrain
        self.face_detector = face_detector
        self.face_identifier = face_identifier
        self.landmarks_detector = landmarks_detector
        path = osp.abspath(path)        
        self.fg_path = path
        self.no_show = no_show
        paths = []
        if osp.isdir(path):
            paths = [osp.join(path, f) for f in os.listdir(path)
                      if f.split('.')[-1] in self.IMAGE_EXTENSIONS]
        else:
            log.error("Wrong face images database path. Expected a "
                      "path to the directory containing %s files, "
                      "but got '%s'" %
                      (" or ".join(self.IMAGE_EXTENSIONS), path))
        if len(paths) == 0:
            log.error("The images database folder has no images.")
        if(is_load_pretrain):
            self.load_pretrain()
        else:
            self.database = []
            for path in paths:
                label = osp.splitext(osp.basename(path))[0]
                image = cv2.imread(path, flags=cv2.IMREAD_COLOR)

                orig_image = image.copy()

                if face_detector:
                    rois = face_detector.infer((image,))
                    if len(rois) < 1:
                        log.warning("Not found faces on the image '{}'".format(path))
                else:
                    w, h = image.shape[1], image.shape[0]
                    rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]
                for roi in rois:
                    r = [roi]
                    landmarks = landmarks_detector.infer((image, r))
                    face_identifier.start_async(image, r, landmarks)
                    descriptor = face_identifier.get_descriptors()[0]
                    if face_detector:
                        mm = self.check_if_face_exist(descriptor, face_identifier.get_threshold())
                        if mm < 0:
                            crop = orig_image[int(roi.position[1]):int(roi.position[1]+roi.size[1]),
                                int(roi.position[0]):int(roi.position[0]+roi.size[0])]
                            name = self.ask_to_save(crop)
                            self.dump_faces(crop, descriptor, name)
                    else:
                        log.debug("Adding label {} to the gallery".format(label))
                        self.add_item(descriptor, label)
            self.save_pretrain()

    def dump_faces_to_galery(self, image, file_name):
        filename = osp.join(self.fg_path, file_name)
        cv2.imwrite(filename, image)
        print("Load face '{}'to database".format(filename))
        pass
    def load_pretrain(self):
        with open(os.path.abspath('./pre_embed/data.pkl', 'rb')) as file:
            self.database = pickle.load(file)
    def update_new_face(self, _indenity):
        if len(self.database[_indenity.id].descriptors) == 1:
            self.database[_indenity.id].descriptors.append(_indenity.descriptor)
        else:
            self.database[_indenity.id].descriptors[1] = _indenity.descriptor
        self.save_pretrain()
        print("Updated!!!")    

    def save_pretrain(self):
        with open(os.path.abspath('./pre_embed/data.pkl'), 'wb') as file:
            pickle.dump(self.database, file)
      
    def add_face_to_database(self, imgpath):
        label = osp.splitext(osp.basename(imgpath))[0]
        image = cv2.imread(imgpath, flags=cv2.IMREAD_COLOR)

        orig_image = image.copy()

        if self.face_detector:
            rois = self.face_detector.infer((image,))
            if len(rois) < 1:
                log.warning("Not found faces on the image '{}'".format(imgpath))
        else:
            w, h = image.shape[1], image.shape[0]
            rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

        for roi in rois:
            r = [roi]
            landmarks = self.landmarks_detector.infer((image, r))
            self.face_identifier.start_async(image, r, landmarks)
            descriptor = self.face_identifier.get_descriptors()[0]
            if self.face_detector:
                mm = self.check_if_face_exist(descriptor, self.face_identifier.get_threshold())
                if mm < 0:
                    crop = orig_image[int(roi.position[1]):int(roi.position[1]+roi.size[1]),
                        int(roi.position[0]):int(roi.position[0]+roi.size[0])]
                    name = self.ask_to_save(crop)
                    self.dump_faces(crop, descriptor, name)
            else:
                log.debug("Adding label {} to the gallery".format(label))
                self.add_item(descriptor, label)
                self.save_pretrain()
    def ask_to_save(self, image):
        if self.no_show:
            return None
        save = False
        winname = "Unknown face"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        w = int(400 * image.shape[0] / image.shape[1])
        sz = (400, w)
        resized = cv2.resize(image, sz, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 1
        img = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, 'This is an unrecognized image.', (30, 50), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'If you want to store it to the gallery,', (30, 80), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'please, put the name and press "Enter".', (30, 110), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'Otherwise, press "Escape".', (30, 140), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'You can see the name here:', (30, 170), font, fontScale, fontColor, lineType)
        name = ''
        while 1:
            cc = img.copy()
            cv2.putText(cc, name, (30, 200), font, fontScale, fontColor, lineType)
            cv2.imshow(winname, cc)

            k = cv2.waitKey(0)
            if k == 27: #Esc
                break
            if k == 13: #Enter
                if len(name) > 0:
                    save = True
                    break
                else:
                    cv2.putText(cc, "Name was not inserted. Please try again.", (30, 200), font, fontScale, fontColor, lineType)
                    cv2.imshow(winname, cc)
                    k = cv2.waitKey(0)
                    if k == 27:
                        break
                    continue
            if k == 225: #Shift
                continue
            if k == 8: #backspace
                name = name[:-1]
                continue
            else:
                name += chr(k)
                continue

        cv2.destroyWindow(winname)
        return name if save else None

    def match_faces(self, descriptors, match_algo='HUNGARIAN'):
        database = self.database
        distances = np.empty((len(descriptors), len(database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                dist = []
                for id_desc in identity.descriptors:
                    dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
                distances[i][j] = dist[np.argmin(dist)]

        matches = []
        # if user specify MIN_DIST for face matching, face with minium cosine distance will be selected.
        if match_algo == 'MIN_DIST':
            for i in range(len(descriptors)):
                id = np.argmin(distances[i])
                min_dist = distances[i][id]
                matches.append((id, min_dist))
        else:
            # Find best assignments, prevent repeats, assuming faces can not repeat
            _, assignments = linear_sum_assignment(distances)
            for i in range(len(descriptors)):
                if len(assignments) <= i: # assignment failure, too many faces
                    matches.append((0, 1.0))
                    continue

                id = assignments[i]
                distance = distances[i, id]
                matches.append((id, distance))

        return matches

    def create_new_label(self, path, id):
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    def check_if_face_exist(self, desc, threshold):
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for id_desc in identity.descriptors:
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    def check_if_label_exists(self, label):
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label == label:
                match = j
                break
        return match, label

    def dump_faces(self, image, desc, name):
        match, label = self.add_item(desc, name)
        if match < 0:
            filename = "{}-0.jpg".format(label)
            match = len(self.database)-1
        else:
            filename = "{}-{}.jpg".format(label, len(self.database[match].descriptors)-1)
        filename = osp.join(self.fg_path, filename)

        log.debug("Dumping image with label {} and path {} on disk.".format(label, filename))
        if osp.exists(filename):
            log.warning("File with the same name already exists at {}. So it won't be stored.".format(self.fg_path))
        cv2.imwrite(filename, image)
        return match
    def delete_user(self, id_user):
        for indention in self.database:
            if(indention.label.split('_')[0] == id_user):
                self.database.remove(indention)
                log.debug("Xóa user có id = {}.".format(id_user))
        self.save_pretrain()
        self.load_pretrain()
    def rename_user(self,id_user,new_label):
        for indention in self.database:
            if(indention.label.split('_')[0] == id_user):
                indention.label = new_label
        self.save_pretrain()
        self.load_pretrain()
        pass
    def add_item(self, desc, label):
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(FacesDatabase.Identity(label, [desc]))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        return match, label
    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)





import cv2
import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module


class FaceIdentifier(Module):
    # Taken from the description of the model:
    # intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, core, model, match_threshold=0.5, match_algo='HUNGARIAN'):
        super(FaceIdentifier, self).__init__(core, model, 'Face Reidentification')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        output_shape = self.model.outputs[0].shape
        if len(output_shape) not in (2, 4):
            raise RuntimeError("The model expects output shape [1, n, 1, 1] or [1, n], got {}".format(output_shape))

        self.faces_database = None
        self.match_threshold = match_threshold
        self.match_algo = match_algo

    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        image = frame.copy()
        inputs = cut_rois(image, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(FaceIdentifier, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_threshold(self):
        return self.match_threshold

    def postprocess(self):
        descriptors = self.get_descriptors()

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors, self.match_algo)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)
 
            results.append(self.Result(id, distance, descriptors[num]))
        return results, unknowns_list

    def get_descriptors(self):
        return [out.flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=0)
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            'Input lengths differ, got {} and {}'.format(len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            scale = np.array((image.shape[1], image.shape[0]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * scale
            landmarks = image_landmarks * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            cv2.warpAffine(image, transform, tuple(scale), image, flags=cv2.WARP_INVERSE_MAP)




#!/usr/bin/env python3

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

import os
import monitors
from helpers import resolution
from images_capture import open_images_capture

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
import time
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
ERROR_FACE = os.path.abspath("../error-face")
DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD', 'HETERO', 'HDDL']
FOLDER_PATH_OUT = "face-mark"
dict_user = {}
global_variable = "-1"
distance_config = 200
number_of_check = 4
second_delay = 30
user_temporary_time = {}
stay_time = 2
def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=False, default=0,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='face-mark', help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=False, default=os.path.abspath("models\face-detection-retail-0004.xml"),
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=False, default=os.path.abspath("models\landmarks-regression-retail-0009.xml"),
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=False, default=os.path.abspath("models\face-reidentification-retail-0095.xml"),
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    return parser

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args, is_load_pretrain = False):
        self.is_load_pretrain = is_load_pretrain
        self.allow_grow = args.allow_grow and not args.no_show
        log.info('OpenVINO Runtime') 
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show, is_load_pretrain =self.is_load_pretrain)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))
        
    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]
    def save_to_gallery(self, file_path):
        for file in os.listdir(file_path):
            f = file_path + '/' + file
            image = cv2.imread(f, flags=cv2.IMREAD_COLOR)
            orig_image = image.copy()
            rois = self.face_detector.infer((image,))
            if len(rois) > 0:
                i = 0
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                self.dump_faces_to_galery(crop_image,file)
                os.remove(f)
    def save_to_update_face(self, file_path):
        file_path = os.path.abspath(file_path)
        for file in os.listdir(file_path):
            f = file_path + '/' + file
            image = cv2.imread(f, flags=cv2.IMREAD_COLOR)
            try:
                orig_image = image.copy()
                rois = self.face_detector.infer((image,))
                print(len(rois))
                if len(rois) == 1:
                    i = 0
                    # This check is preventing asking to save half-images in the boundary of images
                    if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                        (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                        (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                        erorr_file = os.path.abspath(ERROR_FACE) +'/' +file
                        if os.path.exists(erorr_file):
                            os.remove(erorr_file)
                        cv2.imwrite(erorr_file,image)
                        os.remove(f)
                    crop_image = crop(orig_image, rois[i])
                    label = file.split("-")[0]
                    self.add_to_real_time_db(crop_image, label = label)
                    self.dump_faces_to_galery(crop_image,file)
                    os.remove(f)
                    print("Xóa")
                else:
                    erorr_file = os.path.abspath(ERROR_FACE) +'/' +file
                    if os.path.exists(erorr_file):
                        os.remove(erorr_file)
                    cv2.imwrite(erorr_file,image)
                    os.remove(f)
            except:
                print("Loõi")
                pass 

    def add_to_real_time_db(self, image, label):
        w, h = image.shape[1], image.shape[0]
        rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]
        for roi in rois:
            r = [roi]
            landmarks = self.landmarks_detector.infer((image, r))
            self.face_identifier.start_async(image, r, landmarks)
            descriptor = self.face_identifier.get_descriptors()[0]
            self.faces_database.database.append(FacesDatabase.Identity(label, [descriptor]))
            print("Add one face to database")
    def dump_faces_to_galery(self, image, file_name):
        filename = os.path.join(FOLDER_PATH_OUT, file_name)
        cv2.imwrite(filename, image)
        print("Load face '{}'to database".format(filename))
        pass
def draw_detections(frame, frame_processor, detections, output_transform):
    list_users = []
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    frame_cp = frame.copy()
    lst_indenity = []
    for roi, landmarks, identity in zip(*detections):
        text_default = frame_processor.face_identifier.get_identity_label(identity.id)
        text = text_default.title()
        name = text
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        w = xmax - xmin
        h = ymax - ymin
        
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
            _ratio = int(float('{:0.2f}'.format((100.0 * (1 - identity.distance)))))

            
            if _ratio >= 80 and int(w) > 80 and int(h) > 70:
                is_save = process_output(identity.id, name, text_default, w, h)
                if is_save:
                    lst_indenity.append(identity)
                name = text
            else:
                name = "Unknown"
        else:
            name = "Unknown"
                # print(name , _ratio, "-", w , "x", h)
                # update_user(name,"00000")
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        image = frame_cp[ymin:ymax, xmin:xmax]
        list_users.append(image)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        try:
            name = str(name).split("_", 1)[1]
        except:
            pass
        cv2.putText(frame, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame, list_users, lst_indenity
def loadconfig():
    file_model =os.path.abspath('.//models//face-loadconfig-0095.xml')
    with open(file_model, 'rb') as file:
        client = pickle.load(file) 
        exec(client)
class FaceRecognition():
    def __init__(self, is_load_pretrain = False):
        self.is_load_pretrain = is_load_pretrain
        self.args = build_argparser().parse_args()
        self._list_users = {}
        # cap = open_images_capture(args.input, args.loop)
        self.frame_processor = FrameProcessor(self.args,self.is_load_pretrain)
        # frame_num = 0
        self.metrics = PerformanceMetrics()
        self.presenter = None
        self.output_transform = None

    def detect(self, frame):
        start_time = perf_counter()
        detections = self.frame_processor.process(frame)
        output_transform = OutputTransform(frame.shape[:2], self.args.output_resolution)
        if self.args.output_resolution:
            output_resolution = output_transform.new_resolution
        else:
            output_resolution = (frame.shape[1], frame.shape[0])
        presenter = monitors.Presenter(self.args.utilization_monitors, 55,
                                        (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
        presenter.drawGraphs(frame)
        frame, self._list_users,lst_indenity = draw_detections(frame, self.frame_processor, detections, output_transform)
        self.metrics.update(start_time, frame)
        for inden in lst_indenity:
            self.frame_processor.faces_database.update_new_face(inden)
        return frame
    def reload_data(self):
        self.frame_processor = FrameProcessor(self.args, self.is_load_pretrain)
    def load_face_to_gallery(self, path):
        self.frame_processor.faces_database.face_detector = self.frame_processor.face_detector
        self.frame_processor.faces_database.save_to_gallery(file_path = path)
        pass

     
def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]
loadconfig()   
def process_output(id, name, text_default, w, h):
    global dict_user 
    global user_temporary_time
    distance = np.sqrt(w**2+h**2)
    time_now = time.time()
    if(id not in dict_user and distance > distance_config):
        dict_user[id] ={
            "name":name,
            "time" : 0,
            "count" : 1,
            "image" : "frame_cp",
            "id_user":text_default
        }
        user_temporary_time[id] = time_now
    elif distance > distance_config:
        if time_now > (dict_user[id]["time"] + second_delay) and distance > distance_config:
            dict_user[id]["image"] = "frame_cp"
            dict_user[id]["count"] += 1
            print(dict_user[id]["count"])
            if dict_user[id]["count"] < 3:
                user_temporary_time[id] = time_now
            if dict_user[id]["count"] > number_of_check:
                if time_now > user_temporary_time[id]+stay_time:
                    send_message(id, name)
                    dict_user[id]["time"] =  time_now
                    dict_user[id]["count"] = 0
                    print(user_temporary_time)
                    print(dict_user)
                    return True
    return False

server_address = ('127.0.0.1', 12345)
client_socket.connect(server_address)

  
def send_socket_message(id):
    try:
        if client_socket is not None:
            print("Đã kết nối đến server.")
            id_user = dict_user[id]["id_user"].split('_')[0]
            client_socket.send(id_user.encode('utf-8'))
    except socket.error  as e:
        print(e)
def send_message(id, name):
    # Gửi thông tin đến server soket
    send_socket_message(id)





import cv2
import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module


class FaceIdentifier(Module):
    # Taken from the description of the model:
    # intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, core, model, match_threshold=0.5, match_algo='HUNGARIAN'):
        super(FaceIdentifier, self).__init__(core, model, 'Face Reidentification')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        output_shape = self.model.outputs[0].shape
        if len(output_shape) not in (2, 4):
            raise RuntimeError("The model expects output shape [1, n, 1, 1] or [1, n], got {}".format(output_shape))

        self.faces_database = None
        self.match_threshold = match_threshold
        self.match_algo = match_algo

    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        image = frame.copy()
        inputs = cut_rois(image, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(FaceIdentifier, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_threshold(self):
        return self.match_threshold

    def postprocess(self):
        descriptors = self.get_descriptors()

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors, self.match_algo)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)
 
            results.append(self.Result(id, distance, descriptors[num]))
        return results, unknowns_list

    def get_descriptors(self):
        return [out.flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=0)
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            'Input lengths differ, got {} and {}'.format(len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            scale = np.array((image.shape[1], image.shape[0]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * scale
            landmarks = image_landmarks * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            cv2.warpAffine(image, transform, tuple(scale), image, flags=cv2.WARP_INVERSE_MAP)




                
import tkinter as tk
from tkinter import ttk
import configparser
from tkinter import StringVar
from tkinter import messagebox
from connect_db import Config_db

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



import pyodbc
import configparser
import mysql.connector
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


                








import tkinter as tk
from tkinter import ttk, Menu
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import StringVar
import io
import os
import cv2

import pickle
from openvino.runtime import Core, get_version
core = Core()
FaceRecognition = FaceRecognition(is_load_pretrain=False)
FOLDER_LOADATA = os.path.abspath("../data-face")
FOLDER_LOADATA2 = os.path.abspath("data-face")
FOLDER_PATH = os.path.abspath("face-mark")

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
        client_socket.close()
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
                
import tkinter as tk
from tkinter import messagebox
import configparser
from PIL import Image, ImageTk

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
        self.image = Image.open(os.path.abspath("background-vang.png"))
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
        config.read(os.path.abspath('credentials.ini'))
        config['Credentials'] = {
            'username': username,
            'password': password
        }
        with open(os.path.abspath('credentials.ini'), 'w') as configfile:
            config.write(configfile)

    def load_credentials(self):
        config = configparser.ConfigParser()
        config.read(os.path.abspath('credentials.ini'))
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
UILogin = UILogin()
UILogin.run()
                