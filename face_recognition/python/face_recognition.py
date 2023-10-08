
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
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
import os
import monitors
from helpers import resolution
from images_capture import open_images_capture

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
import time
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
ERROR_FACE = "../error-face"
DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD', 'HETERO', 'HDDL']
FOLDER_PATH_OUT = "face-mark"
dict_user = {}
global_variable = "-1"
distance_config = 200
number_of_check = 3
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
    models.add_argument('-m_fd', type=Path, required=False, default=os.path.abspath("models/face-detection-retail-0004.xml"),
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=False, default=os.path.abspath("models/landmarks-regression-retail-0009.xml"),
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=False, default=os.path.abspath("models/face-reidentification-retail-0095.xml"),
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
# loadconfig()   
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
