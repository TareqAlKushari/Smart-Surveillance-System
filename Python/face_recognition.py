import os
import argparse
import requests
import logging
import imghdr
import pickle
import tarfile
from functools import partial

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from prettytable import PrettyTable
from PIL import Image, ImageDraw, ImageFont
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

font_path = os.path.join(
            os.path.abspath(os.path.dirname("SourceHanSansCN-Medium.otf")),
            "SourceHanSansCN-Medium.otf")

def check_model_file(model):
    """Check the model files exist and download and untar when no exist.
    """
    print("model_path: ", model)
    
    model_map = {
        "ArcFace": "arcface_iresnet50_v1.0_infer",
        "BlazeFace": "blazeface_fpn_ssh_1000e_v1.0_infer",
        "MobileFace": "mobileface_v1.0_infer",
        "PPYOLOEFace": "PP-YOLOE_plus-S_face_infer",
        "ResNet50Face": "ResNet50_face_infer"
    }

    if os.path.isdir(model):
        model_file_path = os.path.join(model, "inference.pdmodel")
        params_file_path = os.path.join(model, "inference.pdiparams")
        if not os.path.exists(model_file_path) or not os.path.exists(
                params_file_path):
            raise Exception(
                f"The specifed model directory error. The drectory must include \"inference.pdmodel\" and \"inference.pdiparams\"."
            )

#     elif model in model_map:
#         storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
#                                     model)
#         url = BASE_DOWNLOAD_URL.format(model_map[model])

#         tar_file_name_list = [
#             "inference.pdiparams", "inference.pdiparams.info",
#             "inference.pdmodel"
#         ]
#         model_file_path = storage_directory("inference.pdmodel")
#         params_file_path = storage_directory("inference.pdiparams")
#         if not os.path.exists(model_file_path) or not os.path.exists(
#                 params_file_path):
#             tmp_path = storage_directory(url.split("/")[-1])
#             logging.info(f"Download {url} to {tmp_path}")
#             os.makedirs(storage_directory(), exist_ok=True)
#             download_with_progressbar(url, tmp_path)
#             with tarfile.open(tmp_path, "r") as tarObj:
#                 for member in tarObj.getmembers():
#                     filename = None
#                     for tar_file_name in tar_file_name_list:
#                         if tar_file_name in member.name:
#                             filename = tar_file_name
#                     if filename is None:
#                         continue
#                     file = tarObj.extractfile(member)
#                     with open(storage_directory(filename), "wb") as f:
#                         f.write(file.read())
#             os.remove(tmp_path)
#         if not os.path.exists(model_file_path) or not os.path.exists(
#                 params_file_path):
#             raise Exception(
#                 f"Something went wrong while downloading and unzip the model[{model}] files!"
#             )
    else:
        raise Exception(
            f"The specifed model name error. Support \"BlazeFace\" for detection and \"ArcFace\" and \"MobileFace\" for recognition. And support local directory that include model files (\"inference.pdmodel\" and \"inference.pdiparams\")."
        )

    return model_file_path, params_file_path

def normalize_image(img, scale=None, mean=None, std=None, order="chw"):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype("float32")
    std = np.array(std).reshape(shape).astype("float32")

    if isinstance(img, Image.Image):
        img = np.array(img)

    assert isinstance(img,
                      np.ndarray), "invalid input \"img\" in NormalizeImage"
    return (img.astype("float32") * scale - mean) / std

def draw(img, box_list, labels):
    color_map = ColorMap(100)
    color_map.update(labels)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)

    for i, dt in enumerate(box_list):
        bbox, score = dt[2:], dt[1]
        label = labels[i]
        color = tuple(color_map[label])

        xmin, ymin, xmax, ymax = bbox

        font_size = max(int((xmax - xmin) // 6), 10)
        font = ImageFont.truetype(font_path, font_size)

        text = "{} {:.4f}".format(label, score)
        th = sum(font.getmetrics())
#             tw = font.getsize(text)[0]
        left, top, right, bottom = font.getbbox(text)
#             width, height = right - left, bottom - top
        tw = right - left
        
        start_y = max(0, ymin - th)

        draw.rectangle(
            [(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
        draw.text(
            (xmin + 1, start_y),
            text,
            fill=(255, 255, 255),
            font=font,
            anchor="la")
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)], width=2, outline=color)
    return np.array(im)

class ColorMap(object):
    def __init__(self, num):
        super().__init__()
        self.get_color_map_list(num)
        self.color_map = {}
        self.ptr = 0

    def __getitem__(self, key):
        return self.color_map[key]

    def update(self, keys):
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        self.color_list = [
            color_map[i:i + 3] for i in range(0, len(color_map), 3)
        ]

class BasePredictor(object):
    def __init__(self, predictor_config):
        super().__init__()
        self.predictor_config = predictor_config
        self.predictor, self.input_names, self.output_names = self.load_predictor(
            predictor_config["model_file"], predictor_config["params_file"])

    def load_predictor(self, model_file, params_file):
        config = Config(model_file, params_file)
        if self.predictor_config["use_gpu"]:
            config.enable_use_gpu(200, 0)
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.predictor_config[
                "cpu_threads"])

            if self.predictor_config["enable_mkldnn"]:
                try:
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                    config.enable_mkldnn()
                except Exception as e:
                    logging.error(
                        "The current environment does not support `mkldnn`, so disable mkldnn."
                    )
        config.disable_glog_info()
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        return predictor, input_names, output_names

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError
    
class DetectorFace(BasePredictor):
    def __init__(self, det_config, predictor_config):
        super().__init__(predictor_config)
        self.det_config = det_config
        self.target_size = self.det_config["target_size"]
        self.thresh = self.det_config["thresh"]

    def preprocess(self, img):
        resize_h, resize_w = self.target_size
        img_shape = img.shape
        img_scale_x = resize_w / img_shape[1]
        img_scale_y = resize_h / img_shape[0]
        img = cv2.resize(
            img, None, None, fx=img_scale_x, fy=img_scale_y, interpolation=1)
        img = normalize_image(
            img,
            scale=1. / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc")
        img_info = {}
        img_info["im_shape"] = np.array(
            img.shape[:2], dtype=np.float32)[np.newaxis, :]
        img_info["scale_factor"] = np.array(
            [img_scale_y, img_scale_x], dtype=np.float32)[np.newaxis, :]

        img = img.transpose((2, 0, 1)).copy()
        img_info["image"] = img[np.newaxis, :, :, :]
        return img_info

    def postprocess(self, np_boxes):
        expect_boxes = (np_boxes[:, 1] > self.thresh) & (np_boxes[:, 0] > -1)
        return np_boxes[expect_boxes, :]

    def predict(self, img):
        print("Predict_1 start")
        inputs = self.preprocess(img)
        for input_name in self.input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])
        self.predictor.run()
        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        np_boxes = output_tensor.copy_to_cpu()
        # boxes_num = self.detector.get_output_handle(self.detector_output_names[1])
        # np_boxes_num = boxes_num.copy_to_cpu()
        box_list = self.postprocess(np_boxes)
        print("Predict_1 done")
        return box_list
    
class RecognizerFace(BasePredictor):
    def __init__(self, rec_config, predictor_config):
        super().__init__(predictor_config)
        if rec_config["index"] is not None:
            if rec_config["build_index"] is not None:
                raise Exception(
                    "Only one of --index and --build_index can be set!")
            self.load_index(rec_config["index"])
        elif rec_config["build_index"] is None:
            raise Exception("One of --index and --build_index have to be set!")
        self.rec_config = rec_config
        self.cdd_num = self.rec_config["cdd_num"]
        self.thresh = self.rec_config["thresh"]
        self.max_batch_size = self.rec_config["max_batch_size"]

    def preprocess(self, img, box_list=None):
        img = normalize_image(
            img,
            scale=1. / 255.,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            order="hwc")
        if box_list is None:
            height, width = img.shape[:2]
            box_list = [np.array([0, 0, 0, 0, width, height])]
        batch = []
        input_batches = []
        cnt = 0
        for idx, box in enumerate(box_list):
            box[box < 0] = 0
            xmin, ymin, xmax, ymax = list(map(int, box[2:]))
            face_img = img[ymin:ymax, xmin:xmax, :]
            face_img = cv2.resize(face_img, (112, 112)).transpose(
                (2, 0, 1)).copy()
            batch.append(face_img)
            cnt += 1
            if cnt % self.max_batch_size == 0 or (idx + 1) == len(box_list):
                input_batches.append(np.array(batch))
                batch = []
        return input_batches

    def postprocess(self):
        pass

    def retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.index_feature,
                                           feature).squeeze()
            abs_similarity = np.abs(similarity)
            candidate_idx = np.argpartition(abs_similarity,
                                            -self.cdd_num)[-self.cdd_num:]
            remove_idx = np.where(abs_similarity[candidate_idx] < self.thresh)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            candidate_label_list = list(np.array(self.label)[candidate_idx])
            if len(candidate_label_list) == 0:
                maxlabel = "unknown"
            else:
                maxlabel = max(candidate_label_list,
                               key=candidate_label_list.count)
            labels.append(maxlabel)
        return labels

    def load_index(self, file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        self.label = index["label"]
        self.index_feature = np.array(index["feature"]).squeeze()

    def predict(self, img, box_list=None):
        batch_list = self.preprocess(img, box_list)
        feature_list = []
        for batch in batch_list:
            for input_name in self.input_names:
                input_tensor = self.predictor.get_input_handle(input_name)
                input_tensor.copy_from_cpu(batch)
            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(self.output_names[
                0])
            np_feature = output_tensor.copy_to_cpu()
            feature_list.append(np_feature)
        return np.array(feature_list)




class InsightFace(object):
    def __init__(self, 
                 det=False,
                 rec=False,
                 det_model=None,
                 rec_model=None,
                 index=None,
                 input=None,
                 output=None,
                 build_index=None,
                 img_dirt=None,
                 label_file=None,
                 build=False,
                 print_info=True):
        super().__init__()
        use_gpu = True
        enable_mkldnn = False
        cpu_threads = 1
        det_thresh = 0.8
        cdd_num = 5
        rec_thresh = 0.45
        max_batch_size = 1

        self.font_path = os.path.join(
            os.path.abspath(os.path.dirname("../env/SourceHanSansCN-Medium.otf")),
            "../env/SourceHanSansCN-Medium.otf")
        
        print("det", det)
        print("rec", rec)
        print("det_model", det_model)
        print("rec_model", rec_model)
        print("index", index)
        print("input", input)
        print("output", output)
        print("build_index", build_index)
        
        # build index
        if build_index:
            if rec or det:
                warning_str = f"Only one of --rec (or --det) and --build_index can be set!"
                raise Exception(warning_str)
            if img_dirt is None or label_file is None:
                raise Exception(
                    "Please specify the --img_dir and --label when build index."
                )
            self.init_rec(rec_model, max_batch_size, rec_thresh, index, build_index, cdd_num, use_gpu, enable_mkldnn, cpu_threads)

        # detection
        if det:
            print("detection")
            self.init_det(det_model, cpu_threads, enable_mkldnn, use_gpu, det_thresh)
            
        # recognition
        if rec:
            if not index:
                warning_str = f"The index file must be specified when recognition! "
                if det:
                    logging.warning(warning_str + "Detection only!")
                else:
                    raise Exception(warning_str)
            elif not os.path.isfile(index):
                warning_str = f"The index file not found! Please check path of index: \"{index}\". "
                if det:
                    logging.warning(warning_str + "Detection only!")
                else:
                    raise Exception(warning_str)
            else:
                self.init_rec(rec_model, max_batch_size, rec_thresh, index, build_index, cdd_num, use_gpu, enable_mkldnn, cpu_threads)

        if not build_index and not det and not rec:
            raise Exception(
                "Specify at least the detection(--det) or recognition(--rec) or --build_index!"
            )
    
    def init_rec(self, rec_model, max_batch_size, rec_thresh, index, build_index, cdd_num, use_gpu, enable_mkldnn, cpu_threads):
        rec_config = {
            "max_batch_size": max_batch_size,
            "resize": 112,
            "thresh": rec_thresh,
            "index": index,
            "build_index": build_index,
            "cdd_num": cdd_num
        }
        rec_predictor_config = {
            "use_gpu": use_gpu,
            "enable_mkldnn": enable_mkldnn,
            "cpu_threads": cpu_threads
        }
        model_file_path, params_file_path = check_model_file(rec_model)
        rec_predictor_config["model_file"] = model_file_path
        rec_predictor_config["params_file"] = params_file_path
        self.rec_predictor = Recognizer(rec_config, rec_predictor_config)

    def init_det(self, det_model, cpu_threads, enable_mkldnn, use_gpu, det_thresh):
        print("init_det start")
        det_config = {"thresh": det_thresh, "target_size": [640, 640]}
        det_predictor_config = {
            "use_gpu": use_gpu,
            "enable_mkldnn": enable_mkldnn,
            "cpu_threads": cpu_threads
        }
        model_file_path, params_file_path = check_model_file(det_model)
        det_predictor_config["model_file"] = model_file_path
        det_predictor_config["params_file"] = params_file_path
        print("model_file", det_predictor_config["model_file"])
        print("params_file", det_predictor_config["params_file"])
        self.det_predictor = Detector(det_config, det_predictor_config)

        # TODO(gaotingquan): now only support fixed number of color
        self.color_map = ColorMap(100)
        print("init_det done")

    def preprocess(self, img):
        img = img.astype(np.float32, copy=False)
        return img

    def draw(self, img, box_list, labels):
        self.color_map.update(labels)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        for i, dt in enumerate(box_list):
            bbox, score = dt[2:], dt[1]
            label = labels[i]
            color = tuple(self.color_map[label])

            xmin, ymin, xmax, ymax = bbox

            font_size = max(int((xmax - xmin) // 6), 10)
            font = ImageFont.truetype(self.font_path, font_size)

            text = "{} {:.4f}".format(label, score)
            th = sum(font.getmetrics())
#             tw = font.getsize(text)[0]
            left, top, right, bottom = font.getbbox(text)
#             width, height = right - left, bottom - top
            tw = right - left
            
            start_y = max(0, ymin - th)

            draw.rectangle(
                [(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
            draw.text(
                (xmin + 1, start_y),
                text,
                fill=(255, 255, 255),
                font=font,
                anchor="la")
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)], width=2, outline=color)
        return np.array(im)

    def predict_np_img(self, img):
        input_img = self.preprocess(img)
        box_list = None
        np_feature = None
        if hasattr(self, "det_predictor"):
            box_list = self.det_predictor.predict(input_img)
        if hasattr(self, "rec_predictor"):
            np_feature = self.rec_predictor.predict(input_img, box_list)
#         print("input_img: ", input_img)
#         print("box_list: ", box_list)
#         print("np_feature: ", np_feature)
        return box_list, np_feature

    def init_reader_writer(self, input_data):
        if isinstance(input_data, np.ndarray):
            self.input_reader = ImageReader(input_data)
            if hasattr(self, "det_predictor"):
                self.output_writer = ImageWriter(output)
        elif isinstance(input_data, str):
            if input_data.endswith("mp4"):
                self.input_reader = VideoReader(input_data)
                info = self.input_reader.get_info()
                self.output_writer = VideoWriter(output, info)
            else:
                self.input_reader = ImageReader(input_data)
                if hasattr(self, "det_predictor"):
                    self.output_writer = ImageWriter(output)
        else:
            raise Exception(
                f"The input data error. Only support path of image or video(.mp4) and dirctory that include images."
            )

    def predict(self, input_data, print_info=False):
        """Predict input_data.
        Args:
            input_data (str | NumPy.array): The path of image, or the derectory including images, or the image data in NumPy.array format.
            print_info (bool, optional): Wheather to print the prediction results. Defaults to False.
        Yields:
            dict: {
                "box_list": The prediction results of detection.
                "features": The output of recognition.
                "labels": The results of retrieval.
                }
        """
        print("Predict start")
        self.init_reader_writer(input_data)
        for img, file_name in self.input_reader:
            if img is None:
                logging.warning(f"Error in reading img {file_name}! Ignored.")
                continue
#             print("img: ", img)
            box_list, np_feature = self.predict_np_img(img)
            print("box_list_2: ", box_list)
            print("np_feature_2: ", np_feature)
            if np_feature is not None:
                labels = self.rec_predictor.retrieval(np_feature)
            else:
                labels = ["face"] * len(box_list)
            if box_list is not None:
                result = self.draw(img, box_list, labels=labels)
                self.output_writer.write(result, file_name)
            if print_info:
                logging.info(f"File: {file_name}, predict label(s): {labels}")
            yield {
                "box_list": box_list,
                "features": np_feature,
                "labels": labels
            }
        completion_tip = "Predict complete! "
        if output and hasattr(self, "det_predictor"):
            completion_tip += f"All prediction result(s) have been saved in \"{output}\"."
        logging.info(completion_tip)
        print("Predict done")
#         return result

    def build_index(self):
        img_dir = img_dirt
        label_path = label_file
        with open(label_path, "r") as f:
            sample_list = f.readlines()

        feature_list = []
        label_list = []

        for idx, sample in enumerate(sample_list):
            name, label = sample.strip().split(" ")
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                logging.warning(f"Error in reading img {name}! Ignored.")
                continue
            box_list, np_feature = self.predict_np_img(img)
            feature_list.append(np_feature[0])
            label_list.append(label)

            if idx % 100 == 0:
                logging.info(f"Build idx: {idx}")

        with open(build_index, "wb") as f:
            pickle.dump({"label": label_list, "feature": feature_list}, f)
        logging.info(
            f"Build done. Total {len(label_list)}. Index file has been saved in \"{build_index}\""
        )
