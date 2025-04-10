import sys
import cv2
import csv
import time
import psutil
import winsound
import threading
import numpy as np
import tensorflow as tf
from typing import List
from argparse import Namespace
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtGui import QAction, QIcon, QColor, QPalette, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QObject, QEvent
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QGridLayout, QTabWidget, QComboBox, QLabel, QLineEdit, QPushButton, QSizePolicy, 
        QStyleFactory, QToolBar, QFileDialog, QTreeWidget, QTreeWidgetItem, QWidget, 
        QListWidget, QLineEdit, QMessageBox, QComboBox, QScrollArea)

from Python.result_collector import Result
from Python.cfg_utils import merge_cfg
from Python.face_recognition import DetectorFace, RecognizerFace, check_model_file, draw


# Worker thread for processing frames
class FrameProcessor(QThread):
    frame_processed = pyqtSignal(int, QImage)  # Camera index, processed frame

    def __init__(self, camera_index, url):
        super().__init__()
        args = Namespace(
            config='config/infer_cfg_project.yml',
            rtsp=None,
            output_dir='output',
            device='GPU',
            enable_mkldnn=False,
            cpu_threads=1,
            use_gpu=True,
            build_index=None
        )

        # Declare and initialize instance variables.
        self.pipeline_res = Result()
        self.camera_index = camera_index
        self.url = url
        self.running = True
        self.fps = 0
        self._pause = False

        self.ALARM_FREQUENCY = 2000
        self.ALARM_INTERVAL = 2

        self.alarm_active = False
        self.stop_event = threading.Event()
        self.alarm_thread = None

        self.cfg = merge_cfg(args)

        self.with_det_face = self.cfg.get(
            'DET_FACE', False)['enable'] if self.cfg.get('DET_FACE', False) else False
        self.with_rec_face = self.cfg.get(
            'REC_FACE', False)['enable'] if self.cfg.get('REC_FACE', False) else False

        self.with_shoplifting_action = self.cfg.get(
            'SHOPLIFTING_ACTION', False)['enable'] if self.cfg.get('SHOPLIFTING_ACTION',
                                                        False) else False
        self.with_arson_action = self.cfg.get(
            'ARSON_ACTION', False)['enable'] if self.cfg.get('ARSON_ACTION',
                                                        False) else False
            
        if self.with_det_face:
            det_face_cfg = self.cfg['DET_FACE']
            model_dir = det_face_cfg['model_dir']
            det_thresh = det_face_cfg['det_thresh']
            det_config = {"thresh": det_thresh, "target_size": [640, 640]}
            det_predictor_config = {
                "use_gpu": args.use_gpu,
                "enable_mkldnn": args.enable_mkldnn,
                "cpu_threads": args.cpu_threads
            }
            model_file_path, params_file_path = check_model_file(model_dir)
            det_predictor_config["model_file"] = model_file_path
            det_predictor_config["params_file"] = params_file_path
            print("model_file", det_predictor_config["model_file"])
            print("params_file", det_predictor_config["params_file"])
            self.det_face_predictor = DetectorFace(det_config, det_predictor_config)

        if self.with_rec_face:
            rec_face_cfg = self.cfg['REC_FACE']
            model_dir = rec_face_cfg['model_dir']
            batch_size = rec_face_cfg['batch_size']
            index = rec_face_cfg['index']
            cdd_num = rec_face_cfg['cdd_num']
            rec_thresh = rec_face_cfg['rec_thresh']
            rec_config = {
                "max_batch_size": batch_size,
                "resize": 112,
                "thresh": rec_thresh,
                "index": index,
                "build_index": args.build_index,
                "cdd_num": cdd_num
            }
            rec_predictor_config = {
                "use_gpu": args.use_gpu,
                "enable_mkldnn": args.enable_mkldnn,
                "cpu_threads": args.cpu_threads
            }
            model_file_path, params_file_path = check_model_file(model_dir)
            rec_predictor_config["model_file"] = model_file_path
            rec_predictor_config["params_file"] = params_file_path
            self.rec_face_predictor = RecognizerFace(rec_config, rec_predictor_config)

        if self.with_shoplifting_action:
            self.shoplifting_action_predictor = tf.lite.Interpreter(model_path="inference_model/Shoplifting/best_model.tflite")
            self.shoplifting_action_predictor.allocate_tensors()
            self.shoplifting_input_details = self.shoplifting_action_predictor.get_input_details()
            self.shoplifting_output_details = self.shoplifting_action_predictor.get_output_details()

        if self.with_arson_action:
            self.arson_action_predictor = tf.lite.Interpreter(model_path="inference_model/Arson/best_model.tflite")
            self.arson_action_predictor.allocate_tensors()
            self.arson_input_details = self.arson_action_predictor.get_input_details()
            self.arson_output_details = self.arson_action_predictor.get_output_details()
        
    def run(self):
        # Capture video from a network stream.
        self.cap = cv2.VideoCapture(self.url)
        # Get default video FPS.
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(self.fps)

        shoplifting_action_imgs = []
        arson_action_imgs = []
        

        if self.cap.isOpened():
            while self.running:
                if not self._pause:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame_rgb.shape
                        bytes_per_line = ch * w

                        if self.with_det_face or self.with_rec_face:
                            input_img = frame_rgb.astype(np.float32, copy=False)
                            det_face_res = None
                            if self.with_det_face:
                                det_face_res = self.det_face_predictor.predict(input_img)
                                self.pipeline_res.update(det_face_res, 'det_face')
                            if self.with_rec_face:
                                rec_face_res = self.rec_face_predictor.predict(input_img, det_face_res)
                                self.pipeline_res.update(rec_face_res, 'rec_face')

                        if self.with_shoplifting_action:
                            # get the params
                            sequence_length = self.cfg["SHOPLIFTING_ACTION"]["sequence_length"]
                            CLASSES_LIST = ['Shoplifting', 'Normal'] 

                            resized_frame = cv2.resize(frame_rgb, (80, 80)) 
                            shoplifting_action_imgs.append(resized_frame)
                    
                        
                            if len(shoplifting_action_imgs) == sequence_length:
                                frames_array = np.array(shoplifting_action_imgs) / 255.0 
                                frames_array = np.expand_dims(frames_array, axis=0).astype(np.float32) 

                                self.shoplifting_action_predictor.set_tensor(self.shoplifting_input_details[0]['index'], frames_array)
                                self.shoplifting_action_predictor.invoke()
                    
                                predictions = self.shoplifting_action_predictor.get_tensor(self.shoplifting_output_details[0]['index'])[0]
                                scores = predictions * 100

                                if scores[0] > 80:
                                    shoplifting_action_res = {"class": CLASSES_LIST[0], "score": scores[0]}
                                    self.pipeline_res.update(shoplifting_action_res, 'shoplifting_action')
                                else:
                                    self.pipeline_res.clear('shoplifting_action')
                                    self.manage_alarm(False) 
                                    cv2.destroyAllWindows()
                    
                                shoplifting_action_imgs.pop(0)

                        if self.with_arson_action:
                            # get the params
                            sequence_length = self.cfg["ARSON_ACTION"]["sequence_length"]
                            CLASSES_LIST = ['Arson', 'Normal'] 

                            resized_frame = cv2.resize(frame_rgb, (80, 80)) 
                            arson_action_imgs.append(resized_frame)
                    
                        
                            if len(arson_action_imgs) == sequence_length:
                                frames_array = np.array(arson_action_imgs) / 255.0 
                                frames_array = np.expand_dims(frames_array, axis=0).astype(np.float32)  

                                self.arson_action_predictor.set_tensor(self.arson_input_details[0]['index'], frames_array)
                                self.arson_action_predictor.invoke()
                    
                                predictions = self.arson_action_predictor.get_tensor(self.arson_output_details[0]['index'])[0]
                                scores = predictions * 100

                                if scores[0] > 90:
                                    arson_action_res = {"class": CLASSES_LIST[0], "score": scores[0]}
                                    self.pipeline_res.update(arson_action_res, 'arson_action')
                                else:
                                    self.pipeline_res.clear('arson_action')
                                    self.manage_alarm(False)
                                    cv2.destroyAllWindows()
                    
                                arson_action_imgs.pop(0)
                        
                        self.im = self.visualize_video(frame_rgb, self.pipeline_res)  # visualize
                        self.rgb_image = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
                        self.qt_image = QImage(self.rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.qt_image_sc = self.qt_image.scaled(1280, 720, Qt.AspectRatioMode.KeepAspectRatio)  # 720p
            
                        # Emit the processed frame and results
                        self.frame_processed.emit(self.camera_index, self.qt_image_sc)
                    
                    else:
                        self.pipeline_res.clear('arson_action')
                        self.manage_alarm(False)
                        break
        else:
            self.manage_alarm(False)
            cv2.destroyAllWindows()
            # When everything done, release the video capture object.
            self.cap.release()
            # Tells the thread's event loop to exit with return code 0 (success).
            self.quit()

    def alarm_worker(self):
        while not self.stop_event.is_set():
            winsound.Beep(self.ALARM_FREQUENCY, 200)
            time.sleep(self.ALARM_INTERVAL)

    def manage_alarm(self, state):
        # global self.alarm_active, self.alarm_thread, self.stop_event
        
        if state and not self.alarm_active:
            self.stop_event.clear()
            self.alarm_thread = threading.Thread(target=self.alarm_worker)
            self.alarm_thread.start()
            self.alarm_active = True
        elif not state and self.alarm_active:
            self.stop_event.set()
            self.alarm_thread.join()
            self.alarm_active = False     

    def visualize_video(self, image_rgb, result):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)            
        box_list = result.get('det_face')
        np_feature = result.get('rec_face')
        if np_feature is not None:
            labels = self.rec_face_predictor.retrieval(np_feature)
        if box_list is not None:
            if np_feature is None:
                labels = ["face"] * len(box_list)
            image = draw(image, box_list, labels=labels)

        shoplifting_action_res = result.get('shoplifting_action')
        if shoplifting_action_res is not None:
            class_name_shoplifting = shoplifting_action_res["class"]
            probability_shoplifting = shoplifting_action_res["score"]

            text_shoplifting = f'{class_name_shoplifting}:{probability_shoplifting:.2f}%'
            text_scale = 1
            text_thickness = 2
            (tw, th), _ = cv2.getTextSize(text_shoplifting, cv2.FONT_ITALIC, text_scale, text_thickness)

            color_shoplifting = (0, 0, 255) if probability_shoplifting > 90 else (0, 255, 0)
            color_rectangle = (0, 0, 0)


            # Draw the outer frame rectangle
            cv2.rectangle(image, 
                          (2, 2), # Left-top corner
                          (2 + tw, 10 + th + 8), # Right-bottom corner
                          color_rectangle, -1)
            
            cv2.putText(image, 
                        text_shoplifting,
                        (4, 7 + th), # The position (x, y) of the left-bottom corner of the text
                        cv2.FONT_ITALIC,
                        text_scale, color_shoplifting,
                        text_thickness)
            
            if probability_shoplifting > 80:
                self.manage_alarm(probability_shoplifting > 80)
            else:
                self.manage_alarm(False)
                cv2.destroyAllWindows()
            
        arson_action_res = result.get('arson_action')
        if arson_action_res is not None:
            class_name_arson = arson_action_res["class"]
            probability_arson = arson_action_res["score"]

            text_arson = f'{class_name_arson}:{probability_arson:.2f}%'
            text_scale = 1
            text_thickness = 2
            (tw, th), _ = cv2.getTextSize(text_arson, cv2.FONT_ITALIC, text_scale, text_thickness)

            color_arson = (0, 0, 255) if probability_arson > 85 else (0, 255, 0) 
            color_rectangle = (0, 0, 0)

            # Draw the outer frame rectangle
            cv2.rectangle(image, 
                          (2, (10 + th + 8) + 2), # Left-top corner
                          (2 + tw, ((10 + th + 8) * 2)), # Right-bottom corner
                          color_rectangle, -1)

            cv2.putText(image, 
                        text_arson,
                        (4, ((7 + th) * 2) + 12), # The position (x, y) of the left-bottom corner of the text
                        cv2.FONT_ITALIC,
                        text_scale, color_arson,
                        text_thickness)
            if probability_arson > 90:
                self.manage_alarm(probability_arson > 90)
            else:
                self.manage_alarm(False)
                cv2.destroyAllWindows()

        return image

    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self) -> None:
        self._pause = True

    def unpause(self) -> None:
        self._pause = False





def save_cameras_to_file(video_sources: List[str], filename: str = "cameras.csv") -> None:
    """Save video sources to a CSV file."""
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Video Sources"])  # Header row
            for source in video_sources:
                writer.writerow([source])  # Write each source as a separate row
        print(f"Successfully saved video sources to {filename}.")
    except Exception as e:
        print(f"An error occurred while saving video sources: {e}")

def load_cameras_from_file(filename: str = "cameras.csv") -> List[str]:
    """Load video sources from a CSV file."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            # next(reader)  # Skip the header row
            return [row[0] for row in reader]  # Extract the first column
    except FileNotFoundError:
        print(f"File not found: {filename}. Returning an empty list.")
        return []
    except Exception as e:
        print(f"An error occurred while loading video sources: {e}")
        return []

class Color(QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cameras_file = "cameras.csv"
        self.setWindowTitle("Smart Surveillance System")
        self.setMinimumSize(715, 480)
        self.showMaximized()
        styleName = "Fusion"
        QApplication.setStyle(QStyleFactory.create(styleName))
        QApplication.setPalette(QApplication.style().standardPalette())

        self.webcams = []
        for i in range(10):  # Check the first 10 indices
            cap_1 = cv2.VideoCapture(i)
            if cap_1.isOpened():
                self.webcams.append(f"Webcam {i}")
                cap_1.release()

        self.view_page = QWidget()        
        self.grid_layout = QGridLayout()
        
        # Camera Feeds
        self.camera_sources = self.load_cameras()
        self.camera_labels = []  # List of QLabels to display camera feeds
        self.QScrollAreas = []
        self.recording = False
        self.video_writers = []  # List of VideoWriter objects for recording
        self.ai_threads = []  # List of AI processing threads
        self.processors = []
        # Dictionary to keep the state of a camera. The camera state will be: Normal or Maximized.
        self.list_of_cameras_state = {}

        self._start_cameras()

        self._toolBar()
        self._layout()

    def _start_cameras(self):
        for i, url in enumerate(self.camera_sources):
            camera_label = QLabel()
            ScrollArea = QScrollArea()
            if i not in range(len(self.camera_labels)):
                self.camera_labels.insert(i, camera_label)
                self.camera_labels[i].setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                self.camera_labels[i].setScaledContents(True)
                self.camera_labels[i].installEventFilter(self)
                self.camera_labels[i].setObjectName(f"Camera_{i}")
                self.list_of_cameras_state[f"Camera_{i}"] = "Normal"

                self.QScrollAreas.insert(i, ScrollArea)
                self.QScrollAreas[i].setAutoFillBackground(True)
                self.QScrollAreas[i].setWidgetResizable(True)
                self.QScrollAreas[i].setWidget(self.camera_labels[i])

            if i not in range(len(self.processors)):
                processor = FrameProcessor(i, url)
                processor.frame_processed.connect(lambda camera_index, frame_processed: self.update_frame(camera_index, frame_processed))
                self.processors.insert(i, processor)
                processor.start()
        
    
    def _toolBar(self):
        # Toolbar
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(self.toolbar)
        
        self.add_tab_action = QAction(QIcon("env/icons/search.png"), "View", self)
        self.add_tab_action.triggered.connect(self.add_new_tab)  # Connect to method
        self.toolbar.addAction(self.add_tab_action)

        self.toolbar.addSeparator()

        # Create a QWidget to hold the QLabel and align it to the right
        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_widget.setLayout(right_layout)

        # Add a stretch to push the label to the right
        right_layout.addStretch()

        # Resource Label
        self.resource_label = QLabel("CPU: 0% | RAM: 0%")
        self.resource_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.resource_label)

        # Add the right-aligned widget to the toolbar
        self.toolbar.addWidget(right_widget)

    def _layout(self):
        # Layout
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QHBoxLayout(self.widget)

        # Main Layout (Camera Scenes)
        self.main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # self.tabs.setMovable(True)

        self.main_layout.addWidget(self.tabs)

        self._leftLayout()
        # self._mainLayout()

        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.main_layout)

        self.add_new_tab()

    def _leftLayout(self):
        self.left_layout = QVBoxLayout()
        self.settings_label = QLabel("Settings")
        self.settings_label.setMinimumWidth(150)
        self.settings_label.setMaximumWidth(150)
        self.left_layout.addWidget(self.settings_label)

        searchbar = QLineEdit()
        searchbar.setMinimumWidth(150)
        searchbar.setMaximumWidth(150)
        self.left_layout.addWidget(searchbar)

        # Create a QTreeWidget for the left navigation bar
        self.nav_tree = QTreeWidget()
        self.nav_tree.setMinimumWidth(150)
        self.nav_tree.setMaximumWidth(150)
        # self.nav_tree.setFixedWidth(200)  # Set a fixed width for the navigation bar
        self.nav_tree.setHeaderHidden(True)  # Hide the header
        self.left_layout.addWidget(self.nav_tree)


        # Add items to navigation tree
        view_item = QTreeWidgetItem(self.nav_tree)
        view_item.setText(0, "View")
        view_item.setIcon(0, QIcon("env/icons/search.png"))
        # view_item.setData(0, Qt.ItemDataRole.UserRole)

        playback_item = QTreeWidgetItem(self.nav_tree)
        playback_item.setText(0, "Playback")
        playback_item.setIcon(0, QIcon("env/icons/search.png"))
        # playback_item.setData(0, Qt.ItemDataRole.UserRole)

        search_item = QTreeWidgetItem(self.nav_tree)
        search_item.setText(0, "Search")
        search_item.setIcon(0, QIcon("env/icons/search.png"))
        # search_item.setData(0, Qt.ItemDataRole.UserRole)

        settings_item = QTreeWidgetItem(self.nav_tree)
        settings_item.setText(0, "Settings")
        settings_item.setIcon(0, QIcon("env/icons/search.png"))

        cameras_item = QTreeWidgetItem(settings_item)
        cameras_item.setText(0, "Cameras")
        cameras_item.setIcon(0, QIcon("env/icons/search.png"))

        alert_item = QTreeWidgetItem(settings_item)
        alert_item.setText(0, "Alert Center")
        alert_item.setIcon(0, QIcon("env/icons/search.png"))

        emap_item = QTreeWidgetItem(settings_item)
        emap_item.setText(0, "Emap")
        emap_item.setIcon(0, QIcon("env/icons/search.png"))

        
        # Connect navigation tree item selection to stacked widget
        self.nav_tree.itemClicked.connect(self._on_nav_item_clicked)

    def _on_nav_item_clicked(self, item, column):
        # widget = item.data(0, Qt.ItemDataRole.UserRole)
        title = item.text(0)  # Get the text from the first column

        if title == "View":
            self.add_new_tab()
        elif title == "Playback":
            self._create_playback_page()
        elif title == "Search":
            self._create_search_page()
        elif title == "Cameras":
            self._create_cameras_page()
        elif title == "Alert Center":
            self._create_alert_center_page()
        elif title == "Emap":
            self._create_emap_page()

    def add_new_tab(self):
        """Adds a new tab to the QTabWidget."""
        new_label = False
        row, col = 0, 0
        for i, scroll_area in enumerate(self.QScrollAreas):
            label_found = False
            for j in range(self.grid_layout.count()):
                item = self.grid_layout.itemAt(j)
                if item and item.widget() == scroll_area:
                    label_found = True
                    break
            if not label_found:
                self.grid_layout.addWidget(scroll_area, row, col)
                new_label = True
            col += 1
            if col >= 3:
                col = 0
                row += 1

        if new_label:
            self.view_page.setLayout(self.grid_layout)

        tab_index = self.tabs.count()

        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Main View":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(self.view_page, "Main View")
            self.tabs.setCurrentIndex(tab_index)
        elif widget_index != -1 and new_label:
            self.tabs.addTab(self.view_page, "Main View")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)

        # Add the new tab to the QTabWidget
        # if tab_index == 0:
            # self.tabs.addTab(view_page, f"Main View")

        # else:
            # self.tabs.addTab(view_page, f"View {self.tabs.count() + 1}")

        # Activating a Tab
        # self.tabs.setCurrentIndex(tab_index)

    def _create_playback_page(self):
        # Create playback settings page
        playback_page = QWidget()
        playback_layout = QVBoxLayout(playback_page)
        playback_layout.addWidget(QLabel("Customizable Video Playback Settings Content"))

        tab_index = self.tabs.count()

        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Playback":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(playback_page, "Playback")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)
        # return playback_page

    def _create_search_page(self):
        # Create search settings page
        search_page = QWidget()
        search_layout = QVBoxLayout(search_page)
        search_layout.addWidget(QLabel("Search Settings Content"))

        tab_index = self.tabs.count()

        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Search":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(search_page, "Search")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)
    

    def _create_cameras_page(self):
        # Create settings page
        cameras_page = QWidget()
        cameras_layout = QHBoxLayout(cameras_page)

        self.add_camera_layout = QVBoxLayout()
        self.add_camera_layout.setContentsMargins(0,0,0,0)
        # self.add_camera_layout.setSpacing(0)


        # self.add_camera_layout.addWidget(QLabel("Add Cameras:"))

        add_ip_camera_layout = QVBoxLayout()
        add_ip_camera_layout.addWidget(QLabel("IP Camera: "))

        add_ip_camera_layout_h_1 = QHBoxLayout()
        add_ip_camera_layout_h_2 = QHBoxLayout()
        add_ip_camera_layout_h_3 = QHBoxLayout()
        add_ip_camera_layout_h_4 = QHBoxLayout()
        add_ip_camera_layout_h_5 = QHBoxLayout()
        add_ip_camera_layout_h_6 = QHBoxLayout()

        # Add IP camera section
        self.camera_name_input = QLineEdit()
        self.camera_name_input.setPlaceholderText("Enter Camera Name (e.g., camera 1)")
        self.camera_username_input = QLineEdit()
        self.camera_username_input.setPlaceholderText("Enter Username (e.g., admin)")
        self.camera_password_input = QLineEdit()
        self.camera_password_input.setPlaceholderText("Enter Password  (e.g., admin)")
        self.camera_ip_input = QLineEdit()
        self.camera_ip_input.setPlaceholderText("Enter IP Address (e.g., 192.168.0.1)")
        self.camera_port_input = QLineEdit()
        self.camera_port_input.setPlaceholderText("Enter Port (e.g., 9080)")
        
        self.camera_url_input = QLineEdit()
        self.camera_url_input.setPlaceholderText("Enter IP Camera URL (e.g., rtsp://username:password@ip:port)")

        self.name_label = QLabel("Camera Name")
        self.username_label = QLabel("User Name")
        self.password_label = QLabel("Password")
        self.ip_label = QLabel("IP Address")
        self.port_label = QLabel("Port")
        
        self.url_label = QLabel("Full Camera URL")

        add_ip_camera_layout_h_1.addWidget(self.name_label)
        add_ip_camera_layout_h_1.addWidget(self.camera_name_input)

        add_ip_camera_layout_h_2.addWidget(self.username_label)
        add_ip_camera_layout_h_2.addWidget(self.camera_username_input)

        add_ip_camera_layout_h_3.addWidget(self.password_label)
        add_ip_camera_layout_h_3.addWidget(self.camera_password_input)

        add_ip_camera_layout_h_4.addWidget(self.ip_label)
        add_ip_camera_layout_h_4.addWidget(self.camera_ip_input)

        add_ip_camera_layout_h_5.addWidget(self.port_label)
        add_ip_camera_layout_h_5.addWidget(self.camera_port_input)

        add_ip_camera_layout_h_6.addWidget(self.url_label)
        add_ip_camera_layout_h_6.addWidget(self.camera_url_input)

        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_1)
        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_2)
        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_3)
        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_4)
        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_5)
        
        add_ip_camera_layout.addLayout(add_ip_camera_layout_h_6)

        add_ip_camera_button = QPushButton("Add IP Camera")
        add_ip_camera_button.clicked.connect(self.add_ip_camera)
        add_ip_camera_layout.addWidget(add_ip_camera_button)


        # styleLabel = QLabel("&Style:")
        # styleLabel.setBuddy(styleComboBox)

        # Add webcam section
        add_webcam_layout = QVBoxLayout()
        add_webcam_layout.addWidget(QLabel("Webcam: "))
        add_webcam_layout_h = QHBoxLayout()
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItems(self.webcams)
        self.webcam_label = QLabel("Select Webcam")
        self.webcam_label.setBuddy(self.webcam_combo)
        add_webcam_layout_h.addWidget(self.webcam_label)
        add_webcam_layout_h.addWidget(self.webcam_combo)
        add_webcam_layout.addLayout(add_webcam_layout_h)
        add_webcam_button = QPushButton("Add Webcam")
        add_webcam_button.clicked.connect(self.add_webcam)
        add_webcam_layout.addWidget(add_webcam_button)




        self.add_camera_layout.addLayout(add_ip_camera_layout)
        self.add_camera_layout.addLayout(add_webcam_layout)


        cameras_layout.addLayout(self.add_camera_layout)

        # List of cameras
        self.camera_list_layout = QVBoxLayout()
        self.camera_list = QListWidget()
        # for cam in self.camera_sources:
            # self.camera_list.addItems(str(cam))
        self.camera_list_layout.addWidget(QLabel("Added Cameras:"))
        self.camera_list_layout.addWidget(self.camera_list)  # Use addWidget instead of addLayout

        # Remove camera button
        remove_camera_button = QPushButton("Remove Selected Camera")
        remove_camera_button.clicked.connect(self.remove_camera)
        self.camera_list_layout.addWidget(remove_camera_button)

        cameras_layout.addLayout(self.camera_list_layout)

        # Add the cameras page to the tabs
        tab_index = self.tabs.count()
        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Cameras":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(cameras_page, "Cameras")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)

    def _create_alert_center_page(self):
        # Create alert center page
        alert_center_page = QWidget()
        alert_center_layout = QVBoxLayout(alert_center_page)
        alert_center_layout.addWidget(QLabel("Alert Center Content"))

        tab_index = self.tabs.count()

        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Alert Center":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(alert_center_page, "Alert Center")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)
        return alert_center_page

    def _create_emap_page(self):
        # Create EMAP page
        emap_page = QWidget()
        emap_layout = QVBoxLayout(emap_page)
        emap_layout.addWidget(QLabel("EMAP Content"))

        tab_index = self.tabs.count()

        # Check if the tab already exists
        widget_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Emap":
                widget_index = i
                break

        if widget_index == -1:
            self.tabs.addTab(emap_page, "Emap")
            self.tabs.setCurrentIndex(tab_index)
        else:
            self.tabs.setCurrentIndex(widget_index)
        # return emap_page

    
    @QtCore.pyqtSlot()
    def update_frame(self, camera_index: int, frame_processed: QImage):
        # Display the processed frame

        self.line = len(self.camera_labels) / 2

        self.widget_width = self.widget.width()
        self.widget_height = self.widget.height()

        self.camera_width = self.widget_width - 220
        self.camera_height = self.widget_height - 80

        self.video_width = self.camera_width // 3
        self.video_height = self.camera_height // 3

        self.camera_labels[camera_index].setPixmap(QtGui.QPixmap.fromImage(frame_processed))
        
        self.QScrollAreas[camera_index].setMinimumSize(self.video_width, self.video_height)

        # Update CPU and RAM usage
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        self.resource_label.setText(f"CPU: {cpu_usage}% | RAM: {ram_usage}%")
    
    def add_ip_camera(self):
        """Add a new IP camera to the list."""
        camera_url = self.camera_url_input.text().strip()
        if camera_url:
            if camera_url not in [self.camera_list.item(i).text() for i in range(self.camera_list.count())]:
                self.camera_list.addItem(f"IP Camera: {camera_url}")
                self.camera_sources.append(camera_url)
                self._start_cameras()
                self.camera_url_input.clear()
            else:
                QMessageBox.warning(self, "Duplicate Camera", "This IP camera URL already exists.")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid IP camera URL.")

    def add_webcam(self):
        """Add a new webcam to the list."""
        webcam_index = self.webcam_combo.currentText().split()[-1]  # Extract index from "Webcam X"
        webcam_name = f"Webcam: {webcam_index}"
        if webcam_name not in [self.camera_list.item(i).text() for i in range(self.camera_list.count())]:
            self.camera_list.addItem(webcam_name)
            self.camera_sources.append(webcam_index)
            self._start_cameras()
        else:
            QMessageBox.warning(self, "Duplicate Camera", "This webcam is already added.")


    def add_camera(self):
        # Add a new camera feed
        if len(self.camera_sources) < 30:  # Limit to 30 cameras for this example
            camera_url, _ = QFileDialog.getOpenFileName(self, "Select Camera URL or File")
            if camera_url:
                camera = cv2.VideoCapture(camera_url)
                if camera.isOpened():
                    self.camera_sources.append(camera_url)
                    self._start_cameras()

    def remove_camera(self):
        """Remove the selected camera from the list."""
        selected_item = self.camera_list.currentItem()
        if selected_item:
            self.camera_list.takeItem(self.camera_list.row(selected_item))
            self.camera_sources.pop(selected_item)
            label = self.camera_labels.pop(selected_item)
            label.deleteLater()
            self._start_cameras()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a camera to remove.")

    def load_cameras(self):
        """Load cameras from the JSON file."""
        cameras = load_cameras_from_file(self.cameras_file)
        return cameras

    def save_cameras(self):
        """Save cameras to the JSON file."""
        # cameras = [self.camera_list.item(i).text() for i in range(self.camera_list.count())]
        cameras = [source for source in self.camera_sources]
        save_cameras_to_file(cameras, self.cameras_file)

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        # self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    # Override method for class MainWindow.
    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """
        Method to capture the events for objects with an event filter installed.
        :param source: The object for whom an event took place.
        :param event: The event that took place.
        :return: True if event is handled.
        """
        #
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            camera_index = source.objectName()
            if camera_index in self.list_of_cameras_state:
                #
                if self.list_of_cameras_state[camera_index] == "Normal":
                    for index in self.list_of_cameras_state:
                        if index != camera_index:
                            self.QScrollAreas[int(index.split("_")[1])].hide()
                    self.list_of_cameras_state[camera_index] = "Maximized"
                else:
                    for index in self.list_of_cameras_state:
                        if index != camera_index:
                            self.QScrollAreas[int(index.split("_")[1])].show()
                    self.list_of_cameras_state[camera_index] = "Normal"
            else:
                return super(MainWindow, self).eventFilter(source, event)
            return True
        else:
            return super(MainWindow, self).eventFilter(source, event)

    def closeEvent(self, event):
        """
        Override the closeEvent method to clean up resources before closing the application.
        """
        # Stop all camera feeds and release resources
        # for capture in self.captures:
        #     if capture.isOpened():
        #         capture.release()

        # Stop all processing threads
        for processor in self.processors:
            processor.stop()  # Ensure the thread stops
            processor.wait()  # Wait for the thread to finish

        # Release all video writers
        for writer in self.video_writers:
            if writer.isOpened():
                writer.release()

        """Save cameras when the window is closed."""
        self.save_cameras()
        event.accept() # Close the application


def exit_application():
    """Exit program event handler"""

    sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    
    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), window, exit_application)
    
    if(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec()