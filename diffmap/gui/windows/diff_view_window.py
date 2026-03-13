
import sys
import os
import json
import collections
import h5py
import runpy
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from pyqtgraph import functions as fn
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
from scipy.ndimage import center_of_mass
from functools import wraps
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog,QErrorMessage,QDialog, QLabel, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt,QObject, QTimer, QThread, pyqtSignal
ui_path = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from diff_fileio import *


#beamline specific
detector_list = ["merlin1","merlin2", "eiger1", "eiger2_image"]
scalars_list = ["None", "sclr1_ch1","sclr1_ch2","sclr1_ch3","sclr1_ch4","sclr1_ch5"]

print("passed module loading")

def remove_nan_inf(im):
    im = np.array(im)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def remove_hot_pixels(image_array, NSigma=3):
    image_array = remove_nan_inf(image_array)
    image_array[abs(image_array) > np.std(image_array) * NSigma] = 0
    return image_array


def show_error_message_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            pass
    return wrapper


class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

class DiffViewWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(DiffViewWindow, self).__init__()
        print("before ui load")
        uic.loadUi(os.path.join(ui_path,'diff_view.ui'), self)

        self.apply_stylesheet(os.path.join(ui_path,"uswds_style.qss"))
        # After loading the UI
        central_widget = self.centralWidget()
        layout = central_widget.layout()

        # Set margins: left, top, right, bottom
        layout.setContentsMargins(20, 10, 20, 10)  # e.g., 20px left/right margins

        print("ui loaded")
        
        self.setup_terminal_redirect()
        #sys.stdout = EmittingStream(text_written=self.normalOutputWritten)
        #sys.stderr = EmittingStream(text_written=self.errorOutputWritten)

        self.prev_config = {} # TODO, record the workflow later
        self.wd = None
        self.xrf_img = None
        self.diff_sum_img = None
        self.single_diff = None
        # self.roi = pg.PolyLineROI( positions=[[-10, -10], [-10, 10], [10, 10], [10, -10]],
        #                         pen="r",
        #                         closed=True,
        #                         removable=True,
        #                         )
        self.roi = None
        self.create_pointer()
        self.points = [] # Record Points
        self.roi_exists = False
        self.display_param = {
                              "diff_wd":None,
                              "xrf_wd":None,
                              "xrf_img_settings":
                                    {"lut":"viridis",
                                     "hist_lim":(None,None),
                                     "remove_hot_pixels":(True,5),
                                     "display_log":False,
                                    },
                                "diff_sum_img_settings":
                                    {"lut":"viridis",
                                     "hist_lim":(None,None),
                                     "remove_hot_pixels":(True,5),
                                     "display_log":False,
                                    },
                                "diff_img_settings":
                                    {"lut":'turbo',
                                     "hist_lim":(None,None),
                                     "remove_hot_pixels":(True,5),
                                     "display_log":False,
                                     }
                            }
        # self.diff_img_view.ui.menuBtn.hide()
        # self.diff_img_view.ui.roiBtn.hide()
        
 

        #beamline specific paramaters
        self.cb_norm_scalars.addItems(scalars_list)
        self.cb_det_list.addItems(detector_list)
        self.cb_det_list.setCurrentIndex(0)
        self.cb_norm_scalars.setCurrentIndex(4)

        #connections
        self.pb_select_wd.clicked.connect(self.choose_wd)
        self.pb_show_mask.clicked.connect(self.get_mask_from_roi)
        self.pb_load_data_from_db.clicked.connect(self.load_from_db)
        self.pb_load_from_h5.clicked.connect(self.load_from_h5)
        self.actionExport_mask_data.triggered.connect(self.save_mask_data)
        self.cb_xrf_elem_list.currentIndexChanged.connect(self.display_xrf_img)
        self.pb_batch_export.clicked.connect(self.do_batch_export)
    
        QtWidgets.qApp.aboutToQuit.connect(self.restore_stdout)

    def restore_stdout(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


    def setup_terminal_redirect(self):
        self.stdout_stream = EmittingStream()
        self.stderr_stream = EmittingStream()

        self.stdout_stream.text_written.connect(self.append_stdout)
        self.stderr_stream.text_written.connect(self.append_stderr)

        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream

    def append_stdout(self, text):
        QtTest.QTest.qWait(100)
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.terminal_output.setTextCursor(cursor)
        self.terminal_output.ensureCursorVisible()

    def append_stderr(self, text):
        QtTest.QTest.qWait(100)
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor("red"))
        cursor.insertText(text, fmt)
        self.terminal_output.setTextCursor(cursor)
        self.terminal_output.ensureCursorVisible()


    
    def apply_stylesheet(self, style_path):
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
    
    def choose_wd(self):
        """updates the line edit for working directory"""
        self.wd = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.le_wd.setText((str(self.wd)))


    def create_load_params(self):

        """    
        param_dict = {wd:'.', 
                 "sid":-1, 
                 "det":"merlin1", 
                 "mon":"sclr1_ch4", 
                 "roi":None, 
                 "mask":None, 
                 "threshold":None}
                 
        """

        self.load_params = {"wd":self.le_wd.text(),
                            "sid":int(self.le_sid.text()), 
                            "threshold":(self.sb_low_threshold.value(),self.sb_high_threshold.value()),
                            "mon":self.cb_norm_scalars.currentText(),
                            "det":self.cb_det_list.currentText(),
                            "roi":None,
                            "mask":None,
                            "save_to_disk":True
                            }
        
        if self.load_params['mon'] == 'None':
            self.load_params['mon'] = None


    def load_and_save_from_db(self):
        
        self.create_load_params()
        real_sid = db[int(self.load_params['sid'])].start["scan_id"]
        self.load_params['sid'] = real_sid
        print(self.load_params)
        print(f"Loading {self.load_params['sid']} please wait...this may take a while...")
        
        
        QtTest.QTest.qWait(1000)
        #saves data to a default folder with sid name      
        self.all_data = export_diff_data_as_h5_single(int(self.load_params['sid']),
                        det=self.load_params['det'],
                        wd= self.load_params['wd'],
                        mon = self.load_params['mon'],
                        save_to_disk= self.load_params.get('save_to_disk', True)
                        )   
        self.load_from_local_and_display() #looks for the filename matching with sid
        #TODO add assertions and exceptions, thread it


    def load_from_db(self):
        
        self.create_load_params()
        real_sid = db[int(self.load_params['sid'])].start["scan_id"]
        self.load_params['sid'] = real_sid
        print(self.load_params)
        print(f"Loading {self.load_params['sid']} please wait...this may take a while...")
        self.all_data = export_diff_data_as_h5_single(int(self.load_params['sid']),
                               self.load_params['det'],save_and_return=True
                               )
        print(f"{self.all_data.keys() = }")
        
        self.display_diff_sum_img(self.all_data)
        QtTest.QTest.qWait(1000)
        self.cb_xrf_elem_list.blockSignals(True)
        self.cb_xrf_elem_list.clear()
        self.cb_xrf_elem_list.addItems(self.xrf_elem_list)
        self.cb_xrf_elem_list.blockSignals(False)
        self.display_xrf_img()

    def load_from_local_and_display(self):
        #self.create_load_params()
        # self.display_param["diff_wd"] = os.path.join(os.path.join(self.load_params["wd"],f"{self.load_params['sid']}_diff_data"),
        #                                         f"{self.load_params['sid']}_diff_{self.load_params['det']}.tiff")

        self.display_param["diff_wd"] = os.path.join(self.load_params["wd"],
                                                     f"scan_{self.load_params['sid']}_{self.load_params['det']}.h5")
        
        self.display_diff_sum_img()
        QtTest.QTest.qWait(1000)
        self.display_xrf_img()


    def load_from_h5(self):
        """
        Open an HDF5 diffraction file, unpack it, update UI elements,
        and display both the summed diffraction image and the XRF image.
        """
        self.create_load_params()

        file_filter = "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)"
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Diffraction File",
            self.load_params.get("wd", '.'),
            file_filter
        )
        if not filename:
            return

        fpath = Path(filename)

        # --- Resolve detector with your precedence rules ---
        user_det = (self.load_params.get("det") or "").strip()
        user_det_l = user_det.lower()

        # Extract trailing token before extension after an underscore (scan_xxx_<det>.h5)
        # e.g., "scan_279757_merlin1.h5" -> "merlin1"
        fname_token = fpath.stem.split("_")[-1].lower() if "_" in fpath.stem else ""

        # Case-insensitive membership check and canonicalization
        allowed_lower = {d.lower(): d for d in detector_list}
        file_det = allowed_lower.get(fname_token)  # canonical name or None

        if file_det:
            # Filename matches an allowed detector
            if user_det and user_det_l != file_det.lower():
                print(f"Detector '{user_det}' does not match filename; using '{file_det}' from filename.")
            det = file_det
        else:
            # Filename does not encode a known detector -> use user's input
            det = user_det if user_det else ""
            if not det:
                print("No detector found in filename and none provided by user.")
            elif user_det_l not in allowed_lower:
                # Not blocking, just warn (per your rule we still go with user's input)
                print(f"Warning: '{user_det}' is not in the allowed list {detector_list}; proceeding with user's value.")

        try:
            print(f"Loading {fpath.name}; this may take a while")
            self.all_data = unpack_diff_h5(str(fpath), det)
            print(f"{self.all_data.keys() = }")

            if hasattr(self, "le_workdir") and isinstance(self.le_workdir, QtWidgets.QLineEdit):
                self.le_workdir.setText(str(fpath.parent))

            self.display_diff_sum_img(self.all_data)

            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50
            )

            if hasattr(self, "cb_xrf_elem_list") and hasattr(self, "xrf_elem_list"):
                blocker = QtCore.QSignalBlocker(self.cb_xrf_elem_list)
                try:
                    self.cb_xrf_elem_list.clear()
                    items = [str(x) for x in self.xrf_elem_list] if self.xrf_elem_list else []
                    self.cb_xrf_elem_list.addItems(items)
                finally:
                    del blocker

            self.display_xrf_img()

            # remember last dir:
            self.load_params["wd"] = str(fpath.parent)

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to Load HDF5",
                f"An error occurred while loading:\n{fpath}\n\n{exc}"
            )


    def create_pointer(self):
        # Use ScatterPlotItem to draw points
        self.scatterItem = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        self.scatterItem.setZValue(2) # Ensure scatterPlotItem is always at top
            

    def display_diff_sum_img(self, all_data):

        """ add_data is the dictonary with keys
        
            "det_images":      
            "Io":              
            "scan_positions":  
            "scan_params":     
            "xrf_array":       
            "xrf_names":       
            "scalar_array":    
            "scalar_names":    
        """
        
        
        self.Io = all_data.get('Io')
        self.scan_pos = all_data.get('scan_positions')
        self.xrf_stack = all_data.get('xrf_array')
        self.xrf_elem_list = all_data.get('xrf_names')
        self.scan_params = all_data.get('scan_params')
        scan_x, scan_y = self.scan_params['scan']['shape']
        
        data = all_data.get('det_images')  # shape: (scan_y * scan_x, roi_y, self.roi_x)
        num_frames, self.roi_y, self.roi_x = data.shape
        assert scan_y * scan_x == num_frames, "Mismatch between scan shape and data shape"
        # Reshape to (scan_y, scan_x, self.roi_y, self.roi_x)
        self.diff_stack = data.reshape(scan_y, scan_x, self.roi_y, self.roi_x)


        if self.diff_stack.ndim != 4:
            raise ValueError(f"{np.shape(self.diff_stack)}; only works for data shape with (im1,im2,det1,det2) structure")
        # print(np.shape(self.diff_stack))

        self.diff_sum_img = np.nansum(self.diff_stack, axis = (-2,-1))#memory efficient?
        
        if not self.roi is None:
            self.p1_diff_img.removeItem(self.roi)
            self.roi = None
        try:
            self.diff_sum_plot_canvas.clear()
            self.p1_diff_sum.clear()
            self.create_pointer()
            
        except:
            pass
        
        
        im_array = self.diff_sum_img 
        ysize,xsize = np.shape(im_array)
        #self.statusbar.showMessage(f"Image Shape = {np.shape(im_array)}")
        # A plot area (ViewBox + axes) for displaying the image

        self.p1_diff_sum = self.diff_sum_plot_canvas.addPlot(title= "Diff_Sum_Image")
        #self.p1_diff_sum.setAspectLocked(True)
        vb = self.p1_diff_sum.getViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)

        # Instead of setLimits, use setRange to define the visible region
        vb.setRange(
            xRange=(0, xsize),
            yRange=(0, ysize),
            padding=0  # optional: no extra margin
            )   
        
        self.p1_diff_sum.addItem(self.scatterItem)
        self.set_up_diff_img_canvas()

        # Item for displaying image data
        #self.img_item_diff_sum = pg.ImageItem(axisOrder = 'row-major')
        self.img_item_diff_sum = pg.ImageItem()
        if self.display_param["diff_sum_img_settings"]["display_log"]:
            im_array = np.nan_to_num(np.log10(im_array), nan=np.nan, posinf=np.nan, neginf=np.nan)
        self.img_item_diff_sum.setImage(im_array, opacity=1)
        self.p1_diff_sum.addItem(self.img_item_diff_sum)
        
    
        self.hist_diff_sum = pg.HistogramLUTItem()
        color_map_diff_sum = pg.colormap.get(self.display_param["diff_sum_img_settings"]["lut"])
        self.hist_diff_sum.gradient.setColorMap(color_map_diff_sum)
        self.hist_diff_sum.setImageItem(self.img_item_diff_sum)
        if self.display_param["diff_sum_img_settings"]["hist_lim"][0] == None or self.display_param["diff_sum_img_settings"]["hist_lim"][1] == None:
            self.hist_diff_sum.autoHistogramRange = False
        else:
            self.hist_diff_sum.autoHistogramRange = False
            self.hist_diff_sum.setLevels(min=self.display_param["diff_sum_img_settings"]["hist_lim"][0], 
                                max=self.display_param["diff_sum_img_settings"]["hist_lim"][1])
        
        self.hist_diff_sum.setFixedWidth(80)
        self.hist_diff_sum.axis.setStyle(tickFont=QtGui.QFont("Arial", 4))
        self.diff_sum_plot_canvas.addItem(self.hist_diff_sum)
        #self.diff_img_view.hoverEvent = self.imageHoverEvent_diff
        self.img_item_diff_sum.mousePressEvent = self.MouseClickEvent_diff_sum
        self.img_item_diff_sum.hoverEvent = self.imageHoverEvent_diff_sum
        self.roi_exists = False


    def display_xrf_img(self, num=0):


        try:
            self.xrf_plot_canvas.clear()
            self.create_pointer()
        except:
            pass

        z, ysize,xsize = np.shape(self.xrf_stack)
        print(f" XRF Image Shape = {np.shape(self.xrf_stack)}")
        # A plot area (ViewBox + axes) for displaying the image


        self.p1_xrf = self.xrf_plot_canvas.addPlot(title="xrf_Image")
        vb = self.p1_xrf.getViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)

        # Instead of setLimits, use setRange to define the visible region
        vb.setRange(
            xRange=(0, xsize),
            yRange=(0, ysize),
            padding=0  # optional: no extra margin
)
            
        self.p1_xrf.addItem(self.scatterItem)

        # Item for displaying image data
        #self.img_item_xrf = pg.ImageItem(axisOrder = 'row-major')
        self.img_item_xrf = pg.ImageItem()
        if self.display_param["xrf_img_settings"]["display_log"]:
            self.xrf_stack = np.nan_to_num(np.log10(self.xrf_stack), nan=np.nan, posinf=np.nan, neginf=np.nan)
        self.img_item_xrf.setImage(self.xrf_stack[int(num)], opacity=1)
        self.p1_xrf.addItem(self.img_item_xrf)
        
    
        self.hist_xrf = pg.HistogramLUTItem(fillHistogram=False)
        color_map_xrf = pg.colormap.get(self.display_param["xrf_img_settings"]["lut"])
        self.hist_xrf.gradient.setColorMap(color_map_xrf)
        self.hist_xrf.setImageItem(self.img_item_xrf)
        if self.display_param["xrf_img_settings"]["hist_lim"][0] == None or self.display_param["xrf_img_settings"]["hist_lim"][1] == None:
            self.hist_xrf.autoHistogramRange = False
        else:
            self.hist_xrf.autoHistogramRange = False
            self.hist_xrf.setLevels(min=self.display_param["xrf_img_settings"]["hist_lim"][0], 
                                max=self.display_param["xrf_img_settings"]["hist_lim"][1])
        
        self.hist_xrf.axis.setStyle(tickFont=QtGui.QFont("Arial", 8))
        self.hist_xrf.setFixedWidth(80)
        self.xrf_plot_canvas.addItem(self.hist_xrf)
        # self.img_item.hoverEvent = self.imageHoverEvent
        self.img_item_xrf.mousePressEvent = self.MouseClickEvent_xrf
        self.img_item_xrf.hoverEvent = self.imageHoverEvent_xrf
        self.roi_exists = False
        # self.roi_state = None

    def MouseClickEvent_xrf(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.points = []
                pos = self.img_item_xrf.mapToParent(event.pos())
                i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
                self.points.append([i, j])
                self.scatterItem.setData(pos=self.points)
                self.single_diff = self.diff_stack[j,i, :,:]
                self.display_param["diff_img_settings"]["display_log"] = self.cb_diff_log_scale.isChecked()

                if self.display_param["diff_img_settings"]["display_log"]:
                    self.single_diff = np.nan_to_num(np.log10(self.single_diff), nan=np.nan, posinf=np.nan, neginf=np.nan)
                
                self.img_item_diff_img.setImage(self.single_diff, opacity=1)

                if self.roi == None:
                    self.create_roi(self.single_diff)
                    self.p1_diff_img.addItem(self.roi)


            else: event.ignore()
        else: event.ignore()

    def set_up_diff_img_canvas(self):
        try: 
            self.diff_img_canvas.clear()
            self.p1_diff_img.clear()
        except:pass

        self.p1_diff_img = self.diff_img_canvas.addPlot(title= "Diff_Image")
        self.img_item_diff_img = pg.ImageItem()
        self.p1_diff_img.addItem(self.img_item_diff_img)

        vb = self.p1_diff_img.getViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)

        # Instead of setLimits, use setRange to define the visible region
        vb.setRange(
            xRange=(0, self.roi_x),
            yRange=(0, self.roi_y),
            padding=0  # optional: no extra margin
            )   
            
        self.hist_diff_img = pg.HistogramLUTItem()
        color_map_diff_img = pg.colormap.get(self.display_param["diff_img_settings"]["lut"])
        self.hist_diff_img.gradient.setColorMap(color_map_diff_img)
        self.hist_diff_img.setImageItem(self.img_item_diff_img)
        if self.display_param["diff_img_settings"]["hist_lim"][0] == None or self.display_param["diff_img_settings"]["hist_lim"][1] == None:
            self.hist_diff_img.autoHistogramRange = False
        else:
            self.hist_diff_img.autoHistogramRange = False
            self.hist_diff_img.setLevels(min=self.display_param["diff_img_settings"]["hist_lim"][0], 
                                max=self.display_param["diff_img_settings"]["hist_lim"][1])
        self.diff_img_canvas.addItem(self.hist_diff_img)
        


    def MouseClickEvent_diff_sum(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.points = []
                pos = self.img_item_diff_sum.mapToParent(event.pos())
                i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
                self.points.append([i, j])
                self.scatterItem.setData(pos=self.points)
                self.single_diff = self.diff_stack[j,i, :,:]
                self.display_param["diff_img_settings"]["display_log"] = self.cb_diff_log_scale.isChecked()

                if self.display_param["diff_img_settings"]["display_log"]:
                    self.single_diff = np.nan_to_num(np.log10(self.single_diff), nan=np.nan, posinf=np.nan, neginf=np.nan)
                
                self.img_item_diff_img.setImage(self.single_diff, opacity=1)

                if self.roi == None:
                    self.create_roi(self.single_diff)
                    self.p1_diff_img.addItem(self.roi)

            else: event.ignore()
        else: event.ignore()

    def imageHoverEvent_xrf(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item_xrf.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            # Set bounds for clipping

            #TODO does not work the hover event si,np.clip to avoid hovering outside
            val = self.xrf_img[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')

    def imageHoverEvent_diff(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            #TODO does not work the hover event si
            val = self.single_diff[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')

    def imageHoverEvent_diff_sum(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item_diff_sum.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            #TODO does not work the hover event si
            val = self.diff_sum_img[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')
    
    def toggle_hist_scale_diff(self, auto = False):        
        
        if auto:
            self.display_param["diff_img_settings"]["hist_lim"] = (None,None)

        else:
            hist_min, hist_max = self.diff_img_view.getLevels()
            self.display_param["diff_img_settings"]["hist_lim"] = (hist_min, hist_max)
            self.statusbar.showMessage(f"Histogram level set to [{hist_min :.4f}, {hist_max :.4f}]")
    


    def create_roi(self, im_array):
        """
        Create a PolyLineROI centered at the image's center of mass.
        """
        # Compute center of mass
        com_y, com_x = center_of_mass(im_array)  # note: rows (y), cols (x)

        # ROI size relative to image
        sz = np.ceil(im_array.shape[0] * 0.2)  # 20% of height
        roi_w = roi_h = int(sz)

        # Define a square ROI
        self.roi = pg.PolyLineROI(
            [[0, 0], [0, roi_h], [roi_w, roi_h], [roi_w, 0]],
            pos=(int(com_x - roi_w // 2), int(com_y - roi_h // 2)),
            maxBounds=QtCore.QRectF(0, 0, im_array.shape[1], im_array.shape[0]),
            pen=pg.mkPen("r", width=1),
            hoverPen=pg.mkPen("w", width=1),
            handlePen=pg.mkPen("m", width=3),
            closed=True,
            removable=True,
            snapSize=1,
            translateSnap=True,
        )
        self.roi.setZValue(10)
        

    def get_mask_from_roi(self):

        # get the roi region:QPaintPathObject
        roiShape = self.roi.mapToItem(self.img_item_diff_img, self.roi.shape())
        
        grid_shape = np.shape(self.single_diff)

        # get data in the scatter plot
        scatterData = np.meshgrid(np.arange(grid_shape[1]), np.arange(grid_shape[0]))
        scatterData = np.reshape(scatterData,(2, grid_shape[0]*grid_shape[1]))

        #xprint(f"{np.shape(scatterData) = }")

        # generate a binary mask for points inside or outside the roishape
        selected = [roiShape.contains(QtCore.QPointF(pt[0], pt[1])) for pt in scatterData.T]

        #print(f"{np.shape(selected) = }")

        # # reshape the mask to image dimensions
        self.mask2D = np.reshape(selected, (self.single_diff.shape))

        # # get masked image1
        # self.maskedImage = self.mask2D * self.single_diff
        print(f"{np.shape(self.single_diff) = }")
        print(f"{self.mask2D.shape}")

        plot1 = pg.image(self.mask2D)
        plot1.setPredefinedGradient("bipolar")
        # plot2 = pg.image(self.single_diff*self.mask2D)
        # plot2.setPredefinedGradient("bipolar")

        masked_diff_sum, masked_diff_img = self.apply_mask_to_diff_stack(self.diff_stack,self.mask2D)
        plot3 = pg.image(masked_diff_sum)
        plot3.setPredefinedGradient("viridis")
        
        plot4 = pg.image(masked_diff_img)
        plot4.setPredefinedGradient("viridis")

    def apply_mask_to_diff_stack(self,diff_data_4d, mask):

        masked_stack = diff_data_4d*mask[np.newaxis,np.newaxis,:,:]

        self.masked_diff_sum, self.masked_diff_img = np.sum(masked_stack,axis = (-1,-2)), np.sum(masked_stack,axis = (0,1))
        return self.masked_diff_sum,self.masked_diff_img
    
    def save_mask_data(self):
        """Save mask and masked diff sum data as both TIFF and CSV files in a versioned folder"""
        # Select base directory - default to working directory from load_params
        default_dir = self.load_params.get('wd', self.wd) if hasattr(self, 'load_params') else self.wd
        base_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', default_dir)
        if not base_folder:
            return  # User cancelled
        
        # Get scan ID from load_params
        scan_id = self.load_params.get('sid', 'unknown')
        
        # Create versioned folder name
        folder_name = f"roi_results_{scan_id}"
        version = 1
        save_folder = os.path.join(base_folder, folder_name)
        
        # Check if folder exists and increment version if needed
        while os.path.exists(save_folder):
            save_folder = os.path.join(base_folder, f"{folder_name}_v{version}")
            version += 1
        
        # Create the folder
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving results to: {save_folder}")
        
        # Save ROI parameters as JSON
        from datetime import datetime
        roi_params = {}
        if self.roi is not None:
            # Get ROI state
            roi_state = self.roi.saveState()
            # Convert to serializable format
            # Handle roi_handles - points can be in different formats
            try:
                if 'points' in roi_state:
                    points = roi_state['points']
                    if len(points) > 0:
                        # Check if points are tuples or dicts
                        if isinstance(points[0], (tuple, list)):
                            roi_handles = [[float(p[0]), float(p[1])] for p in points]
                        elif isinstance(points[0], dict) and 'pos' in points[0]:
                            roi_handles = [[float(h['pos'][0]), float(h['pos'][1])] for h in points]
                        else:
                            roi_handles = []
                    else:
                        roi_handles = []
                else:
                    roi_handles = []
            except Exception as e:
                print(f"Warning: Could not extract ROI handles: {e}")
                roi_handles = []
            
            roi_params = {
                'timestamp': datetime.now().isoformat(),
                'roi_position': list(self.roi.pos()),
                'roi_size': list(self.roi.size()),
                'roi_angle': self.roi.angle() if hasattr(self.roi, 'angle') else 0,
                'roi_handles': roi_handles,
                'roi_state': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                             for k, v in roi_state.items() if k != 'points'},
                'mask_shape': list(self.mask2D.shape),
                'scan_id': scan_id,
            }
        else:
            roi_params = {
                'timestamp': datetime.now().isoformat(),
                'roi_position': None,
                'roi_size': None,
                'message': 'No ROI was defined',
                'mask_shape': list(self.mask2D.shape) if hasattr(self, 'mask2D') else None,
                'scan_id': scan_id,
            }
        
        with open(os.path.join(save_folder, "roi_parameters.json"), 'w') as f:
            json.dump(roi_params, f, indent=4)
        
        # Save as TIFF files
        tf.imwrite(os.path.join(save_folder, "masked_diff_sum.tiff"), self.masked_diff_img)
        tf.imwrite(os.path.join(save_folder, "mask.tiff"), self.mask2D)
        
        # Save as 2D CSV files (preserving array structure)
        np.savetxt(os.path.join(save_folder, "masked_diff_sum.csv"), self.masked_diff_sum, delimiter=',')
        np.savetxt(os.path.join(save_folder, "mask.csv"), self.mask2D, delimiter=',')
        
        # Save as flattened CSV files with pixel number and intensity columns
        # For masked_diff_sum
        masked_diff_sum_flat = self.masked_diff_sum.flatten()
        pixel_numbers = np.arange(len(masked_diff_sum_flat))
        masked_diff_sum_data = np.column_stack((pixel_numbers, masked_diff_sum_flat))
        np.savetxt(os.path.join(save_folder, "masked_diff_sum_flattened.csv"), 
                   masked_diff_sum_data, 
                   delimiter=',', 
                   header='pixel_number,intensity',
                   comments='')
        
        # For mask
        mask_flat = self.mask2D.flatten()
        pixel_numbers_mask = np.arange(len(mask_flat))
        mask_data = np.column_stack((pixel_numbers_mask, mask_flat))
        np.savetxt(os.path.join(save_folder, "mask_flattened.csv"), 
                   mask_data, 
                   delimiter=',', 
                   header='pixel_number,value',
                   comments='')
        
        print(f"Successfully saved mask and masked diff sum data")
        QMessageBox.information(self, "Save Complete", f"Data saved to:\n{save_folder}")


    def do_batch_export(self):
        """Export a batch of scans to HDF5 based on GUI input."""
        try:
            # Parse scan range from GUI input (e.g., "12345,12346-12349")
            scan_list = parse_scan_range(self.le_batch_export.text())
            if len(scan_list) == 0:
                QMessageBox.warning(self, "Input Error", "No valid scan numbers found.")
                return

            self.create_load_params()

            # Extract parameters from load_params
            det = self.load_params["det"]
            mon = self.load_params["mon"]
            wd = self.load_params["wd"]
            compression = "gzip"
            copy_if_possible = True

            # Call your batch export function
            export_diff_data_as_h5_batch(
                sid_list=scan_list,
                det=det,
                wd=wd,
                mon=mon,
                compression=compression,
                copy_if_possible=copy_if_possible
            )

            QMessageBox.information(
                self, "Export Complete", f"Exported {len(scan_list)} scans to HDF5."
            )

        except Exception as e:
            QMessageBox.critical(self, "Batch Export Failed", str(e))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = DiffViewWindow()
    w.show()
    sys.exit(app.exec_())