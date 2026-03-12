
import sys
import os
import json
import collections
import h5py
import runpy
import numpy as np
import pyqtgraph as pg
import tifffile as tf
import warnings

from pyqtgraph import functions as fn
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
from scipy.ndimage import center_of_mass
from functools import wraps
from PyQt6 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt6.QtWidgets import QMessageBox, QFileDialog,QErrorMessage,QDialog, QLabel, QVBoxLayout, QProgressBar
from PyQt6.QtCore import Qt,QObject, QTimer, QThread, pyqtSignal


warnings.filterwarnings('ignore', category=RuntimeWarning)
from utils.diff_fileio import *
from diffmap.gui import UI_DIR
STYLE_PATH = UI_DIR / 'css' / 'uswds.qss'

#sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
# from hxntools.CompositeBroker import db
# from hxntools.scan_info import get_scan_positions

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

# class EmittingStream(QObject):

#     textWritten = pyqtSignal(str)

#     def write(self, text):
#         self.textWritten.emit(str(text))

class DiffMapWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(DiffMapWindow, self).__init__()
        print("before ui load")
        uic.loadUi(os.path.join(UI_DIR,'diff_view.ui'), self)

        #self.apply_stylesheet(os.path.join(ui_path,"uswds_style.qss"))
        # After loading the UI
        central_widget = self.centralWidget()
        layout = central_widget.layout()

        # Set margins: left, top, right, bottom
        layout.setContentsMargins(20, 10, 20, 10)  # e.g., 20px left/right margins

        print("ui loaded")
        
        # sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        # sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

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
        self.actionExport_mask_data.triggered.connect(self.save_mask_data)
        self.cb_xrf_elem_list.currentIndexChanged.connect(self.display_xrf_img)

    


    '''
    def __del__(self):
        import sys
        # Restore sys.stdout
        sys.stdout = sys.__stdout__


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def errorOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()

    '''
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

    def choose_diff_file(self):

        """updates the line edit for working directory"""

        filename_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select DiffFile')
        if filename_[0]:
            self.diff_file = filename_[0]
            self.display_param["diff_wd"] = self.diff_file
            print(f"Loading {filename_[0]} please wait...\n this may take a while...")
            self.display_diff_sum_img()
            QtTest.QTest.qWait(1000)
            self.display_xrf_img()
            print("Done")
        else:
            pass


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
        scan_y, scan_x = self.scan_params['scan']['shape']
        
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
        """updates the line edit for working directory"""
        self.save_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', self.wd)
        tf.imwrite(os.path.join(self.save_folder,"_masked_diff_sum.tiff"),  self.masked_diff_img)
        tf.imwrite(os.path.join(self.save_folder,"_mask.tiff"),  self.mask2D)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = DiffMapWindow()
    w.show()
    sys.exit(app.exec())