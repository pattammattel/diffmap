
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
from functools import wraps
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog,QErrorMessage,QDialog, QLabel, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt,QObject, QTimer, QThread, pyqtSignal
ui_path = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from nanorsm_v2 import *

sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

#beamline specific
detector_list = ["merlin1","merlin2", "eiger1", "eiger2_image"]
scalars_list = ["None", "sclr1_ch1","sclr1_ch2","sclr1_ch3","sclr1_ch4","sclr1_ch5"]

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

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class DiffViewWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(DiffViewWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'diff_view.ui'), self)
        
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

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
                                    {"lut":'bipolar',
                                     "hist_lim":(None,None),
                                     "remove_hot_pixels":(True,5),
                                     "display_log":False,
                                     }
                            }
        
        self.diff_img_view.ui.menuBtn.hide()
        self.diff_img_view.ui.roiBtn.hide()

        #beamline specific paramaters
        self.cb_norm_scalars.addItems(scalars_list)
        self.cb_det_list.addItems(detector_list)
        self.cb_det_list.setCurrentIndex(1)
        self.cb_norm_scalars.setCurrentIndex(4)

        #connections
        self.pb_select_wd.clicked.connect(self.choose_wd)
        self.pb_load_xrf.clicked.connect(self.choose_xrf_file)
        self.pb_load_diff.clicked.connect(self.choose_diff_file)
        self.pb_set_hist_levels_diff.clicked.connect(lambda:self.toggle_hist_scale_diff(auto = False))
        self.pb_auto_hist_levels_diff.clicked.connect(lambda:self.toggle_hist_scale_diff(auto = True))
        self.pb_show_mask.clicked.connect(self.get_mask_from_roi)
        self.pb_load_data_from_db.clicked.connect(self.load_and_save_from_db)
        #self.pb_load_data_from_db.clicked.connect(self.load_from_db)
        self.pb_swap_diff_axes.clicked.connect(lambda:self.diff_stack.transpose(0,1,3,2))
        self.actionExport_mask_data.triggered.connect(self.save_mask_data)



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
                            "mask":None
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
        export_single_diff_data(self.load_params) #saves data to a default folder with sid name        
        self.load_from_local_and_display() #looks for the filename matching with sid
        #TODO add assertions and exceptions, thread it


    def load_from_db(self):
        
        self.create_load_params()
        real_sid = db[int(self.load_params['sid'])].start["scan_id"]
        self.load_params['sid'] = real_sid
        print(self.load_params)
        print(f"Loading {self.load_params['sid']} please wait...this may take a while...")
        diff_array = return_diff_array(int(self.load_params['sid']), 
                                     det=self.load_params['det'], 
                                     mon=self.load_params['mon'], 
                                     threshold=self.load_params['threshold'])
        
        self.display_diff_sum_img()
        QtTest.QTest.qWait(1000)

    def load_from_local_and_display(self):
        #self.create_load_params()
        self.display_param["diff_wd"] = os.path.join(os.path.join(self.load_params["wd"],f"{self.load_params['sid']}_diff_data"),
                                                f"{self.load_params['sid']}_diff_{self.load_params['det']}.tiff")
        
        self.display_diff_sum_img()

    def choose_diff_file(self):

        """updates the line edit for working directory"""

        filename_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select DiffFile')
        if filename_[0]:
            self.diff_file = filename_[0]
            self.display_param["diff_wd"] = self.diff_file
            print(f"Loading {filename_[0]} please wait...\n this may take a while...")
            self.display_diff_sum_img()
        else:
            pass

    def choose_xrf_file(self):

        filename_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select XRF File')
        if filename_[0]:
            self.xrf_file = filename_[0]
            self.display_param["xrf_wd"] = self.xrf_file
            self.display_xrf_img()
            
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
            

    def display_xrf_img(self):

        
        if not self.display_param["xrf_wd"] == None:

            name_ = os.path.basename(self.display_param["xrf_wd"])
            self.xrf_img = tf.imread(self.display_param["xrf_wd"])
            
            try:
                self.xrf_plot_canvas.clear()
                self.create_pointer()
            except:
                pass

            im_array = self.xrf_img
            ysize,xsize = np.shape(im_array)
            self.statusbar.showMessage(f"Image Shape = {np.shape(im_array)}")
            # A plot area (ViewBox + axes) for displaying the image

            

            self.p1_xrf = self.xrf_plot_canvas.addPlot(title= "xrf_Image")
            #self.p1_xrf.setAspectLocked(True)
            self.p1_xrf.getViewBox().invertY(True)
            self.p1_xrf.getViewBox().setLimits(xMin = 0,
                                        xMax = xsize,
                                        yMin = 0,
                                        yMax = ysize
                                        )
            
            self.p1_xrf.addItem(self.scatterItem)

            # Item for displaying image data
            #self.img_item_xrf = pg.ImageItem(axisOrder = 'row-major')
            self.img_item_xrf = pg.ImageItem()
            if self.display_param["xrf_img_settings"]["display_log"]:
                im_array = np.nan_to_num(np.log10(im_array), nan=np.nan, posinf=np.nan, neginf=np.nan)
            self.img_item_xrf.setImage(im_array, opacity=1)
            self.p1_xrf.addItem(self.img_item_xrf)
        
            self.hist_xrf = pg.HistogramLUTItem()
            color_map_xrf = pg.colormap.get(self.display_param["xrf_img_settings"]["lut"])
            self.hist_xrf.gradient.setColorMap(color_map_xrf)
            self.hist_xrf.setImageItem(self.img_item_xrf)
            if self.display_param["xrf_img_settings"]["hist_lim"][0] == None or self.display_param["xrf_img_settings"]["hist_lim"][1] == None:
                self.hist_xrf.autoHistogramRange = False
            else:
                self.hist_xrf.autoHistogramRange = False
                self.hist_xrf.setLevels(min=self.display_param["xrf_img_settings"]["hist_lim"][0], 
                                    max=self.display_param["xrf_img_settings"]["hist_lim"][1])
            self.xrf_plot_canvas.addItem(self.hist_xrf)
            # self.img_item.hoverEvent = self.imageHoverEvent
            self.img_item_xrf.mousePressEvent = self.MouseClickEvent_xrf
            self.img_item_xrf.hoverEvent = self.imageHoverEvent_xrf
            self.roi_exists = False
            # self.roi_state = None

    def display_diff_sum_img(self):


        if not self.display_param["diff_wd"] == None:
            #TODO, may have memory issues
            self.diff_stack = tf.imread(self.display_param["diff_wd"])
            name_ = os.path.basename(self.display_param["diff_wd"])

            if self.diff_stack.ndim != 4:
                raise ValueError(f"{np.shape(self.diff_stack)}; only works for data shape with (im1,im2,det1,det2) structure")
            # print(np.shape(self.diff_stack))

            self.diff_sum_img = np.nansum(self.diff_stack, axis = (-2,-1))#memory efficient?
            if not self.roi is None:
                self.diff_img_view.removeItem(self.roi)
                self.roi = None
            try:
                self.diff_sum_plot_canvas.clear()
                self.diff_img_view.clear()
                self.create_pointer()
                
            except:
                pass
            
            
            im_array = self.diff_sum_img 
            ysize,xsize = np.shape(im_array)
            self.statusbar.showMessage(f"Image Shape = {np.shape(im_array)}")
            # A plot area (ViewBox + axes) for displaying the image

            self.p1_diff_sum = self.diff_sum_plot_canvas.addPlot(title= "Diff_Sum_Image")
            #self.p1_diff_sum.setAspectLocked(True)
            self.p1_diff_sum.getViewBox().invertY(True)
            self.p1_diff_sum.getViewBox().setLimits(xMin = 0,
                                        xMax = xsize,
                                        yMin = 0,
                                        yMax = ysize
                                        )
            
            self.p1_diff_sum.addItem(self.scatterItem)

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
            self.diff_sum_plot_canvas.addItem(self.hist_diff_sum)
            self.diff_img_view.hoverEvent = self.imageHoverEvent_diff
            self.img_item_diff_sum.mousePressEvent = self.MouseClickEvent_diff_sum
            self.img_item_diff_sum.hoverEvent = self.imageHoverEvent_diff_sum
            self.roi_exists = False

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
                
                if self.display_param["diff_img_settings"]["hist_lim"] == (None,None):
                    self.diff_img_view.setImage(self.single_diff)
                else:
                    self.diff_img_view.setImage(self.single_diff,autoLevels = False,autoHistogramRange=True)
                    levels = self.display_param["diff_img_settings"]["hist_lim"]
                    self.diff_img_view.setLevels(levels[0], levels[1])
                if self.roi == None:
                    self.create_roi(self.single_diff.shape)
                self.diff_img_view.addItem(self.roi)
                self.diff_img_view.setPredefinedGradient(self.display_param["diff_img_settings"]["lut"] )


            else: event.ignore()
        else: event.ignore()

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

                
                if self.display_param["diff_img_settings"]["hist_lim"] == (None,None):
                    self.diff_img_view.setImage(self.single_diff)
                else:
                    self.diff_img_view.setImage(self.single_diff,autoLevels = False,autoHistogramRange=True)
                    levels = self.display_param["diff_img_settings"]["hist_lim"]
                    self.diff_img_view.setLevels(levels[0], levels[1])
                if self.roi == None:
                    self.create_roi(self.single_diff.shape)
                self.diff_img_view.addItem(self.roi)
                self.diff_img_view.setPredefinedGradient(self.display_param["diff_img_settings"]["lut"] )

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
    
    def create_roi(self, im_array_dim):
        # if self.roi_state !=None:
        #     self.roi.setState(self.roi_state)   
        sz = np.ceil(im_array_dim[0]*0.2)
        roi_x = im_array_dim[1] 
        roi_y = im_array_dim[0] 
        self.roi =pg.PolyLineROI(
                                [[0, 0], [0, sz], [sz, sz], [sz, 0]],
                                pos=(int(roi_x // 2), int(roi_y // 2)),
                                maxBounds=QtCore.QRectF(0, 0, im_array_dim[1], im_array_dim[0]),
                                pen=pg.mkPen("r", width=1), 
                                hoverPen=pg.mkPen("w", width=1),
                                handlePen = pg.mkPen("m", width=3, ),
                                closed=True,
                                removable=True,
                                snapSize = 1,
                                translateSnap = True
                                )
        self.roi.setZValue(10)
        

    def get_mask_from_roi(self):

        # get the roi region:QPaintPathObject
        roiShape = self.roi.mapToItem(self.diff_img_view.getImageItem(), self.roi.shape())
        
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
    w = DiffViewWindow()
    w.show()
    sys.exit(app.exec_())