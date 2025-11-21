# This Python file uses the following encoding: utf-8
import sys
import os

os.environ["QT_API"] = "PySide6"

from PySide6.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene, QFileDialog, QMessageBox, QHeaderView
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, Signal

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MyApp

import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from scipy.stats import mode

class Data():
    def __init__(self):
        self.Sessions = None
        self.SessionNames = None
        self.Channels = None
        self.NumInChannels = None
        self.locations = None
        self.channel_locations = None
        self.acg = None
        self.peth = None
        self.idx_sort = None

        self.SimilarityMatrix = None
        self.Waveforms = None
        self.nTemplates = None

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

class ClickableGraphicsView(QGraphicsView):
    # Custom signal that emits integer pixel coordinates
    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        # Use position() instead of deprecated pos()
        scene_pos = self.mapToScene(event.position().toPoint())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        # Emit the signal with integer coordinates
        self.clicked.emit(x, y)

        # Call the base implementation (so selection etc. still works)
        super().mousePressEvent(event)



class MyApp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MyApp()
        self.ui.setupUi(self)

        self.ui.tabWidget.setCurrentWidget(self.ui.tab_1)

        # Let ImageViewer replace Designer
        self.ui.viewer1 = ImageViewer(self)
        self.ui.viewer1.setGeometry(self.ui.PETH_Fig1.geometry())
        self.ui.viewer1.setObjectName("PETH_Fig1")
        self.ui.PETH_Fig1.deleteLater()
        self.ui.PETH_Fig1 = self.ui.viewer1

        self.ui.viewer2 = ImageViewer(self)
        self.ui.viewer2.setGeometry(self.ui.PETH_Fig2.geometry())
        self.ui.viewer2.setObjectName("PETH_Fig2")
        self.ui.PETH_Fig2.deleteLater()
        self.ui.PETH_Fig2 = self.ui.viewer2

        self.ui.viewer3 = ClickableGraphicsView(self.ui.tab_2)
        self.ui.viewer3.setGeometry(self.ui.similarityMatrixView.geometry())
        self.ui.viewer3.setObjectName("similarityMatrixView")
        self.ui.similarityMatrixView.deleteLater()
        self.ui.similarityMatrixView = self.ui.viewer3

        self.ui.scene1 = QGraphicsScene()
        self.ui.scene2 = QGraphicsScene()
        self.ui.scene_similarity = QGraphicsScene()
        self.ui.sceneColorbar = QGraphicsScene()
        self.ui.sceneUnitsSelected = QGraphicsScene()

        # Create a Matplotlib Figure + Canvas
        self.ui.figure_depth = plt.Figure()
        self.ui.ax_depth = self.ui.figure_depth.add_subplot(111)
        self.ui.canvas = FigureCanvas(self.ui.figure_depth)
        self.ui.canvas.setGeometry(self.ui.depthView.geometry())
        self.ui.canvas.setParent(self.ui.depthView.parent())
        self.ui.canvas.setObjectName("depthView")
        self.ui.depthView.deleteLater()
        self.ui.depthView = self.ui.canvas

        self.ui.figure_waveform = plt.Figure()
        self.ui.ax_waveform = self.ui.figure_waveform.add_subplot(111)
        self.ui.canvas_waveform = FigureCanvas(self.ui.figure_waveform)
        self.ui.canvas_waveform.setGeometry(self.ui.waveformView.geometry())
        self.ui.canvas_waveform.setParent(self.ui.waveformView.parent())
        self.ui.canvas_waveform.setObjectName("waveformView")
        self.ui.waveformView.deleteLater()
        self.ui.waveformView = self.ui.canvas_waveform

        self.ui.figure_acg = plt.Figure()
        self.ui.ax_acg = self.ui.figure_acg.add_subplot(111)
        self.ui.canvas_acg = FigureCanvas(self.ui.figure_acg)
        self.ui.canvas_acg.setGeometry(self.ui.acgView.geometry())
        self.ui.canvas_acg.setParent(self.ui.acgView.parent())
        self.ui.canvas_acg.setObjectName("acgView")
        self.ui.acgView.deleteLater()
        self.ui.acgView = self.ui.canvas_acg

        self.ui.figure_peth_1 = plt.Figure()
        self.ui.ax_peth_1 = self.ui.figure_peth_1.add_subplot(111)
        self.ui.canvas_peth_1 = FigureCanvas(self.ui.figure_peth_1)
        self.ui.canvas_peth_1.setGeometry(self.ui.PETHView_1.geometry())
        self.ui.canvas_peth_1.setParent(self.ui.PETHView_1.parent())
        self.ui.canvas_peth_1.setObjectName("PETHView_1")
        self.ui.PETHView_1.deleteLater()
        self.ui.PETHView_1 = self.ui.canvas_peth_1

        self.ui.figure_peth_2 = plt.Figure()
        self.ui.ax_peth_2 = self.ui.figure_peth_2.add_subplot(111)
        self.ui.canvas_peth_2 = FigureCanvas(self.ui.figure_peth_2)
        self.ui.canvas_peth_2.setGeometry(self.ui.PETHView_2.geometry())
        self.ui.canvas_peth_2.setParent(self.ui.PETHView_2.parent())
        self.ui.canvas_peth_2.setObjectName("PETHView_2")
        self.ui.PETHView_2.deleteLater()
        self.ui.PETHView_2 = self.ui.canvas_peth_2

        self.ui.figure_peth_3 = plt.Figure()
        self.ui.ax_peth_3 = self.ui.figure_peth_3.add_subplot(111)
        self.ui.canvas_peth_3 = FigureCanvas(self.ui.figure_peth_3)
        self.ui.canvas_peth_3.setGeometry(self.ui.PETHView_3.geometry())
        self.ui.canvas_peth_3.setParent(self.ui.PETHView_3.parent())
        self.ui.canvas_peth_3.setObjectName("PETHView_3")
        self.ui.PETHView_3.deleteLater()
        self.ui.PETHView_3 = self.ui.canvas_peth_3

        # Table
        self.ui.model = QStandardItemModel()
        self.ui.model.setHorizontalHeaderLabels(["ClusterID", "ClusterSize", "Similarity", "Distance"])
        self.ui.tableView.setModel(self.ui.model)

        # Connect events
        self.ui.selectFolderButton.clicked.connect(self.select_folder)
        self.ui.LoadDataButton.clicked.connect(self.load_data)
        self.ui.saveButton.clicked.connect(self.save)
        self.ui.previousButton.clicked.connect(self.previousCluster)
        self.ui.nextButton.clicked.connect(self.nextCluster)
        self.ui.similarityMatrixView.clicked.connect(self.clickOnImage)
        self.ui.unitsToSplitEdit.textChanged.connect(self.updateUnitsSelectedView)
        self.ui.clusterEdit.textChanged.connect(self.clusterID_TextChanged)
        self.ui.selectUnitsButton.clicked.connect(self.plotModeChangedToSelected)
        self.ui.allUnitsButton.clicked.connect(self.plotModeChangedToAll)
        self.ui.splitButton.clicked.connect(self.split_cluster)
        self.ui.undoButton.clicked.connect(self.undo_clicked)
        self.ui.tabWidget.currentChanged.connect(self.change_tab)
        self.ui.tableView.clicked.connect(self.clickOnTable)
        self.ui.mergeButton.clicked.connect(self.merge_cluster)
        self.ui.clusterMergeEdit_1.textChanged.connect(self.mergeClusterID1_TextChanged)
        self.ui.clusterMergeEdit_2.textChanged.connect(self.mergeClusterID2_TextChanged)

        # Initialize data
        self.Data = Data()

        self.split_cluster_id = None
        self.merge_cluster_id1 = None
        self.merge_cluster_id2 = None

        self.last_splitted_units = None
        self.last_splitted_cluster_raw = None
        self.last_splitted_cluster_new = None
        self.last_idx_clusters = None
        self.last_merged_cluster = None
        self.last_merged_cluster_gone = None

        # Params
        self.plotMode = 'all' # all or separated
        self.isSplitState = True

    def change_tab(self, index):
        ui_list = [self.ui.Unit1_label, self.ui.Unit2_label, self.ui.unit1Edit, self.ui.unit2Edit, self.ui.label_5, self.ui.label_6,
            self.ui.previousButton, self.ui.nextButton, self.ui.undoButton, self.ui.saveButton,
            self.ui.selectUnitsButton, self.ui.allUnitsButton,
            self.ui.label_13, self.ui.label_colorbar_min, self.ui.label_colorbar_max, self.ui.colorbarView,
            self.ui.similarityMatrixView, self.ui.selectedUnitsView,
            self.ui.PETHView_1, self.ui.PETHView_2, self.ui.PETHView_3, self.ui.label, self.ui.label_2, self.ui.label_3,
            self.ui.depthView, self.ui.waveformView, self.ui.acgView]
        if index == 0:
            return
        elif index == 1:
            self.isSplitState = True
            if self.split_cluster_id is None:
                self.split_cluster_id = 1
            self.change_parents(ui_list, self.ui.tab_2)
            self.initializeSplit()
        elif index == 2:
            self.isSplitState = False
            if self.merge_cluster_id1 is None:
                self.merge_cluster_id1 = 1
            self.change_parents(ui_list, self.ui.tab_3)
            self.initializeMerge()

    def change_parents(self, widget_list, tab_parent):
        for w in widget_list:
            w.setParent(tab_parent)
            w.show()


    def update_units_selected(self):
        self.SessionName1 = self.Data.SessionNames[self.Unit1]
        self.SessionName2 = self.Data.SessionNames[self.Unit2]

        self.Channel1 = self.Data.Channels[self.Unit1]
        self.Channel2 = self.Data.Channels[self.Unit2]
        self.NumInChannel1 = self.Data.NumInChannels[self.Unit1]
        self.NumInChannel2 = self.Data.NumInChannels[self.Unit2]

        folder_image = os.path.join(self.Folder, 'PETH_Figs')
        img_path1 = os.path.join(folder_image, self.Animal+'_'+self.SessionName1+'_Ch'+str(self.Channel1)+'_Unit'+str(self.NumInChannel1)+'.png')
        img_path2 = os.path.join(folder_image, self.Animal+'_'+self.SessionName2+'_Ch'+str(self.Channel2)+'_Unit'+str(self.NumInChannel2)+'.png')

        img1 = QPixmap(img_path1)
        self.ui.scene1.clear()
        self.ui.scene1 = QGraphicsScene()
        self.ui.scene1.addPixmap(img1)
        self.ui.PETH_Fig1.setScene(self.ui.scene1)
        self.ui.PETH_Fig1.fitInView(self.ui.scene1.sceneRect(), Qt.KeepAspectRatio)

        img2 = QPixmap(img_path2)
        self.ui.scene2.clear()
        self.ui.scene2 = QGraphicsScene()
        self.ui.scene2.addPixmap(img2)
        self.ui.PETH_Fig2.setScene(self.ui.scene2)
        self.ui.PETH_Fig2.fitInView(self.ui.scene2.sceneRect(), Qt.KeepAspectRatio)

        # update labels
        self.ui.Unit1_label.setText(self.Animal+' '+self.SessionName1+' Ch'+str(self.Channel1)+' Unit'+str(self.NumInChannel1))
        self.ui.Unit2_label.setText(self.Animal+' '+self.SessionName2+' Ch'+str(self.Channel2)+' Unit'+str(self.NumInChannel2))
        self.ui.unit1Edit.setText(f"{self.Unit1}")
        self.ui.unit2Edit.setText(f"{self.Unit2}")

    def select_folder(self):
        folder_name = QFileDialog.getExistingDirectory()
        self.ui.folderEdit.setText(folder_name)

    def load_data(self):
        folder_name = self.ui.folderEdit.text()

        if not os.path.isdir(folder_name):
            QMessageBox.critical(self, "Error", "The data folder doesn't exist!")

        self.Folder = folder_name
        self.FolderCuration = os.path.join(self.Folder, 'DANT_Curation')

        print('Loading data ...')

        df_meta = pd.read_csv(os.path.join(self.FolderCuration, 'Meta.csv'))

        # Read meta data
        self.Animal = df_meta['Animal'][0]
        self.NumSessions = df_meta['NumSessions'][0]
        self.NumUnits = df_meta['NumUnits'][0]

        self.ui.animalEdit.setText(self.Animal)
        self.ui.unitNumEdit.setText(f"{self.NumUnits}")
        self.ui.sessionNumEdit.setText(f"{self.NumSessions}")

        # Load all data
        if os.path.isfile(os.path.join(self.FolderCuration, 'IdxClusters.npy')):
            # self.ClusterMatrix = np.load(os.path.join(self.FolderCuration, 'ClusterMatrix.npy'))
            self.IdxClusters = np.squeeze(np.load(os.path.join(self.FolderCuration, 'IdxClusters.npy')))
        else:
            # self.ClusterMatrix = np.load(os.path.join(self.FolderCuration, 'ClusterMatrixRaw.npy'))
            self.IdxClusters = np.squeeze(np.load(os.path.join(self.FolderCuration, 'IdxClustersRaw.npy')))

        self.Data.Channels = np.squeeze(np.astype(np.load(os.path.join(self.FolderCuration, 'Channels.npy')), np.int64))
        self.Data.NumInChannels = np.squeeze(np.astype(np.load(os.path.join(self.FolderCuration, 'NumInChannels.npy')), np.int64))
        self.Data.Sessions = np.squeeze(np.astype(np.load(os.path.join(self.FolderCuration, 'session_index.npy')), np.int64))
        self.Data.SessionNames = pd.read_csv(os.path.join(self.FolderCuration, 'session_names.csv'), dtype=str)['SessionNames']
        self.Data.locations = np.load(os.path.join(self.FolderCuration, 'locations.npy'))
        self.Data.channel_locations = np.load(os.path.join(self.FolderCuration, 'channel_locations.npy'))
        self.Data.acg = np.load(os.path.join(self.FolderCuration, 'ACG.npy'))
        self.Data.peth = np.load(os.path.join(self.FolderCuration, 'peth.npy'))
        self.Data.idx_sort = np.squeeze(np.astype(np.load(os.path.join(self.FolderCuration, 'sort_index.npy')), np.int64))

        self.Data.SimilarityMatrix = np.load(os.path.join(self.FolderCuration, 'SimilarityMatrix.npy'))
        # self.Data.Waveforms = np.load(os.path.join(self.FolderCuration, 'waveforms_corrected.npy'), mmap_mode="r")
        self.Data.Waveforms = np.load(os.path.join(self.FolderCuration, 'waveforms_corrected.npy'))
        if len(self.Data.Waveforms.shape) > 3:
            self.Data.nTemplates = 2
        else:
            self.Data.nTemplates = 1


        # Load last states in this app
        if os.path.isfile(os.path.join(self.FolderCuration, 'curation_app.csv')):
            df_state = pd.read_csv(os.path.join(self.FolderCuration, 'curation_app.csv'))
            if not np.isnan(df_state['split_cluster_id'][0]):
                self.split_cluster_id = df_state['split_cluster_id'][0]

            if not np.isnan(df_state['merge_cluster_id1'][0]):
                self.merge_cluster_id1 = df_state['merge_cluster_id1'][0]

            if not np.isnan(df_state['merge_cluster_id2'][0]):
                self.merge_cluster_id2 = df_state['merge_cluster_id2'][0]


        # Disable the first step and move to splitting / merging
        self.ui.tab_1.setEnabled(False)
        self.ui.tab_2.setEnabled(True)
        self.ui.tab_3.setEnabled(True)
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_2)

        if self.split_cluster_id is None:
            self.split_cluster_id = 1
        self.initializeSplit()


    def initializeSplit(self):
        self.isSplitState = True

        self.Units = np.where(self.IdxClusters == self.split_cluster_id)[0]
        self.Unit1 = self.Units[0]
        self.Unit2 = self.Units[1]

        self.ui.unitsToSplitEdit.setText('')
        self.ui.sceneUnitsSelected.clear()
        self.plotMode = 'all'
        self.ui.clusterEdit.setText(f"{self.split_cluster_id}")

        self.update_units_selected()
        self.updateFigures()
        self.updateDepthView()
        self.updatePETHView()
        self.updateWaveformView()
        self.updateAcgView()

    def initializeMerge(self):
        self.isSplitState = False

        self.updateMergeTable()
        self.merge_cluster_id2 = self.similar_cluster_ids[0]

        self.updateMergeCluster2()

    def updateMergeCluster2(self):
        self.UnitsCluster1 = np.where(self.IdxClusters == self.merge_cluster_id1)[0]
        self.UnitsCluster2 = np.where(self.IdxClusters == self.merge_cluster_id2)[0]
        self.UnitsMerge = np.concatenate((self.UnitsCluster1, self.UnitsCluster2))
        self.Unit1 = self.UnitsCluster1[0]
        self.Unit2 = self.UnitsCluster2[0]

        self.plotMode = 'separated'

        self.ui.clusterMergeEdit_1.setText(f"{self.merge_cluster_id1}")
        self.ui.clusterMergeEdit_2.setText(f"{self.merge_cluster_id2}")

        self.update_units_selected()
        self.updateFigures()
        self.updateUnitsSelectedView()
        self.updateDepthView()
        self.updatePETHView()
        self.updateWaveformView()
        self.updateAcgView()

    def updateMergeTable(self):
        self.rank_units = np.array([np.where(self.Data.idx_sort == unit+1)[0][0] for unit in range(self.NumUnits)])
        n_clusters = self.IdxClusters.max().astype(np.int64)
        self.rank_clusters = np.array([np.mean(self.rank_units[self.IdxClusters == k+1]) for k in range(n_clusters)])

        n_nearest_clusters = 15
        idx_sort = np.argsort(np.abs(self.rank_clusters - self.rank_clusters[self.merge_cluster_id1-1])).astype(np.int64)

        self.similar_cluster_ids = (idx_sort[:n_nearest_clusters+1] + 1).astype(np.int64)
        self.similar_cluster_ids = self.similar_cluster_ids[self.similar_cluster_ids != self.merge_cluster_id1]

        cluster_size = [len(np.where(self.IdxClusters == id)[0]) for id in self.similar_cluster_ids]

        distance = np.abs(self.rank_clusters[self.similar_cluster_ids-1] - self.rank_clusters[self.merge_cluster_id1-1])
        similarity = np.zeros(n_nearest_clusters)

        units1 = np.where(self.IdxClusters == self.merge_cluster_id1)[0]
        for k in range(n_nearest_clusters):
            units2 = np.where(self.IdxClusters == self.similar_cluster_ids[k])[0]
            similarity[k] = self.Data.SimilarityMatrix[np.ix_(units1, units2)].mean()

        self.ui.model.clear()
        self.ui.model = QStandardItemModel()
        self.ui.model.setHorizontalHeaderLabels(["Cluster", "Size", "Sim", "Dist"])
        self.ui.tableView.setModel(self.ui.model)
        for k in range(n_nearest_clusters):
            row = [
                QStandardItem(f"{self.similar_cluster_ids[k]}"),
                QStandardItem(f"{cluster_size[k]}"),
                QStandardItem(f"{similarity[k]:.2f}"),
                QStandardItem(f"{distance[k]:.2f}")
            ]
            self.ui.model.appendRow(row)

        self.ui.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableView.show()


    def merge_cluster(self):
        # Merge criteria: No session collisions
        # The cluster ids should be continued

        sessions1 = self.Data.Sessions[self.UnitsCluster1]
        sessions2 = self.Data.Sessions[self.UnitsCluster2]

        if len(sessions1) < 1 or len(sessions2) < 1:
            QMessageBox.critical(self, "Error", "Merge failed!")
            return

        if np.any(np.isin(sessions1, sessions2)):
            QMessageBox.critical(self, "Error", "Merge failed! These 2 clusters contain units from the same sessions!")
            return

        new_id = self.merge_cluster_id1
        old_id = self.merge_cluster_id2

        self.last_idx_clusters = self.IdxClusters.copy()
        self.IdxClusters[self.UnitsCluster2] = new_id
        print(f"Cluster #{old_id} is merged into Cluster #{new_id}!")

        # reorder the cluster ids
        n_clusters = np.unique(self.IdxClusters).size
        unique_clusters = np.unique(self.IdxClusters)
        idx_clusters_new = np.zeros(self.NumUnits)
        for k in range(n_clusters):
            idx_clusters_new[self.IdxClusters==unique_clusters[k]] = k+1

        assert(np.all(idx_clusters_new > 0))
        assert(np.unique(idx_clusters_new).size == idx_clusters_new.max())

        self.last_merged_cluster = new_id
        self.last_merged_cluster_gone = old_id
        self.IdxClusters = idx_clusters_new

        # refresh the window
        self.merge_cluster_id1 = new_id
        self.initializeMerge()

    def split_cluster(self):
        unit_str = self.ui.unitsToSplitEdit.text()
        if not unit_str:
            units = []
        else:
            units = [int(x.strip()) for x in unit_str.split(', ')]

        if len(units) == 0 or len(units) == len(self.Units):
            QMessageBox.information(self, "Info", "No units to split!")
            return

        # give the selected units new IDs
        max_id = int(self.IdxClusters.max())
        old_id = self.split_cluster_id
        new_id = max_id+1
        self.IdxClusters[units] = new_id

        self.last_splitted_units = units
        self.last_splitted_cluster_raw = old_id
        self.last_splitted_cluster_new = new_id
        print(f"Cluster #{old_id} is splitted to Cluster #{old_id} and Cluster #{new_id}!")

        # refresh the window
        self.initializeSplit()


    def undo_clicked(self):
        if self.isSplitState:
            self.undo_split()
        else:
            self.undo_merge()

    def undo_split(self):
        if self.last_splitted_units is None or self.last_splitted_cluster_raw is None or self.last_splitted_cluster_new is None:
            QMessageBox.critical(self, "Error", "Undo split failed! No information about last split action!")
            return

        units = np.where(self.IdxClusters == self.last_splitted_cluster_new)[0]
        if set(units) != set(self.last_splitted_units):
            QMessageBox.critical(self, "Error", "Undo split failed! The splitted units are different!")
            return

        if self.IdxClusters.max() != self.last_splitted_cluster_new:
            QMessageBox.critical(self, "Error", "Undo split failed!")
            return

        # give the raw IDs back for these units
        self.IdxClusters[self.last_splitted_units] = self.last_splitted_cluster_raw

        print(f"Undo split action from Cluster #{self.last_splitted_cluster_raw} to Cluster #{self.last_splitted_cluster_raw} and Cluster #{self.last_splitted_cluster_new}!")

        self.split_cluster_id = self.last_splitted_cluster_raw
        self.last_splitted_units = None
        self.last_splitted_cluster_raw = None
        self.last_splitted_cluster_new = None

        # refresh the window
        self.initializeSplit()

    def undo_merge(self):
        if self.last_idx_clusters is None or self.last_merged_cluster is None or self.last_merged_cluster_gone is None:
            QMessageBox.critical(self, "Error", "Undo merge failed! No information about last merge action!")
            return

        # give the raw IDs back for these units
        self.IdxClusters = self.last_idx_clusters.copy()

        print(f"Undo merge action from Cluster #{self.last_merged_cluster_gone} to Cluster #{self.last_merged_cluster}!")

        self.merge_cluster_id1 = self.last_merged_cluster
        self.last_idx_clusters = None
        self.last_merged_cluster = None
        self.last_merged_cluster_gone = None

        # refresh the window
        self.initializeMerge()

    def save(self):
        # save the curated cluster matrix
        # np.save(os.path.join(self.FolderCuration, 'ClusterMatrix.npy'), self.ClusterMatrix)
        np.save(os.path.join(self.FolderCuration, 'IdxClusters.npy'), self.IdxClusters)

        # save the current states
        states = {"split_cluster_id":self.split_cluster_id,
            "merge_cluster_id1":self.merge_cluster_id1,
            "merge_cluster_id2":self.merge_cluster_id2}
        df_state = pd.DataFrame(states, index=[0])
        df_state.to_csv(os.path.join(self.FolderCuration, 'curation_app.csv'))

    def updateFigures(self):
        if self.isSplitState:
            units = self.Units
        else:
            units = self.UnitsMerge
        img = self.Data.SimilarityMatrix[np.ix_(units, units)]

        # Create a scene
        self.ui.scene_similarity.clear()
        self.ui.scene_similarity = QGraphicsScene()
        self.ui.similarityMatrixView.setScene(self.ui.scene_similarity)

        # Normalize to [0.5, 3.5]
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)

        if img_min == img_max:
            img = img / img_max
        else:
            img = (img - img_min) / (img_max - img_min)

        # Apply custom colormap (viridis here, but you can choose any)
        cmap = matplotlib.colormaps.get_cmap("viridis")
        img = (cmap(img)[:, :, :3] * 255).astype(np.uint8)  # RGB only

        # Convert to QImage
        h, w, ch = img.shape
        qimg = QImage(img, w, h, ch * w, QImage.Format_RGB888)

        # Convert to QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # Add to scene
        self.ui.scene_similarity.addPixmap(pixmap)

        # Fit image to view
        self.ui.similarityMatrixView.fitInView(self.ui.scene_similarity.sceneRect(), Qt.KeepAspectRatio)

        # Add colorbar
        # 1. Define range
        vmin, vmax = img_min, img_max

        # 2. Create a vertical gradient (e.g. 256 pixels tall)
        height = 256
        width = 30   # narrow strip
        values = np.linspace(vmax, vmin, height)  # top→bottom

        if vmin == vmax:
            normed = np.ones_like(values)
        else:
            normed = (values - vmin) / (vmax - vmin)

        cmap = matplotlib.colormaps.get_cmap("viridis")

        # 3. Apply colormap
        colored = (cmap(normed)[:, :3] * 255).astype(np.uint8)  # RGB only
        colored = np.tile(colored[:, None, :], (1, width, 1))   # expand to width

        # 4. Convert to QImage
        h, w, ch = colored.shape
        qimg = QImage(colored.data, w, h, ch * w, QImage.Format_RGB888)

        # 5. Wrap in QPixmap and display in colorbarView
        pixmap = QPixmap.fromImage(qimg)
        self.ui.sceneColorbar.clear()
        self.ui.sceneColorbar = QGraphicsScene()
        self.ui.sceneColorbar.addPixmap(pixmap)

        self.ui.colorbarView.setScene(self.ui.sceneColorbar)
        self.ui.colorbarView.fitInView(self.ui.sceneColorbar.sceneRect(), Qt.IgnoreAspectRatio)
        self.ui.label_colorbar_min.setText(f"{img_min:.1f}")
        self.ui.label_colorbar_max.setText(f"{img_max:.1f}")

    def updateDepthView(self):
        if self.isSplitState:
            units = self.Units
        else:
            units = self.UnitsMerge

        sessions_plot = self.Data.Sessions[units]
        depth_plot = self.Data.locations[units, 1]

        self.ui.ax_depth.clear()

        if self.isSplitState:
            self.ui.ax_depth.plot(sessions_plot, depth_plot, 'k-')
        else:
            idx1 = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster1]
            idx2 = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster2]
            self.ui.ax_depth.plot(sessions_plot[idx1], depth_plot[idx1], 'k-')
            self.ui.ax_depth.plot(sessions_plot[idx2], depth_plot[idx2], 'b-')

        self.ui.ax_depth.set_xlim(sessions_plot.min()-0.5, sessions_plot.max()+0.5)
        self.ui.ax_depth.set_ylim(depth_plot.min()-5, depth_plot.max()+5)
        self.ui.ax_depth.set_xlabel('Sessions', fontsize=7)
        self.ui.ax_depth.set_ylabel('Depth (μm)', fontsize=7)
        self.ui.ax_depth.tick_params(axis="both", labelsize=7)

        cmap = matplotlib.colormaps.get_cmap("winter")
        cmap2 = matplotlib.colormaps.get_cmap("copper")
        norm = Normalize(vmin=1, vmax=self.NumSessions)

        colors = [cmap(norm(i)) for i in sessions_plot]

        if self.plotMode == 'separated':
            if self.isSplitState:
                unit_str = self.ui.unitsToSplitEdit.text()
                if not unit_str:
                    units_this = []
                else:
                    units_this = [int(x.strip()) for x in unit_str.split(', ')]

                idx = [np.where(self.Units==unit)[0][0] for unit in units_this]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))
            else:
                idx = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster1]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))

        self.ui.ax_depth.scatter(sessions_plot, depth_plot, c=colors)

        self.ui.figure_depth.tight_layout()
        self.ui.depthView.draw()


    def updatePETHView(self):
        if self.isSplitState:
            units = self.Units
        else:
            units = self.UnitsMerge

        sessions_plot = self.Data.Sessions[units]
        t = np.linspace(-500, 500, 1000)
        peth_1 = np.squeeze(self.Data.peth[units,:,0])
        peth_2 = np.squeeze(self.Data.peth[units,:,1])
        peth_3 = np.squeeze(self.Data.peth[units,:,2])

        # Define colormap + normalization
        cmap = matplotlib.colormaps.get_cmap("winter")
        cmap2 = matplotlib.colormaps.get_cmap("copper")
        norm = Normalize(vmin=1, vmax=self.NumSessions)

        colors = [cmap(norm(i)) for i in sessions_plot]

        if self.plotMode == 'separated':
            if self.isSplitState:
                unit_str = self.ui.unitsToSplitEdit.text()
                if not unit_str:
                    units_this = []
                else:
                    units_this = [int(x.strip()) for x in unit_str.split(', ')]

                idx = [np.where(self.Units==unit)[0][0] for unit in units_this]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))
            else:
                idx = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster1]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))

        # Plot
        self.ui.ax_peth_1.clear()
        self.ui.ax_peth_2.clear()
        self.ui.ax_peth_3.clear()

        for i in range(len(sessions_plot)):
            self.ui.ax_peth_1.plot(t, np.squeeze(peth_1[i,:]), color=colors[i])
            self.ui.ax_peth_2.plot(t, np.squeeze(peth_2[i,:]), color=colors[i])
            self.ui.ax_peth_3.plot(t, np.squeeze(peth_3[i,:]), color=colors[i])

        # Suppose each axis already has data plotted
        ymins = []
        ymaxs = []

        for ax in [self.ui.ax_peth_1, self.ui.ax_peth_2, self.ui.ax_peth_3]:
           ymin, ymax = ax.get_ylim()
           ymins.append(ymin)
           ymaxs.append(ymax)

        # Compute global limits
        global_ymin = min(ymins)
        global_ymax = max(ymaxs)

        # Apply to all axes
        for ax in [self.ui.ax_peth_1, self.ui.ax_peth_2, self.ui.ax_peth_3]:
            ax.set_ylim(global_ymin, global_ymax)

        self.ui.ax_peth_1.tick_params(axis="both", labelsize=7)
        self.ui.ax_peth_2.tick_params(axis="both", labelsize=7)
        self.ui.ax_peth_3.tick_params(axis="both", labelsize=7)
        self.ui.figure_peth_1.tight_layout()
        self.ui.figure_peth_2.tight_layout()
        self.ui.figure_peth_3.tight_layout()
        self.ui.PETHView_1.draw()
        self.ui.PETHView_2.draw()
        self.ui.PETHView_3.draw()

    def updateAcgView(self):
        if self.isSplitState:
            units = self.Units
        else:
            units = self.UnitsMerge

        sessions_plot = self.Data.Sessions[units]
        t = np.linspace(-300, 300, 601)
        acg = self.Data.acg[units,:]

        # Define colormap + normalization
        cmap = matplotlib.colormaps.get_cmap("winter")
        cmap2 = matplotlib.colormaps.get_cmap("copper")
        norm = Normalize(vmin=1, vmax=self.NumSessions)

        colors = [cmap(norm(i)) for i in sessions_plot]
        if self.plotMode == 'separated':
            if self.isSplitState:
                unit_str = self.ui.unitsToSplitEdit.text()
                if not unit_str:
                    units_this = []
                else:
                    units_this = [int(x.strip()) for x in unit_str.split(', ')]

                idx = [np.where(self.Units==unit)[0][0] for unit in units_this]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))
            else:
                idx = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster1]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))

        # Plot
        self.ui.ax_acg.clear()

        for i in range(len(sessions_plot)):
            self.ui.ax_acg.plot(t, np.squeeze(acg[i,:]), color=colors[i])

        self.ui.ax_acg.set_xlabel('Time (ms)', fontsize=7)
        self.ui.ax_acg.tick_params(axis="both", labelsize=7)

        self.ui.figure_acg.tight_layout()
        self.ui.acgView.draw()

    def updateWaveformView(self):
        if self.isSplitState:
            units = self.Units
        else:
            units = self.UnitsMerge

        if self.Data.nTemplates == 1:
            waveforms = self.Data.Waveforms[units,:,:]
        else:
            mean_ptt_1 = (np.max(self.Data.Waveforms[units,:,:,0], axis=2) - np.min(self.Data.Waveforms[units,:,:,0], axis=2)).mean()
            mean_ptt_2 = (np.max(self.Data.Waveforms[units,:,:,1], axis=2) - np.min(self.Data.Waveforms[units,:,:,1], axis=2)).mean()
            if mean_ptt_1 >= mean_ptt_2:
                waveforms = self.Data.Waveforms[units,:,:,0]
            else:
                waveforms = self.Data.Waveforms[units,:,:,1]

        ptt = np.max(waveforms, axis=2) - np.min(waveforms, axis=2)
        peak_channels = np.argmax(np.squeeze(ptt), axis=1)
        ch = mode(peak_channels)[0]
        amplitude = ptt.max()

        n_channels = 14
        dx_scale = 1
        dy_scale = 3

        distance_to_location = np.abs(self.Data.channel_locations[:,1] - self.Data.channel_locations[ch,1])*dy_scale + np.abs(self.Data.channel_locations[:,0] - self.Data.channel_locations[ch,0])*dx_scale

        idx_sort = np.argsort(distance_to_location)
        ch_included = idx_sort[:n_channels]
        waveforms_plot = waveforms[:,ch_included,:]

        x_plot = []
        y_plot = []

        samples_plot = np.arange(waveforms_plot.shape[2])
        samples_plot = samples_plot - samples_plot.mean()

        sample_scale = 1
        x_scale = 3
        y_scale = 1
        waveform_scale = 1/amplitude*50;
        for j in range(len(units)):
            x_this = []
            y_this = []
            for k in range(n_channels):
                x = self.Data.channel_locations[ch_included[k], 0];
                y = self.Data.channel_locations[ch_included[k], 1];

                x_this.append(x*x_scale+samples_plot*sample_scale)
                x_this.append([np.nan])
                y_this.append(y*y_scale+np.squeeze(waveforms_plot[j,k,:]).T * waveform_scale)
                y_this.append([np.nan])

            x_plot.append(np.concatenate(x_this))
            y_plot.append(np.concatenate(y_this))

        sessions_plot = self.Data.Sessions[units]
        # Define colormap + normalization
        cmap = matplotlib.colormaps.get_cmap("winter")
        cmap2 = matplotlib.colormaps.get_cmap("copper")
        norm = Normalize(vmin=1, vmax=self.NumSessions)

        colors = [cmap(norm(i)) for i in sessions_plot]

        if self.plotMode == 'separated':
            if self.isSplitState:
                unit_str = self.ui.unitsToSplitEdit.text()
                if not unit_str:
                    units_this = []
                else:
                    units_this = [int(x.strip()) for x in unit_str.split(', ')]

                idx = [np.where(self.Units==unit)[0][0] for unit in units_this]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))
            else:
                idx = [np.where(self.UnitsMerge==unit)[0][0] for unit in self.UnitsCluster1]
                for j in idx:
                    colors[j] = cmap2(norm(sessions_plot[j]))

        self.ui.ax_waveform.clear()
        for k in range(len(units)):
            self.ui.ax_waveform.plot(x_plot[k], y_plot[k], color=colors[k])

        self.ui.ax_waveform.axis("off")

        self.ui.waveformView.draw()

    def previousCluster(self):
        if self.isSplitState:
            cluster_id = self.split_cluster_id - 1

            units = np.where(self.IdxClusters == cluster_id)[0]
            while len(units) <= 1:
                cluster_id = cluster_id - 1
                if cluster_id < 1:
                    QMessageBox.information(self, "Info", "No more clusters!")
                    return
                units = np.where(self.IdxClusters == cluster_id)[0]

            self.split_cluster_id = cluster_id
            self.initializeSplit()
        else:
            cluster_id = self.merge_cluster_id1 - 1
            if cluster_id < 1:
                QMessageBox.information(self, "Info", "No more clusters!")
                return

            self.merge_cluster_id1 = cluster_id
            self.initializeMerge()

    def nextCluster(self):
        max_cluster = max(self.IdxClusters)
        if self.isSplitState:
            cluster_id = self.split_cluster_id + 1

            units = np.where(self.IdxClusters == cluster_id)[0]
            while len(units) <= 1:
                cluster_id = cluster_id + 1
                if max_cluster < cluster_id:
                    QMessageBox.information(self, "Info", "No more clusters!")
                    return
                units = np.where(self.IdxClusters == cluster_id)[0]

            self.split_cluster_id = cluster_id
            self.initializeSplit()
        else:
            cluster_id = self.merge_cluster_id1 + 1
            if cluster_id > max_cluster:
                QMessageBox.information(self, "Info", "No more clusters!")
                return

            self.merge_cluster_id1 = cluster_id
            self.initializeMerge()

    def updateUnitsSelectedView(self):
        if self.isSplitState:
            img = 255*np.ones((1, len(self.Units), 3)).astype(np.uint8)

            unit_str = self.ui.unitsToSplitEdit.text()
            if not unit_str:
                units = []
            else:
                units = [int(x.strip()) for x in unit_str.split(', ')]

            idx = [np.where(self.Units==unit)[0][0] for unit in units]
        else:
            img = 255*np.ones((1, len(self.UnitsMerge), 3)).astype(np.uint8)
            idx = [np.where(self.UnitsMerge == unit)[0][0] for unit in self.UnitsCluster1]

        img[:,idx,:] = np.uint8(0)

        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.ui.sceneUnitsSelected.clear()
        self.ui.sceneUnitsSelected = QGraphicsScene()
        self.ui.sceneUnitsSelected.addPixmap(pixmap)

        self.ui.selectedUnitsView.setScene(self.ui.sceneUnitsSelected)
        self.ui.selectedUnitsView.fitInView(self.ui.sceneUnitsSelected.sceneRect(), Qt.IgnoreAspectRatio)

        if self.plotMode == 'separated' and self.isSplitState:
            self.plotModeChangedToSelected()

    def clusterID_TextChanged(self):
        raw_id = self.split_cluster_id

        if not self.ui.clusterEdit.text().isdigit():
            return

        new_id = int(self.ui.clusterEdit.text())

        units = np.where(self.IdxClusters == new_id)[0]
        if len(units) <= 1:
            QMessageBox.information(self, "Info", "This cluster contains less than 1 unit!")
            self.ui.clusterEdit.setText(f"{raw_id}")
            return

        self.split_cluster_id = new_id
        self.initializeSplit()

    def mergeClusterID1_TextChanged(self):
        if not self.ui.clusterMergeEdit_1.text().isdigit():
            return

        new_id = int(self.ui.clusterMergeEdit_1.text())

        units = np.where(self.IdxClusters == new_id)[0]
        if len(units) <= 0:
            QMessageBox.information(self, "Info", "Bad cluster!")
            return

        self.merge_cluster_id1 = new_id
        self.initializeMerge()

    def mergeClusterID2_TextChanged(self):
        if not self.ui.clusterMergeEdit_2.text().isdigit():
            return

        new_id = int(self.ui.clusterMergeEdit_2.text())

        units = np.where(self.IdxClusters == new_id)[0]
        if len(units) <= 0:
            QMessageBox.information(self, "Info", "Bad cluster!")
            return

        self.merge_cluster_id2 = new_id
        self.updateMergeCluster2()

    def plotModeChangedToSelected(self):
        self.plotMode = 'separated'
        self.updateWaveformView()
        self.updatePETHView()
        self.updateAcgView()
        self.updateDepthView()
    def plotModeChangedToAll(self):
        self.plotMode = 'all'
        self.updateWaveformView()
        self.updatePETHView()
        self.updateAcgView()
        self.updateDepthView()

    def clickOnImage(self, x, y):
        if self.isSplitState:
            if x < 0: x = 0
            if x >= len(self.Units): x = len(self.Units)-1
            if y < 0: y = 0
            if y >= len(self.Units): y = len(self.Units)-1

            self.Unit1 = self.Units[x]
            self.Unit2 = self.Units[y]
        else:
            if x < 0: x = 0
            if x >= len(self.UnitsMerge): x = len(self.UnitsMerge)-1
            if y < 0: y = 0
            if y >= len(self.UnitsMerge): y = len(self.UnitsMerge)-1

            self.Unit1 = self.UnitsMerge[x]
            self.Unit2 = self.UnitsMerge[y]

        if x != y:
            self.update_units_selected()
        elif self.isSplitState:
            unit_str = self.ui.unitsToSplitEdit.text()
            if not unit_str:
                units = []
            else:
                units = [int(x.strip()) for x in unit_str.split(', ')]

            if self.Units[x] not in units:
                units.append(self.Units[x])

            self.ui.unitsToSplitEdit.setText(', '.join(str(x) for x in units))
            self.updateUnitsSelectedView()



    def clickOnTable(self, index):
        self.merge_cluster_id2 = self.similar_cluster_ids[index.row()]
        self.updateMergeCluster2()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Save Data",
            "Do you want to save your changes before closing?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self.save()
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:  # Cancel
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyApp()
    widget.showMaximized()
    sys.exit(app.exec())
