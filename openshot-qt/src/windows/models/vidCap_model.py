"""
 @file
 @brief This file contains the emoji model, used by the main window
 @author Jonathan Thomas <jonathan@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """

import os

from PyQt5.QtCore import QMimeData, Qt, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtWidgets import QMessageBox
import openshot  # Python module for libopenshot (required video editing module installed separately)

from classes import info
from classes.logger import log
from classes.app import get_app
from classes.time_parts import secondsToTimecode

import json


class VidCapStandardItemModel(QStandardItemModel):
    def __init__(self, parent=None):
        QStandardItemModel.__init__(self)

    def mimeData(self, indexes):
        # Create MimeData for drag operation
        data = QMimeData()

        # Get list of all selected file ids
        files = []
        for item in indexes:

            selected_item = self.itemFromIndex(item)
            files.append(selected_item.data())
        data.setText(json.dumps(files))
        data.setHtml("clip")

        # Return Mimedata
        return data

class VidCapsModel():
    def update_model(self, clear=True):
        log.info("updating VidCap model.")
        app = get_app()

        _ = app._tr

        # Clear all items
        if clear:
            self.model_paths = {}
            self.model.clear()
            self.emoji_groups.clear()

        # Add Headers
        self.model.setHorizontalHeaderLabels([_("Name")])

        # get BBOX with current timestamp
        bboxs = self.get_bbox()
        # view bboxes in VidCaps
        for bb in bboxs:
            row = []
            col = QStandardItem("Name")
            col.setText(bb["translation"])
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            self.model.appendRow(row)
    
    ### [Modify] ###
    def get_bbox(self):
        if not hasattr(self.app.window, "preview_thread"):
            return []
        
        # get current timestamp
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        current_position = (self.app.window.preview_thread.current_frame - 1) / fps_float
        current_timestamp = secondsToTimecode(current_position, fps["num"], fps["den"], use_milliseconds=True)
        # get detection result with current timestamp
        current_bbox = []
        if current_timestamp in self.app.window.timeline_sync.detections:
            current_bbox = self.app.window.timeline_sync.detections[current_timestamp]
        return current_bbox
    ### [End] ###

    def __init__(self, *args):

        # Create standard model
        self.app = get_app()
        self.model = VidCapStandardItemModel()
        self.model.setColumnCount(2)
        self.model_paths = {}
        self.emoji_groups = []

        # Create proxy models (for grouping, sorting and filtering)
        self.group_model = QSortFilterProxyModel()
        self.group_model.setDynamicSortFilter(False)
        self.group_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.group_model.setSortCaseSensitivity(Qt.CaseSensitive)
        self.group_model.setSourceModel(self.model)
        self.group_model.setSortLocaleAware(True)
        self.group_model.setFilterKeyColumn(1)

        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setDynamicSortFilter(False)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseSensitive)
        self.proxy_model.setSourceModel(self.group_model)
        self.proxy_model.setSortLocaleAware(True)

        # Attempt to load model testing interface, if requested
        # (will only succeed with Qt 5.11+)
        if info.MODEL_TEST:
            try:
                # Create model tester objects
                from PyQt5.QtTest import QAbstractItemModelTester
                self.model_tests = []
                for m in [self.proxy_model, self.group_model, self.model]:
                    self.model_tests.append(
                        QAbstractItemModelTester(
                            m, QAbstractItemModelTester.FailureReportingMode.Warning)
                    )
                log.info("Enabled {} model tests for emoji data".format(len(self.model_tests)))
            except ImportError:
                pass
'''
checkbox 사용시 참고용

    cb = QCheckBox('', self)
    cb.move(x,y)
    cb.toggle()
    cb.stateChanged.connect(self.acceptCaption)

    def acceptCaption(self, state):

        if state == Qt.Checked:
            #그 박스를 저장합니다!
        else:
            #그 박스는 저장 안합니다.

'''
