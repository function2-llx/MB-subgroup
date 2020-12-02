import os

import csv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from MainWindow import Ui_MainWindow


def get_label_path(dirname):
    return os.path.join(dirname, 'label.csv')


class Annotator(QMainWindow, Ui_MainWindow):
    def update_label(self, label):
        if self.files[self.cur_idx][1] != label:
            if self.files[self.cur_idx][1] == -1:
                self.annotated += 1
                self.update_hint()
            self.files[self.cur_idx][1] = label
            with open(self.label_path, 'w', newline='') as f:
                csv.writer(f).writerows(self.files)

    def find_first(self):
        for idx, (_, label) in enumerate(self.files):
            if label == -1:
                return idx
        return -1

    def find_prev(self):
        for idx, (_, label) in reversed(list(enumerate(self.files[self.cur_idx:] + self.files[:self.cur_idx]))):
            if label == -1:
                return (idx + self.cur_idx) % len(self.files)
        return -1

    def find_next(self):
        for idx, (_, label) in enumerate(self.files[self.cur_idx + 1:] + self.files[:self.cur_idx + 1]):
            if label == -1:
                return (idx + self.cur_idx + 1) % len(self.files)
        return -1

    def __init__(self):
        super().__init__()
        self.files = []
        self.setupUi(self)
        self.act_open.triggered.connect(self.open_dir)
        self.act_first.triggered.connect(lambda: self.set_display(self.find_first()))
        self.act_prev.triggered.connect(lambda: self.set_display(self.find_prev()))
        self.act_next.triggered.connect(lambda: self.set_display(self.find_next()))
        self.display.hide()
        self.selector.hide()
        for idx, button in enumerate(self.group.buttons()):
            self.group.setId(button, idx)
        QShortcut(QtCore.Qt.Key_Up, self.selector).activated.connect(lambda: self.group.button((self.group.checkedId() + 2) % 3).click())
        QShortcut(QtCore.Qt.Key_Down, self.selector).activated.connect(lambda: self.group.button((self.group.checkedId() + 1) % 3).click())
        self.group.buttonClicked.connect(lambda: self.update_label(self.group.checkedId()))
        self.cur_idx = 0
        QShortcut(QtCore.Qt.Key_Left, self).activated.connect(lambda: self.set_display((self.cur_idx + len(self.files) - 1) % len(self.files)) if self.files else None)
        QShortcut(QtCore.Qt.Key_Right, self).activated.connect(lambda: self.set_display((self.cur_idx + 1) % len(self.files)) if self.files else None)
        self.show()

    def open_dir(self):
        dir = str(QFileDialog.getExistingDirectory(self, '打开文件夹'))
        if not dir:
            return

        self.label_path = get_label_path(dir)
        if os.path.exists(self.label_path):
            self.files = list(csv.reader(open(self.label_path)))
            for item in self.files:
                item[1] = int(item[1])
        else:
            self.files = []
            for dirpath, _, filenames in os.walk(dir):
                for filename in filenames:
                    if filename.endswith('.png'):
                        self.files.append([os.path.join(dirpath, filename), -1])
            self.statusBar().showMessage(f'共{len(self.files)}张图片')
        self.annotated = 0
        for _, label in self.files:
            if label != -1:
                self.annotated += 1
        self.set_display(0)

    def update_hint(self):
        self.hint.setText(f'第 {self.cur_idx + 1} 张图片，共 {len(self.files)} 张\n'
                          f'已标注 {self.annotated} 张，剩余 {len(self.files) - self.annotated} 张')

    def set_display(self, idx):
        if idx < 0 or len(self.files) <= idx:
            return
        self.cur_idx = idx
        self.update_hint()
        filepath, label = self.files[idx]
        pixmap = QPixmap(filepath)
        self.display.setPixmap(pixmap)
        self.display.setAlignment(QtCore.Qt.AlignRight)
        self.display.show()
        self.selector.show()
        self.group.setExclusive(False)
        self.no.setChecked(False)
        self.yes.setChecked(False)
        self.invalid.setChecked(False)
        self.group.setExclusive(True)
        if label != -1:
            self.group.button(label).setChecked(True)


if __name__ == '__main__':
    app = QApplication([])
    annotator = Annotator()
    app.exec()
