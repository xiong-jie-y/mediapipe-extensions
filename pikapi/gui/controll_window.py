import sys
from PyQt5.Qt import Qt

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

class MainWindow(QMainWindow):

    def __init__(self, key_state, action):
        super().__init__()
        self.title = 'PyQt test(QMainWindow)'
        self.width = 400
        self.height = 200
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height)
        label = QLabel('This is PyQt test.', self)
        self.key_state = key_state
        self.action = action

        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.key_state.value = 1
        elif event.key() == Qt.Key_B:
            self.key_state.value = -1

        if event.key() == Qt.Key_D:
            self.action.value = 1


    def keyReleaseEvent(self, event):
        self.key_state.value = 0
        self.action.value = 0
        super(MainWindow, self).keyReleaseEvent(event)


def create_window(key_state, action):
    app = QApplication(sys.argv)
    window = MainWindow(key_state, action)
    app.exec_()