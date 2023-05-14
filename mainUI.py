import sys
import os
import subprocess

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon

from ui import Ui_MainWindow

import ctypes
if sys.platform == 'win32':
    CP_console = f"cp{ctypes.cdll.kernel32.GetConsoleOutputCP()}"

else:
    CP_console = 'utf-8'


class AveragerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(AveragerGUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_UI()

        self.ifile = ''

        self.ui.pushButton.clicked.connect(self.open_file)
        self.ui.RunButton.clicked.connect(self.callProgram)

        self.process = QtCore.QProcess(self)
        self.process.readyRead.connect(self.dataReady)

        # self.process.started.connect(lambda: self.ui.RunButton.setEnabled(False))
        self.process.finished.connect(lambda: print("FINISHED !!!"))

    def init_UI(self):
        self.setWindowTitle("Averager GUI - Lycoris Radiata")
        self.setWindowIcon(QIcon("icon.png"))

    def dataReady(self):
        cursor = self.ui.OutputBrowser.textCursor()
        cursor.movePosition(cursor.End)

        # Here we have to decode the QByteArray
        cursor.insertText(
            str(self.process.readAll(), CP_console))

        cursor.insertText(
            str(self.process.readAllStandardError().data().decode(CP_console).rstrip(' \n')) + '\n')

        self.ui.OutputBrowser.ensureCursorVisible()

    def callProgram(self):
        # run the process
        # `start` takes the exec and a list of arguments
        command = self.make_run()
        print('Running: ' + command)
        self.process.start(command)

    def open_file(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV Data",
                                                      os.getcwd(),
                                                      "CSV files (*.csv);; All Files (*)")

        if fname[0] != '':
            self.ui.pushButton.setText(fname[0].split('/')[-1])
            self.ifile = fname[0]
            self.make_test_run()

    def make_test_run(self):
        command = "python main.py " + self.ifile + " -v --leave"
        print("Executing: " + command)

        # Create a subprocess and start the ping command
        ping_process = subprocess.Popen(command, shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        self.ui.OutputBrowser.clear()
        for line in iter(ping_process.stdout.readline, b''):
            line = line.decode('utf-8').rstrip()

            # if line.startswith("⚊") or "⚊" in line:
            #     text.delete('end - %d chars' % (len(previous_line) + 2), 'end-1c')

            # text.insert(tk.END, line + '\n')
            self.ui.OutputBrowser.append(line)

    def make_run(self):
        # pylint: disable=too-complex

        command = 'python main.py '

        if self.ifile == '':
            return

        command += self.ifile + ' '

        if self.ui.headerEdit.text().isnumeric():
            command += '-h ' + self.ui.headerEdit.text() + ' '

        if self.ui.checkBox.isChecked():
            command += '-v '

        command += '-d ' + self.ui.XEdit.text() + ',' + self.ui.YEdit.text() + ',' + \
            self.ui.ZEdit.text() + ' '

        command += '-j '
        if self.ui.basic2Button.isChecked():
            command += 'basic2d,'

        if self.ui.basic2pButton.isChecked():
            command += 'basic2d_paral,'

        if self.ui.basic3Button.isChecked():
            command += 'basic3d,'

        if self.ui.basic3pButton.isChecked():
            command += 'basic3d_paral,'

        if self.ui.basicGaussButton.isChecked():
            command += 'gauss,'

        if self.ui.plot2dButton.isChecked():
            command += 'plot2d,'

        if self.ui.plot3dButton.isChecked():
            command += 'scatter3d,'

        command = command[:-1] + ' '

        command += '-c ' + '"' + self.ui.ColEdit.text() + '" '

        command += '-r ' + self.ui.RadiusEdit.text() + ' -i ' + self.ui.IterEdit.text()

        if self.ui.OutputEdit.text() != '':
            command += ' -o ' + self.ui.OutputEdit.text()

        return command


app = QtWidgets.QApplication([])
application = AveragerGUI()
application.show()

sys.exit(app.exec())
