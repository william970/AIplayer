from PyQt5.QtWidgets import QApplication
import win32gui
import sys
import os
import glob
app = QApplication(sys.argv)


# 图像帮助类，用于提取游戏截屏
class ImgHelper(object):
    def __init__(self, RootName):
        self.RootName = RootName
        if not os.path.exists(RootName):
            os.mkdir(RootName)
        self.fileList = glob.glob(os.path.join(RootName, '*.jpg'))

    def GetImg(self, windowsName, saveName):
        hwnd = win32gui.FindWindow(None, windowsName)
        screen = app.primaryScreen()
        img = screen.grabWindow(hwnd).toImage()
        img.save(saveName)
        return img

    def findLastImg(self):
        maxNumber = 0
        for i in range(len(self.fileList)):
            fileName = self.fileList[i]
            basefile = os.path.basename(fileName)
            number = int(os.path.splitext(basefile)[0])
            if number > maxNumber:
                maxNumber = number
        return maxNumber
