import os
import time

import _thread
from pynput import keyboard

from utils.ImgHelper import ImgHelper

currentRoot = os.getcwd()
ProjectRoot = os.path.abspath(os.path.join(currentRoot, '..'))
ImgRoot = os.path.join(ProjectRoot, 'dataSet')
configPath = os.path.join(ProjectRoot, 'config')

# 动作序列
Keylist = {}
Allkey = []
Finish = False


def initKeyList(keylistpath):
    with open(keylistpath, 'r', encoding="utf-8") as keylistfile:
        for keyline in keylistfile.readlines():
            keyline = keyline.rstrip('\n')
            if len(keyline) == 1:
                Allkey.append(keyline)
                Keylist[keyline] = 0


def on_press(key):
    try:
        print('key {0} pressed'.format(key.char))
        Keylist[key.char] = 1
        print(Keylist)
    except AttributeError:
        print('special key {0} pressed'.format(key))


def on_release(key):
    if key == keyboard.Key.esc:
        global Finish
        Finish = True
        return False
    try:
        print('key {0} released'.format(key.char))
        Keylist[key.char] = 0
    except AttributeError:
        print('special key {0} released'.format(key))


class DataSetCollector(object):
    def __init__(self, imgRoot, windowName):
        self.imgHelper = ImgHelper(imgRoot)
        self.currentNumber = self.imgHelper.findLastImg()
        self.windowName = windowName
        self.keyboardListener = keyboard.Listener(on_press=on_press, on_release=on_release)
        print(self.currentNumber)

    def GetImg(self):
        print(self.currentNumber)
        self.currentNumber = self.currentNumber + 1
        saveImgName = os.path.join(ImgRoot, str(self.currentNumber) + ".jpg")
        self.imgHelper.GetImg(self.windowName, saveImgName)
        return saveImgName

    def ListenKeyboard(self, delay):
        while True:
            time.sleep(delay)
            saveImgName = self.GetImg()
            with open(os.path.join(ImgRoot, 'train.txt'), 'a') as trainfile:
                trainfile.write(saveImgName)

                for key in Allkey:
                    trainfile.write(" " + str(Keylist[key]))
                trainfile.write("\n")

    def StartCollect(self):
        self.keyboardListener.start()
        # 开启键盘监听线程
        _thread.start_new_thread(self.ListenKeyboard, tuple([1]))
        global Finish
        while True:
            # print(Finish)
            if Finish:
                break
        # 停止监听
        self.keyboardListener.stop()


initKeyList(os.path.join(configPath, 'allKey.txt'))
print(Keylist)
collector = DataSetCollector(ImgRoot, '雷电模拟器')
collector.StartCollect()
