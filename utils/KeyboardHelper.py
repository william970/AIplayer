from pynput.keyboard import Listener
from pynput import keyboard

Keylist = {}
allKey = []


def on_release(key):
    if key == keyboard.Key.esc:
        with open('allKey.txt', 'a') as allKeyfile:
            for i in range(len(allKey)):
                allKeyfile.write(allKey[i] + "\n")
        return False
    try:
        print('key {0} released'.format(key.char))
        Keylist[key.char] = 0
    except AttributeError:
        print('special key {0} released'.format(key))


def on_press(key):
    try:
        print('key {0} pressed'.format(key.char))
        allKey.append(key.char)
        Keylist[key.char] = 1
        print(Keylist)
    except AttributeError:
        print('special key {0} pressed'.format(key))


class KeyboardHelper(object):
    def __init__(self):
        pass

# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()
