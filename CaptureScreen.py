import datetime

import pyautogui as pag


def capture():

    timeFormat = '%y%m%d%H%M%S'

    take = pag.screenshot()
    take.save(r"Images/ss{}.png".format(datetime.datetime.now().strftime(timeFormat)))


