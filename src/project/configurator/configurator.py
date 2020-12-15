import base64

import cv2
import numpy as np
import PySimpleGUI as sg


def start():
    # All the stuff inside your window.
    layout = [
        [sg.Text("Siot-Domain")],
        [sg.InputText(key="domain", default_text="dsiot")],
        [sg.Text("Siot-Subdomain")],
        [sg.InputText(key="subdomain", default_text="struj1")],
        [sg.Text("Siot-Username")],
        [sg.InputText(key="username", default_text="struj1")],
        [sg.Text("Siot-Password")],
        [sg.InputText(key="password", password_char="*", default_text="qN4qBZu<]QN2")],
        [sg.Submit(), sg.Cancel()]
    ]

    # Create the Window
    window = sg.Window('Test', layout)
    # window.Maximize()
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, "Cancel"):  # if user closes window or clicks cancel
            break
        if event in "Submit":
            window.close()
            return values


class Configurator:
    def __init__(self):
        self.pictures = {}
        self.image = None
        self.config_window = None

    def build_image(self, msg):
        pic_id = msg["pic_id"]
        pos = msg["pos"]
        packet_number = msg["packet_number"]
        data = msg["data"]

        if pic_id not in self.pictures:
            self.pictures[pic_id] = {"count": 0, "total": packet_number, "pieces": {}, "pic_id": pic_id}
            self.pictures[pic_id]["pieces"][pos] = data[2:len(data) - 1]

        else:
            self.pictures[pic_id]["pieces"][pos] = data[2:len(data) - 1]
            self.pictures[pic_id]["count"] += 1

            self.config_window["txt"].update(
                "Receiving image: {}/{}".format(self.pictures[pic_id]["count"], self.pictures[pic_id]["total"]))

            if self.pictures[pic_id]["count"] == self.pictures[pic_id]["total"]:
                img = ""

                for i in range(self.pictures[pic_id]["total"] + 1):
                    img = img + self.pictures[pic_id]["pieces"][i]

                decode = base64.b64decode(img)
                jpg_np = np.frombuffer(decode, dtype=np.uint8)
                self.image = cv2.imdecode(jpg_np, 1)

    def send_config(self):
        pass

    def configure(self):
        layout = [
            [sg.Text(key="txt", text="Receiving image: {}/{}".format(0, 0), size=[100,20])]
        ]

        self.config_window = sg.Window('Waiting for image', layout)

        while True:
            event, values = self.config_window.read()
            if event in (None, "Cancel"):  # if user closes window or clicks cancel
                break
