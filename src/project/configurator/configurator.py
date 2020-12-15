import base64
import copy

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
    sg.theme("Dark Brown")
    window = sg.Window('Test', layout)
    # window.Maximize()
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, "Cancel"):  # if user closes window or clicks cancel
            return False, None
        if event in "Submit":
            window.close()
            return True, values


def draw_rois(frame, boxes):
    for name, box in boxes.items():
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)

        x = int(x + w / 2)
        y = int(y + h / 2)
        text = "{}".format(name)
        cv2.putText(frame, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


class Configurator:
    def __init__(self):
        self.pictures = {}
        self.image = None
        self.waiting_window = None

        self.image = None

        sg.theme("Dark Brown")

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

            self.waiting_window["txt"].update(
                "Receiving image: {}/{}".format(self.pictures[pic_id]["count"], self.pictures[pic_id]["total"]))

            if self.pictures[pic_id]["count"] == self.pictures[pic_id]["total"]:
                img = ""

                for i in range(self.pictures[pic_id]["total"] + 1):
                    img = img + self.pictures[pic_id]["pieces"][i]

                decode = base64.b64decode(img)
                jpg_np = np.frombuffer(decode, dtype=np.uint8)
                self.image = cv2.imdecode(jpg_np, 1)
                self.waiting_window["ok_button"].update(disabled=False)

    def send_config(self):
        pass

    def configure(self, image):
        img_bytes = cv2.imencode(".png", image)[1].tobytes()
        layout = [
            [sg.Image(data=img_bytes, key="img")],
            [sg.Button(button_text="Add ROI", key="add_roi")],
            [sg.Text("ROI's", key="list_of_rois")]
        ]

        window = sg.Window('Configurator', layout)
        ROIs = {}
        while True:
            event, values = window.read()
            if event is None:
                break
            if event in "add_roi":
                roi = cv2.selectROI("ROI select", image, False)
                cv2.destroyWindow("ROI select")
                while True:
                    name = sg.popup_get_text("name")
                    if name in ROIs.keys():
                        sg.popup("Name already used")
                    else:
                        break

                ROIs[name] = roi
                window.extend_layout(window,
                                     [[sg.Text("{}: {} ".format(name, roi), key="roi_{}".format(name)),
                                       sg.Button(button_text="-", key="remove_roi_{}".format(name))]])
            if "remove_roi" in event:
                name = event.split("_")[2]
                window["roi_{}".format(name)].update(visible=False)
                window["roi_{}".format(name)].update(size=[0, 0])
                window["remove_roi_{}".format(name)].update(visible=False)

                ROIs.pop(name)

            img_copy = copy.copy(image)
            draw_rois(img_copy, ROIs)
            img_bytes = cv2.imencode(".png", img_copy)[1].tobytes()
            window["img"].update(data=img_bytes)

    def wait_for_image(self):
        layout = [
            [sg.Text(key="txt", text="Receiving image: {}/{}".format(0, 0), size=[20, 0])],
            [sg.OK(disabled=True, key="ok_button")]
        ]

        self.waiting_window = sg.Window('Waiting for image', layout)

        while True:
            event, values = self.waiting_window.read()
            if event in (None, "ok_button"):  # if user closes window or clicks cancel
                self.waiting_window.Close()
                return True, self.image
