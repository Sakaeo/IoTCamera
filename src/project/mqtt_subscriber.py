import json

import paho.mqtt.client as paho

global camera


class MqttSubscriber:
    def __init__(self, domain, subdomain, cam):
        global camera
        self.domain = domain
        self.subdomain = subdomain
        self.client = None
        camera = cam

    def connect(self, broker, port, username, password):
        self.client = paho.Client()
        # assign function to callback
        self.client.on_subscribe = on_subscribe
        self.client.on_message = on_message
        # set username and password
        self.client.username_pw_set(username, password)
        # establish connection
        self.client.connect(broker, port)
        self.client.loop_start()

    def subscribe(self):
        self.client.subscribe("s2/{}/{}/+/+/config".format(self.domain, self.subdomain))
        self.client.subscribe("s2/{}/{}/+/+/snapshot_req".format(self.domain, self.subdomain))


def on_subscribe(client, userdata, mid, granted_qos):
    print("subscibed\n")
    pass


def on_message(client, userdata, message):
    topic = message.topic.split("/")
    topic_handler = topic[-3]
    topic_class = topic[-2]
    topic_id = topic[-1]

    msg: dict = json.loads(str(message.payload.decode("utf-8")))

    if topic_id == "config":
        write_config(msg)
        camera.reload_config()
    elif topic_id == "snapshot_req":
        if "request" in msg:
            if msg["request"]:
                camera.snapshot()


def write_config(message):
    with open("config.json", "w") as outfile:
        json.dump(message, outfile)
    print("saved config")
