import paho.mqtt.client as paho


class MqttPublisher:
    def __init__(self, domain, subdomain):
        self.domain = domain
        self.subdomain = subdomain
        self.client = None

    def connect(self, broker, port, username, password):
        self.client = paho.Client()
        # assign function to callback
        self.client.on_publish = on_publish
        # set username and password
        self.client.username_pw_set(username, password)
        # establish connection
        self.client.connect(broker, port)

    def publish(self, msg, topic):
        self.client.publish("s2/{}/{}/{}".format(self.domain, self.subdomain, topic), msg)


def on_publish(ret, userdata, result):  # create function for callback
    pass
