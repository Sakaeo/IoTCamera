# Run with
# python main.py

import json

from project.configurator.configurator import Configurator, start
from project.configurator.mqtt_publisher import MqttPublisher
from project.configurator.mqtt_subscriber import MqttSubscriber

ret, values = start()

if not ret:
    exit(0)

broker = "siot1.dsiot.ch"
port = 1883
username = values["username"]
password = values["password"]
domain = values["domain"]
subdomain = values["subdomain"]

publisher = MqttPublisher(domain, subdomain)
publisher.connect(broker, port, username, password)

configurator = Configurator(publisher)

subscriber = MqttSubscriber(domain, subdomain, configurator)
subscriber.connect(broker, port, username, password)

subscriber.subscribe()

publisher.publish(json.dumps({"request": True}), "test/test/snapshot_req")

ret, image = configurator.wait_for_image()

if ret:
    subscriber.exit()

configurator.configure(image)
