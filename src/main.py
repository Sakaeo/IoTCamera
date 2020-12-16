# Run with
# python mqtt.py -u struj1 -p qN4qBZu<]QN2 -d dsiot -sd struj1

import argparse
import json

from camera import Camera
from mqtt_publisher import MqttPublisher
from mqtt_subscriber import MqttSubscriber

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--username", required=True,
                help="Mqtt Broker username")
ap.add_argument("-p", "--password", required=True,
                help="Mqtt Broker password")
ap.add_argument("-d", "--domain", required=True,
                help="Siot Topic domain")
ap.add_argument("-sd", "--subdomain", required=True,
                help="Siot Topic subdomain")
args = vars(ap.parse_args())

broker = "siot1.dsiot.ch"
port = 1883
username = args["username"]
password = args["password"]
domain = args["domain"]
subdomain = args["subdomain"]

publisher = MqttPublisher(domain, subdomain)
publisher.connect(broker, port, username, password)

camera = Camera(publisher)

subscriber = MqttSubscriber(domain, subdomain, camera)
subscriber.connect(broker, port, username, password)

subscriber.subscribe()
publisher.publish(json.dumps({"online": True}), "test/test/status")

camera.run_camera()
while True:
    pass
