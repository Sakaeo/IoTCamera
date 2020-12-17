# Run with
# python main.py -u struj1 -p qN4qBZu<]QN2 -d dsiot -sd struj1

import argparse

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
username = args["username"]
password = args["password"]
domain = args["domain"]
subdomain = args["subdomain"]
port = 1883
skip_frame = 30
min_confidence = 0.4
resolution = (320, 240)
debug = False

publisher = MqttPublisher(domain, subdomain)
publisher.connect(broker, port, username, password)

camera = Camera(publisher, skip_frame, min_confidence, resolution, debug)

subscriber = MqttSubscriber(domain, subdomain, camera)
subscriber.connect(broker, port, username, password)

subscriber.subscribe()

camera.run_camera()
while True:
    pass
