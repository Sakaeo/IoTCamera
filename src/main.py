# Run with
# python main.py -u struj1 -p qN4qBZu<]QN2 -d dsiot -sd struj1

# for Raspberry
# python main.py -u struj1 -p qN4qBZu<]QN2 -d dsiot -sd struj1 -sf 30 -mc 0.4 -r 320,240

import argparse
import time

from camera import Camera
from mqtt_publisher import MqttPublisher
from mqtt_subscriber import MqttSubscriber

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--username", required=True,
                help="Siot Mqtt Broker username")
ap.add_argument("-p", "--password", required=True,
                help="Siot Mqtt Broker password")
ap.add_argument("-d", "--domain", required=True,
                help="Siot Topic domain")
ap.add_argument("-sd", "--subdomain", required=True,
                help="Siot Topic subdomain")
ap.add_argument("-sf", "--skip_frame",
                help="How many frames are skipped before a new object recognition")
ap.add_argument("-mc", "--min_confidence",
                help="Min Confidence an object needs to have to be classified (between 0 and 1)")
ap.add_argument("-r", "--resolution",
                help="Resolution of the image, lower is faster (ex: with,height)")
ap.add_argument("-deb", "--debug",
                help="Ture/False if the image should be shown or not")
args = vars(ap.parse_args())

broker = "siot1.dsiot.ch"
port = 1883

username = args.pop("username")
password = args.pop("password")
domain = args.pop("domain")
subdomain = args.pop("subdomain")

publisher = MqttPublisher(domain, subdomain)
publisher.connect(broker, port, username, password)

camera = Camera(publisher, args)

subscriber = MqttSubscriber(domain, subdomain, camera)
subscriber.connect(broker, port, username, password)

subscriber.subscribe()

stopped = False
error = False

while True:
    if stopped:
        print("Camera stopped, trying to restart")
        time.sleep(2)
    if error:
        print("error")
        break
    stopped, error = camera.run_camera()
