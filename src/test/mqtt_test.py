import paho.mqtt.client as paho

broker = "siot1.dsiot.ch"
port = 1883
username = "struj1"
password = "qN4qBZu<]QN2"


def on_publish(client, userdata, result):  # create function for callback
    print("data published \n")
    pass


# create client object
client = paho.Client(None)
# assign function to callback
client.on_publish = on_publish
# set username and password
client.username_pw_set(username, password)
# establish connection
client.connect(broker, port)

# publish
ret = client.publish("s2/dsiot/struj1/test/test/1", '{"msg:test2"}')

import time
import paho.mqtt.client as paho

broker = "broker.hivemq.com"
broker = "iot.eclipse.org"


# define callback
def on_message(client, userdata, message):
    time.sleep(1)
    print("received message =", str(message.payload.decode("utf-8")))


client = paho.Client(
    "client-001")  # create client object client1.on_publish = on_publish #assign function to callback client1.connect(broker,port) #establish connection client1.publish("house/bulb1","on")
######Bind function to callback
client.on_message = on_message
#####
print("connecting to broker ", broker)
client.connect(broker)  # connect
client.loop_start()  # start loop to process received messages
print("subscribing ")
client.subscribe("house/bulb1")  # subscribe
time.sleep(2)
print("publishing ")
client.publish("house/bulb1", "on")  # publish
time.sleep(4)
client.disconnect()  # disconnect
client.loop_stop()  # stop loop
