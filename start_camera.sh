#!/bin/bash
# startup script

FLAG=~/Desktop/touch_me.txt

USERNAME='struj1'
PASSWORD='qN4qBZu<]QN2'
DOMAIN='dsiot'
SUBDOMAIN='struj1'
SKIPFRAME=30
MINCONFIDENCE=0.4
RESOLUTION=320,240

touch $FLAG

cd ~/IoTCamera/src/
python3 main.py -u "$USERNAME" -p "$PASSWORD" -d "$DOMAIN" -sd "$SUBDOMAIN" -sf $SKIPFRAME -mc $MINCONFIDENCE -r $RESOLUTION