import cv2

while 1:
    print("wait for Keypress")

    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break
    elif k == -1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k)  # else print its value
