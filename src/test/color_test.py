import cv2
import numpy as np

frame_hsv = None
FRAME_WIDTH = 920
FRAME_HEIGHT = 560


def nope(*arg):
    pass


# mouse callback function
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frame_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])

        cv2.setTrackbarPos('lowHue', 'colorTest', lower[0])
        cv2.setTrackbarPos('lowSat', 'colorTest', lower[1])
        cv2.setTrackbarPos('lowVal', 'colorTest', lower[2])
        cv2.setTrackbarPos('highHue', 'colorTest', upper[0])
        cv2.setTrackbarPos('highSat', 'colorTest', upper[1])
        cv2.setTrackbarPos('highVal', 'colorTest', upper[2])


def main():
    global frame_hsv

    cv2.namedWindow('colorTest')
    cv2.setMouseCallback('colorTest', pick_color)

    # Lower range colour sliders.
    cv2.createTrackbar('lowHue', 'colorTest', 0, 255, nope)
    cv2.createTrackbar('lowSat', 'colorTest', 0, 255, nope)
    cv2.createTrackbar('lowVal', 'colorTest', 0, 255, nope)
    # Higher range colour sliders.
    cv2.createTrackbar('highHue', 'colorTest', 0, 255, nope)
    cv2.createTrackbar('highSat', 'colorTest', 0, 255, nope)
    cv2.createTrackbar('highVal', 'colorTest', 0, 255, nope)
    # Contour min Area
    cv2.createTrackbar('minContourArea', 'colorTest', 20, 5000, nope)

    # vid_capture = cv2.VideoCapture(0)
    # vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    # vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame = cv2.imread("../img/in/Erkennung_01.png")

    while True:
        # FPS Count
        timer = cv2.getTickCount()

        # Get HSV values from the GUI sliders.
        low_hue = cv2.getTrackbarPos('lowHue', 'colorTest')
        low_sat = cv2.getTrackbarPos('lowSat', 'colorTest')
        low_val = cv2.getTrackbarPos('lowVal', 'colorTest')
        high_hue = cv2.getTrackbarPos('highHue', 'colorTest')
        high_sat = cv2.getTrackbarPos('highSat', 'colorTest')
        high_val = cv2.getTrackbarPos('highVal', 'colorTest')

        min_area = cv2.getTrackbarPos('minContourArea', 'colorTest')

        # Get webcam frame
        # ret, frame = vid_capture.read()

        # Convert the frame to HSV colour model.
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('frame_hsv', frame_hsv)

        color_low = np.array([low_hue, low_sat, low_val])
        color_high = np.array([high_hue, high_sat, high_val])
        mask = cv2.inRange(frame_hsv, color_low, color_high)
        dilated = cv2.dilate(mask, None, iterations=3)

        # Show the first mask
        cv2.imshow('mask-dilated', dilated)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rect_frame = np.copy(frame)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                print('Contour Area: ' + str(cv2.contourArea(contour)))
                # Original Contour
                # epsilon = 0.1 * cv2.arcLength(contour, True)
                # approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)

                # Bounding Rectangle
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(rect_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # cv2.putText(frame, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show final output image
        cv2.imshow('colorTest', frame)
        cv2.imshow('Rectangle Contours', rect_frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # Esc
            break

    cv2.destroyAllWindows()
    vid_capture.release()


if __name__ == '__main__':
    main()
