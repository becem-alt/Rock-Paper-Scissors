import numpy as np
import cv2
import os
import sys


Type='train'

try:
    Type=sys.argv[3]
except:
    print("Training.")
IMG_SAVE_PATH=Type

assert  IMG_SAVE_PATH in ('train','valid')
try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    exit(-1)

IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0
nb=0
speed=10
startpoint=(150,150)
endpoint=(374,374)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break
    cv2.rectangle(frame, startpoint,endpoint, (255, 255, 255), 2)

    if start and nb%speed==0:
        roi = frame[150:374, 150:374]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1
    nb+=1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
