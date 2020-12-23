import cv2
import numpy as np
from random import choice
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import requests
import json
from tensorflow.keras.preprocessing.image import load_img

MODEL_URL="http://localhost:3001/v1/models/model:predict"
DICT={  0:'none', 1:'paper',2:'rock', 3:'scissors'}
def get_predictions(image):
    data=json.dumps({
        'instances':image.tolist()
    })
    res=requests.post(MODEL_URL,data=data.encode())
    predict=json.loads(res.text)
    Class=DICT[np.argmax(predict['predictions'][0])]
    return str(Class)


startpoint=(150,150)
endpoint=(374,374)
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,980)
prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, startpoint, endpoint, (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 150), (1024, 374), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[150:374, 150:374]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img=preprocess_input(img)
    img=np.expand_dims(img,axis=0)
    # predict the move made
    user_move_name = get_predictions(img)
    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            print(computer_move_name)
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":

        icon=load_img(
                "images/{}.png".format(computer_move_name),
                target_size=(224,224)
            )
        frame[150:374, 800:1024] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
