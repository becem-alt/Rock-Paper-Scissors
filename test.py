import numpy as np
import cv2
import requests
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
"""
Don't forget to start the docker image ( tensorflow/serving ) .

docker run -p3001:8501  --name rockpaperscissors -v ${PWD}\\model:/models/model/1 -e MODEL_NAME=model tensorflow/serving

"""

MODEL_URL="http://localhost:3001/v1/models/model:predict"
DICT={  0:'none', 1:'paper',2:'rock', 3:'scissors'}

def get_predictions(image):
    data=json.dumps({
        'instances':image.tolist()
    })
    res=requests.post(MODEL_URL,data=data.encode())
    predict=json.loads(res.text)
    print(predict['predictions'])
    Class=DICT[np.argmax(predict['predictions'][0])]
    return str(Class)

    """
def process_image(path):
    image=tf.keras.preprocessing.image.load_img(
        path,
        target_size=(224,224,3)
    )
    image=tf.keras.preprocessing.image.img_to_array(image)
    image=tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image=np.expand_dims(image,axis=0)
    return image
print(get_predictions(process_image('1.jpg')))
"""

cap = cv2.VideoCapture(0)
startpoint=(150,150)
endpoint=(374,374)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, startpoint,endpoint, (255, 255, 255), 2)
    image = frame[150:374, 150:374]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=preprocess_input(image)
    image=np.expand_dims(image,axis=0)
    c =get_predictions(image)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, " {}".format(c),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
