import cv2
import pickle


# Create the haar cascade path and map to cascade classifier
cascPath = "../cascades/data/haarcascade_frontalface_default.xml"  # sys.argv[2]

face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eyes_cascade =

recognizer = cv2.face.LBPHFaceRecognizer_create()  #face recognizer module
recognizer.read("./recognizers/face-trainer.yml")  #face trained file
# create label dictionary from pickle labels
labels = {"person": 1}
with open("pickles/labels.pickle", "rb") as f:
    first_labels=pickle.load(f)
    labels = { v:k for k,v in first_labels.items()}  # we invert to use id_ as our call out value

cap = cv2.VideoCapture(0)
while True:
    # capture video frame
    ret, frame = cap.read()

    # create gray filter and map rectangle to faces
    gray=cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY )
    faces=face_cascade.detectMultiScale( gray , scaleFactor=4.4 , minNeighbors=5 )
    for (x , y , w , h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y + h , x:x + w]
        roi_color=frame[y:y + h , x:x + w]

        img_item='framedImages/myImg.png'
        img2_item='framedImages/myImg2.png'
        cv2.imwrite( img_item , roi_gray )
        cv2.imwrite( img2_item , roi_color )

        id_ , conf=recognizer.predict( roi_gray )
        if conf>=45 and conf<=85:
            # print(id_)
            # print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            font_size=1
            name=labels[id_]
            color=(255 , 255 , 255)
            stroke=2
            cv2.putText( frame , name , (x , y) , font , font_size , color , stroke , cv2.LINE_AA )

        color = (255,0,0)
        stroke=2  # rectangle thickness
        end_cord_x=x + w
        end_cord_y=y + h
        cv2.rectangle( frame , (x , y) , (end_cord_x , end_cord_y) , color , stroke )


        # display the frame
    cv2.imshow( 'frame' , frame )
    if cv2.waitKey( 20 ) & 0xFF == ord( 'q' ):
        break

