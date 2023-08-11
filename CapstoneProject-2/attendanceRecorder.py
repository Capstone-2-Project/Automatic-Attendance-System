import cv2
import imutils
import numpy as np
import argparse
import pickle
import dlib
import cv2
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import tkinter as tk
from tkinter import messagebox
import smtplib
from email.mime.text import MIMEText
from turtle import textinput

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("C:\\Users\\annan\\Documents\\Study\\Konda\\AIDI Course\\CapstoneProjects\\CapstoneProject-2\\images\\shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("C:\\Users\\annan\Documents\\Study\\Konda\\AIDI Course\\CapstoneProjects\\CapstoneProject-2\\images\\dlib_face_recognition_resnet_model_v1.dat")
roll_numbers=[1,2,3,4,5,6]

roll_number_to_name={1:'Yedukondalu',2:'Reddi',3:'Nandini',4:'Likith',5:'Manvitha',6:'Ruchita'}


# root = tk.Tk()
# root.title("Attendance Recorder")
# root.attributes("-fullscreen", True)

def face_rects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    rects = face_detector(gray, 1)
    return rects
def face_landmarks(image):
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]
def face_encodings(image):
    # compute the facial embeddings for each face 
    # in the input image. the `compute_face_descriptor` 
    # function returns a 128-d vector that describes the face in an image
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark)) 
            for face_landmark in face_landmarks(image)]


def nb_of_matches(known_encodings, unknown_encoding):
    # compute the Euclidean distance between the current face encoding 
    # and all the face encodings in the database
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.6
    # return the number of matches
    return sum(small_distances)

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)
    return frame
# def humanDetector(name_encodings_dict):
#     writer = None    
#     # Create a button to show the popup
#     popup_button_1 = tk.Button(root, text="Camera(For Students)", command=detectByCamera(writer,name_encodings_dict))
#     popup_button_2 = tk.Button(root, text="Screenshare(For Lecturer)", command=detectByScreenShare(writer,name_encodings_dict))

#     popup_button_1.pack(pady=20)
#     popup_button_2.pack(pady=20)
#     root.mainloop()
#     # return detectByCamera(writer,name_encodings_dict)
#     # return detectByScreenShare(writer,name_encodings_dict)
    
def detectByCamera(writer,name_encodings_dict):  
    video = cv2.VideoCapture(0)
    print('Opening WebCam')
    while True:
        check, frame = video.read()
        # frame = detect(frame)
        encodings = face_encodings(frame)
        names=[]
        for encoding in encodings:
            counts = {}
            for (roll_number, encodings) in name_encodings_dict.items():
                counts[roll_number] = nb_of_matches(encodings, encoding)
            if all(count == 0 for count in counts.values()):
                roll_number = 0
            else:
                roll_number = max(counts, key=counts.get)
                roll_numbers.append(roll_number)
                attendance.add(roll_number)
        for rect, roll_number in zip(face_rects(frame), roll_numbers):
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, roll_number_to_name.get(roll_number) , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows() 
    
def detectByScreenShare(writer,name_encodings_dict):  
    print('Sharing the screen')
    while True:
        screenshot = ImageGrab.grab()
        # Convert the screenshot to an OpenCV image
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        encodings = face_encodings(frame)
        for encoding in encodings:
            counts = {}
            for (roll_number, encodings) in name_encodings_dict.items():
                counts[roll_number] = nb_of_matches(encodings, encoding)
            if all(count == 0 for count in counts.values()):
                roll_number = 0
            else:
                roll_number = max(counts, key=counts.get)
                roll_numbers.append(roll_number)
                attendance.add(roll_number)
        for rect, roll_number in zip(face_rects(frame), roll_numbers):
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, roll_number_to_name.get(roll_number) , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows() 
def send_email(subject, body, sender, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    recipients= textinput("MailId", "Please enter your recievers MailId:")
    msg['To'] =recipients
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")
    # root.destroy()
    # return attendance;

with open("C:\\Users\\annan\\Documents\\Study\\Konda\\AIDI Course\\CapstoneProjects\\CapstoneProject-2\\pickleFile\\encodings.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)

writer=None
root = tk.Tk()
root.title("Attandance Register")
# Maximize the main window
root.attributes("-fullscreen", True)
attendance={1,4}
# Create a button to show the popup
popup_button_1 = tk.Button(root, text="Camera(For Students)", command= lambda:detectByCamera(writer,name_encodings_dict))
popup_button_2 = tk.Button(root, text="Screenshare(For Lecturer)", command= lambda:detectByScreenShare(writer,name_encodings_dict))
subject="Attendance For the Class"
body="list of studentsIDs who attended the class is "+str(attendance)
popup_button_3 = tk.Button(root, text="SendMail", command= lambda:send_email(subject, body, "yedukondaluannangi@gmail.com", "lumeccvgsgbeybau"))
popup_button_1.pack(pady=20)
popup_button_2.pack(pady=20)
popup_button_3.pack(pady=20)

# Start the main GUI event loop
root.mainloop()


