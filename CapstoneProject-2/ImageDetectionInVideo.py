import cv2
import dlib

# Load the pre-trained face detector provided by dlib
detector = dlib.get_frontal_face_detector()

face_dataset = {}
# Function to append an array to the value of a given key
def append_array(key, value):
    if key in face_dataset:
        face_dataset[key].append(value)
    else:
        face_dataset[key] = [value]

# Load the pre-trained face recognition model provided by dlib
face_recognizer = dlib.face_recognition_model_v1('"C:\\Users\\annan\Documents\\Study\\Konda\\AIDI Course\\CapstoneProjects\\CapstoneProject-2\\images\\dlib_face_recognition_resnet_model_v1.dat"')

# Load the face dataset (a list of face encodings and corresponding labels)
# You will need to populate this dataset with your own data
persons =['kondal']
for person in  persons:
    path= 'C:\\Users\\annan\\Documents\\Study\\AIDI Course\\CapstoneProjects\\CapstoneProject-2\\images\\' + person+'\\WIN_20230612_04_49_20_Pro.jpg'
    image = dlib.load_rgb_image(path)
    # Detect faces in the image
    faces = dlib.get_frontal_face_detector()(image)
    # Iterate over detected faces and compute face encodings
    for face in faces:
        face_encoding = face_recognizer.compute_face_descriptor(image, face)
        append_array(person,face_encoding)

print(face_dataset)

# Open the default camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the camera
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Get the face encodings for the current face
        face_encodings = face_recognizer.compute_face_descriptor(frame, face)

        # Compare face encodings with the known face dataset
        for label, encodings in face_dataset.items():
            for known_encoding in encodings:
                # Compare the current face encoding with the known encoding
                distance = dlib.distance(face_encodings, known_encoding)

                # Set a threshold for face recognition
                if distance < 0.6:
                    # Draw a rectangle around the face
                    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the recognized label
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()