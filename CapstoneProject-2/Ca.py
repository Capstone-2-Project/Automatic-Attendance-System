import cv2
import dlib

# Load the pre-trained face detector provided by dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained face recognition model provided by dlib
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model.dat')

# Load the face dataset (a list of face encodings and corresponding labels)
# You will need to populate this dataset with your own data
face_dataset = {
    'person_1': [face_encoding_1, face_encoding_2, ...],
    'person_2': [face_encoding_1, face_encoding_2, ...],
    # Add more entries for each person in your dataset
}

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
cv2.destroyAllWindows(