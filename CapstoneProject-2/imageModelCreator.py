import dlib

# Load the pre-trained face recognition model provided by dlib
face_recognizer = dlib.face_recognition_model_v1('C:\Users\annan\Documents\Study\AIDI Course\CapstoneProjects\CapstoneProject-2\images\dlib_face_recognition_resnet_model_v1.dat')

# Load an example training image
image = dlib.load_rgb_image('c:\Users\annan\Documents\Study\AIDI Course\CapstoneProjects\CapstoneProject-2\images\kondal\WIN_20230612_04_49_20_Pro.jpg')

# Detect faces in the image
faces = dlib.get_frontal_face_detector()(image)

# Iterate over detected faces and compute face encodings
face_encodings = []
for face in faces:
    face_encoding = face_recognizer.compute_face_descriptor(image, face)
    face_encodings.append(face_encoding)
print(face_encoding)