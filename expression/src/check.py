import cv2
from .model import Model
from PIL import Image
from torchvision import transforms
import torch

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
shape = (48,48)
classes = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprised',
    'Neutral'
]

net = Model(num_classes=len(classes))
checkpoint = torch.load('expression/trained/private_model_233_66.t7', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
net.eval()

def preprocess(image):
    transform_test = transforms.Compose([
        transforms.CenterCrop(shape[0]),
        transforms.ToTensor()
    ])
    
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        print('no face found')
        face = cv2.resize(image, shape)
    else:
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, shape)
    img = Image.fromarray(face).convert('L')
    inputs = transform_test(img)
    return inputs

def predict_emotion(image):

    tf_img = preprocess(image)
    tf_img = tf_img.unsqueeze(0)
    pred = net(tf_img)
    pred_class = classes[torch.argmax(pred)]
    return pred_class
    