# Import de librerias
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


emotionModel = load_model("modelTuning.keras")


def predict_emotion(frame, faceNet, emotionModel):

    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()


    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0
            
            face = frame[Yi:Yf, Xi:Xf]

            
            if face.size == 0:
                continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (299, 299))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

@csrf_exempt
def emotion_detection(request):
    if request.method == 'POST':
        try:
      
            data = request.POST.get('image')
            if not data:
                return JsonResponse({'status': 'error', 'message': 'No image data provided'})
            
   
            if data.startswith('data:image'):
                data = data.split(',')[1]

        
            image_data = base64.b64decode(data)
    
            image = Image.open(BytesIO(image_data))
            
    
            if image is None:
                return JsonResponse({'status': 'error', 'message': 'Unable to decode image'})


            image = np.array(image)

           
            if image.size == 0:
                return JsonResponse({'status': 'error', 'message': 'Decoded image is empty'})


            if len(image.shape) == 2: 
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4: 
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

     
            (locs, preds) = predict_emotion(image, faceNet, emotionModel)
            
   
            response = []
            for (box, pred) in zip(locs, preds):
                label = classes[np.argmax(pred)]
                accuracy = np.max(pred) * 100
                response.append({
                    'emotion': label,
                    'accuracy': accuracy,
                    'box': {
                        'Xi': int(box[0].item()),  
                        'Yi': int(box[1].item()),  
                        'Xf': int(box[2].item()), 
                        'Yf': int(box[3].item())   
                    }
                })

            return JsonResponse({'status': 'success', 'predictions': response})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
