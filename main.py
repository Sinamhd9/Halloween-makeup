
''' 

A joker holloween makeup that builds from the user camera input.

Requires face recognition to detect the face landmarks, OpenCV for frame manipulation,
and PIL to draw shapes.

If the user smile, the eyes will turn into a bold random color.

'''

# Importing the required libraries
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
from scipy.spatial.distance import euclidean

def smile(lip, sc):
    '''
    detects a smile based on the ratio of 
    euclidean distance between two points
    in the lip and the scale(sc) which is the euclidean distance of two points on chin 
    returns True if smile is detected
    '''
    smile = False
    a = lip[0]
    b = lip[6]
    ratio = euclidean(a,b)/sc
    if ratio > 0.45:
        smile = True
    return smile

def mouth(landmarks, sc):
    '''
    returns the coordinate required to
    draw the joker's mouth,
    sc is scale based on the euclidean distance of two points on chin 
    '''
    p = ([(x-4.5*sc,y-4*sc) for (x,y) in [landmarks['bottom_lip'][7]]]+
        [(x-sc,y+sc) for (x,y) in [landmarks['bottom_lip'][7]]]+
        [(x+sc,y+2.5*sc) for (x,y) in [landmarks['bottom_lip'][7]]]+
        [(x-sc,y+2.5*sc) for (x,y) in [landmarks['bottom_lip'][0]]]+
        [(x+sc,y+sc) for (x,y) in [landmarks['bottom_lip'][0]]]+
        [(x+3.5*sc,y-2*sc) for (x,y) in [landmarks['bottom_lip'][0]]]+
        [(x,y-sc) for (x,y) in [landmarks['top_lip'][7]]]+
        [(x,y-sc) for (x,y) in [landmarks['top_lip'][0]]])  
    return p

def face(landmarks):
    '''
    returns the approximate coordinates 
    for the shape of joker's face
    '''
    p = (landmarks['chin']+
          [(x,2*landmarks['chin'][-2][1]-y) for (x,y) in landmarks['chin'][::-1]])
    return p;
    
def eye(eye, sc):
    
    '''
    returns the cooridnates for the shape under the joker's eye
    sc is scale based on the euclidean distance of two points on chin 
    '''
    p = ([eye[0]] + 
         [(x,y+sc) for (x,y) in [eye[5]]]+
         [(x,y+sc*2) for (x,y) in [eye[1]]]+
         [eye[3]])
    return p

def eyebrow(eyebrow, sc):
    '''
    returns the cooridnates for the eyebrow shape
    sc is scale based on the euclidean distance of two points on chin
    '''
    p = ([eyebrow[0]] + 
         [(x,y-2*sc) for (x,y) in [eyebrow[2]]]+
         [eyebrow[4]])
    return p


def second_eyebrow(eyebrow, sc):
    '''
    returns the cooridnates for the joker's second upper eyebrow shape
    sc is scale based on the euclidean distance of two points on chin
    '''
    p = ([(x,y-4*sc) for (x,y) in eyebrow])
    return p


def nose(landmarks):  
    '''
    returns the cooridnates for the joker's nose
    '''
    p = (landmarks['nose_tip'] + 
        [landmarks['nose_bridge'][2]])
    return p


def main():
    
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)
    while(True):
        ret, frame = cap.read()
        
        # Detect face landmarks
        face_landmarks_list = face_recognition.face_landmarks(frame)
        
        # For the landmarks detected in the face
        for face_landmarks in face_landmarks_list:
            
            # A global scale based on the width of the chin
            scale = euclidean(face_landmarks['chin'][0], face_landmarks['chin'][-1])
            pil_image = Image.fromarray(cv2. cvtColor(frame, cv2. COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil_image, 'RGBA')
            
            # draw top eyebrows
            d.polygon(second_eyebrow(face_landmarks['left_eyebrow'], scale/20), fill=(130, 0, 0, 200))
            d.polygon(second_eyebrow(face_landmarks['right_eyebrow'], scale/20), fill=(130, 0, 0, 200))
            
            # eye glaze
            d.polygon(face_landmarks['left_eye'], fill=(255, 0, 0, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 0, 0, 30))
            
            # draw mouth
            d.polygon(mouth(face_landmarks, scale/20), fill=(130, 0, 0, 200))
            
            # whiten face
            d.polygon(face(face_landmarks), fill=(255, 255, 255, 50))
            
            # draw eye concealer
            d.polygon(eye(face_landmarks['left_eye'], scale/10), fill=(0, 0, 0, 180))
            d.polygon(eye(face_landmarks['right_eye'], scale/8), fill=(0, 0, 0, 180))
            
            # draw eyebrows
            d.polygon(eyebrow(face_landmarks['left_eyebrow'], scale/30), fill=(0, 0, 0, 180))
            d.polygon(eyebrow(face_landmarks['right_eyebrow'], scale/20), fill=(0, 0, 0, 180))
            
            # draw nose
            d.polygon(nose(face_landmarks), fill=(130, 0, 0, 180))
            
            # draw eyeliner
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 200), width=5)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 200), width=5)
            
            # Change eye to a random color if smiling
            if smile(face_landmarks['bottom_lip'], scale):
                random_color = tuple(np.append(np.round(np.random.uniform(low=0, high=255, size=(3,))).astype(int), 200))
                d.polygon(face_landmarks['left_eye'], fill=random_color)
                d.polygon(face_landmarks['right_eye'], fill=random_color)
            
            frame = cv2. cvtColor(np.array(pil_image), cv2. COLOR_RGB2BGR)
            
        # Image enhancement
        frame = cv2.detailEnhance(frame, sigma_s=20, sigma_r=0.2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()




