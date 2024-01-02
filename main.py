import os
import cv2
import dlib
import sys
import numpy as np
from imutils import face_utils
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window
from kivy.config import Config

Config.set('graphics', 'resizable', True)
if getattr(sys, 'frozen', False):
   # Running as a bundled executable
     base_path = getattr(sys, '_MEIPASS', os.getcwd())
else:
     # Running as a script
     base_path = os.getcwd()
# Construct the relative path to the file
eyespy_windows_icon = os.path.join(base_path, 'eyespy_icon.png')

Config.set('kivy', 'window_icon',eyespy_windows_icon )
Config.write()


class MainApp(MDApp):

    def build(self):
   
        layout = RelativeLayout() 
        Window.size = (500, 750)
        if getattr(sys, 'frozen', False):
           # Running as a bundled executable
           base_path = getattr(sys, '_MEIPASS', os.getcwd())
        else:
           # Running as a script
           base_path = os.getcwd()

        self.title = "Eyespy"    
       
        # Construct the relative path to the file
        image_file_path=os.path.join( base_path,'drowsiness_onlineclass.png' )
       
        
        # Add a background image to the layout (replace 'background_image.jpg' with your image path)
        background_image = Image(source=image_file_path, size_hint=(1,1), allow_stretch=True,keep_ratio=False)
        layout.add_widget(background_image)

        # Add another image at a specific position (replace 'another_image.png' with your image path)
        self.image = Image(size_hint=(None, None), pos_hint={'center_x': 0.5, 'center_y': 0.85}, size=(650,350))
        layout.add_widget(self.image)

        # Add buttons at specific positions
        self.start_button = Button(text='Start Capture',size_hint=(None, None), size=(350, 50), pos_hint={'center_x': 0.3, 'center_y': 0.09})
        self.start_button.bind(on_press=self.start_capture)
        layout.add_widget(self.start_button)

        
         
        self.stop_button = Button(text='Stop Capture',size_hint=(None, None), size=(350, 50), pos_hint={'center_x': 0.7, 'center_y': 0.09})
        self.stop_button.bind(on_press=self.stop_capture)
        layout.add_widget(self.stop_button)

        

        Window.bind(on_resize=self.update_position)
        Window.bind(on_draw=self.update_position)

        

        self.is_music_playing = False
        #Minimum threshold of eye aspect ratio below which alarm is triggerd
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.3

        #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 30

        #Counts no. of consecutuve frames below threshold value
        self.COUNTER = 0

        print("Image file path %s",os.path.join( base_path,'haarcascade_frontalface_default.xml'))
        self.face_facade = cv2.CascadeClassifier(os.path.join(base_path,'haarcascade_frontalface_default.xml'))
        print("Image file path %s",os.path.join(base_path,'haarcascade_eye.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(base_path,'haarcascade_eye.xml'))

        #Load face detector and predictor using the dlib shape predictor file
        self.detector = dlib.get_frontal_face_detector()
        print("The base path is %s",base_path)
        print("Shape Detector Face Landmark dat file  %s",os.path.join(base_path,'shape_predictor_68_face_landmarks.dat'))
        self.predictor = dlib.shape_predictor(os.path.join(base_path,'shape_predictor_68_face_landmarks.dat'))

        #Extract indexes of facial landmarks for the left and right eye
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        #Start the Video Capture 
        self.capture = cv2.VideoCapture(0)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

       
        self.is_capture_started = False                
        Clock.schedule_interval(self.load_video, 1.0/30.0)
       
        return layout

   

    def update_position(self, *args):
        
        # Update the position of the image based on the window size and display resolution
        x_ratio = Window.width / Window.system_size[0]
        y_ratio = Window.height / Window.system_size[1]

        # Set the new position of the image
        self.image.pos = (x_ratio * (Window.system_size[0] / 2 - self.image.width / 60 ),y_ratio * (Window.system_size[1] / 2 - self.image.height /60))
        

    def start_capture(self, instance):
        self.is_capture_started = True
        self.start_button.disabled = False
        self.stop_button.disabled = False

    def stop_capture(self, instance):
        self.is_capture_started = False
        self.start_button.disabled = False
        self.stop_button.disabled = False

    def play_music(self, *args):
        # Get the directory of the script
        if getattr(sys, 'frozen', False):
           # Running as a bundled executable
           base_path = getattr(sys, '_MEIPASS', os.getcwd())
        else:
           # Running as a script
           base_path = os.getcwd()
        if (not self.is_music_playing):
            music = SoundLoader.load(os.path.join(base_path,'alarm.wav'))
            music.play()
            self.is_music_playing = True

    def load_video(self, *args):
        if self.is_capture_started:
            ret, frame = self.capture.read()
            # frame initialize
            self.image_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
            faces = self.detector(gray, 0)
            face_rectangle = self.face_facade.detectMultiScale(gray, 1.3, 5)

            #Draw rectangle around each face detected
            for (x,y,w,h) in face_rectangle:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            #Detect facial points
            for face in faces:

                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                #Get array of coordinates of leftEye and rightEye
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]

                #Calculate aspect ratio of both eyes
                leftEyeAspectRatio = self.eye_aspect_ratio(leftEye)
                rightEyeAspectRatio = self.eye_aspect_ratio(rightEye)

                eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

                #Use hull to remove convex contour discrepencies and draw eye shape around eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                #Detect if eye aspect ratio is less than threshold
                if(eyeAspectRatio < self.EYE_ASPECT_RATIO_THRESHOLD):
                    self.COUNTER += 1
                    #If no. of frames is greater than threshold frames,
                    if self.COUNTER >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                        self.play_music()
                        cv2.putText(frame, "Wake Up!!!!", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), 2)
                else:
                    self.is_music_playing = False
                    self.COUNTER = 0

            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

   
       


    #This function calculates and return eye aspect ratio
    def eye_aspect_ratio(self, eye):
       #Compute the Euclidean distances between the two sets of vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # Compute the Euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])

        # Calculate the eye aspect ratio (EAR)
        ear = (A + B) / (2.0 * C)

        return ear

if __name__ == '__main__':
    MainApp().run()
