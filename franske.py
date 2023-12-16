# FRANSJE code
# Tom Bergman TU/e M1-2 Artifice project

import os
from cleantext import clean
import sys
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QUrl, Qt
import wave
import threading
import time
import random
from pynput import keyboard
import numpy as np
import googlemaps
import pandas as pd
import json
import time
import pyaudio
from  scipy.io import wavfile 
import wave
from playsound import playsound
from openai import OpenAI
from typing import Sequence
import google.cloud.texttospeech as tts
from pyowm import OWM
from deepface import DeepFace


#Settings --------------------------
enable_memory = True
enable_emotion = True
enable_weather = True
enable_pois = True
enable_timedate = True
enable_user_characteristics = True
use_true_location = True


#Variables -------------------------

#Google
# Set location of your google api credentials here
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=".filename.json"

#Audio
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"
p = pyaudio.PyAudio()
frames = []
empty_frames = []
recording = False

# OpenAI api
API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=API_KEY)

#Emotion recognition
last_emotion = ""

#Weather module
current_w_temp = 0
current_w_stat = ""

#Google maps module
maps_client = googlemaps.Client('YOUR_GOOGLE_MAPS_API_KEY')
adress = "Steenstraat 13, Eersel"
points_of_interest = ""

#Messages
conversation_start = [{"role": "system", "content": "Je bent een grappige, eigenzinnige vriend genaamd Fransje met cynische en sarcastische humor. Je hebt je eigen, creatieve gedachtes en mening. Je krijgt achtergrondinformatie over de situatie en mag deze soms in je antwoorden gebruiken. Gebruik deze alleen niet te letterlijk. Reageer enkel op wat de persoon zegt met een korte en grappige reactie. Het is toegestaan om iemand belachelijk te maken inclusief de gebruiker zelf. \n Begin het gesprek met iets raars wat je die dag hebt meegemaakt of gezien. Hierin mag je de achtergrondinformatie gebruiken."}]
message_history = [{"role": "system", "content": "Je bent een grappige, eigenzinnige vriend genaamd Fransje met cynische en sarcastische humor. Je hebt je eigen, creatieve gedachtes en mening. Je krijgt achtergrondinformatie over de situatie en mag deze soms in je antwoorden gebruiken. Gebruik deze alleen niet te letterlijk. Reageer enkel op wat de persoon zegt met een korte en grappige reactie. Het is toegestaan om iemand belachelijk te maken inclusief de gebruiker zelf. \n Begin het gesprek met iets raars wat je die dag hebt meegemaakt of gezien. Hierin mag je de achtergrondinformatie gebruiken."}]
new_message = {}

#GUI
animateTalking = False
animateIdle = True

#End of variables -------------------







# BEGIN CLASSES AND FUNCTIONS
# --------------------------------------------------------------------------------------------------------------------------------------------------

#Class that creates the GUI
class FransjeGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(0, 0, 1920, 1080)
        self.setScreen(app.screens()[1])

        screen = self.screen()

        self.move(screen.geometry().topLeft())

        # self.media_player = QMediaPlayer()
        # self.video_widget = QVideoWidget()

        self.neutralface = QPixmap("neutral.png")
        self.talkingface = QPixmap("talk.png")
        self.blinkface = QPixmap("blink.png")
        self.face = QLabel()
        self.face.setPixmap(self.neutralface)

        # self.media_player.setVideoOutput(self.video_widget)
        # self.media_player.setSource(QUrl.fromLocalFile("noise.mp4"))
        # self.media_player.setLoops(10000)
        # self.media_player.positionChanged.connect(self.position_changed)
        # self.media_player.durationChanged.connect(self.duration_changed)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        # layout.addWidget(self.video_widget)
        layout.addWidget(self.face)
        

        container = QWidget()
        self.setStyleSheet("QMainWindow { background-color: transparent; }")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        container.setLayout(layout)
        self.setCentralWidget(container)

        
        self.showFullScreen()
        # self.media_player.play()

    def openMouth(self):
        self.face.setPixmap(self.talkingface)

    def closeMouth(self):
        self.face.setPixmap(self.neutralface)

    def blinkEyes(self):
        self.face.setPixmap(self.blinkface)


#Start talking animation in UI
def startTalkAnimation():
    global animateTalking
    while True:
        if animateTalking == True:
            fransje_gui.openMouth()
            time.sleep(random.randint(1, 4) / 10)
            fransje_gui.closeMouth()
            time.sleep(random.randint(2, 4) / 10)


#Start talking animation in UI
def startIdleAnimation():
    global animateIdle
    while True:
        if animateIdle == True:
            fransje_gui.blinkEyes()
            time.sleep(.1)
            fransje_gui.closeMouth()
            time.sleep(random.randint(1, 5))


#Function to wipe memory of Fransje
def wipeMemory():
    global message_history, conversation_start
    message_history = conversation_start
    print("Memory wiped!")
    print(message_history)
    return True
    

#Function to create TTS via Google Cloud TTS
def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, speaking_rate=1.12)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = "speech.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

    return filename


#Class that can play audio file
class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        global animateTalking, animateIdle
        animateTalking = True
        animateIdle = False
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        global animateTalking, animateIdle
        """ Graceful shutdown """ 
        animateTalking = False
        animateIdle = True
        self.stream.close()
        self.p.terminate()


#Runs on press of a button
def on_press(key):
    global recording
    try:
        # print('Key {0} pressed'.format(key.char))
        if 'char' in dir(key):     #check if char method exists,
            if key.char == 'r':    #check if it is 'q' key
                recording = True
                # print("fviokbhvxfb")

            if key.char == 'w':    #check if it is 'w' key
                wipeMemory()
                # print("fviokbhvxfb")
                
        
    except AttributeError:
        print('Key {0} pressed'.format(key))
        #Add Code


#Runs on release of a button
def on_release(key):
    global recording
    # print('{0} released'.format(key))
    #Add your code to stop motor
    if key == keyboard.Key.esc:
        return False
    if 'char' in dir(key):     #check if char method exists,
        if key.char == 'r':    #check if it is 'q' key
            # print("fviokbhvxfb")
            recording = False


# Function that checks and records audio on keypress
def recordAudio():
    global recording
    filename = 'output.wav'

    while True:
        if recording == True:
            # Open the sound file 
            wf = wave.open(filename, 'rb')
            # Create an interface to PortAudio
            p = pyaudio.PyAudio()
            # Open a .Stream object to write the WAV file to
            # 'output = True' indicates that the sound will be played rather than recorded
            stream = p.open(format = FORMAT,
                            channels = CHANNELS,
                            rate = RATE,
                            input = True)

            # Read data in chunks
            frames = []
            print("Recording started")
            # Play the sound by writing the audio data to the stream
            while recording == True:
                data = stream.read(CHUNK)
                frames.append(data)

            # Close and terminate the stream
            print("Recording ended")
            stream.close()
            p.terminate()
            # Save the recorded data as a WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            processConversation()


#Start listening for keyboard events
def startKeyboardListener():
    # Collect events until released

    print("Press and hold the 'r' key to begin recording")
    print("Release the 'r' key to end recording")

    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()


#Process the audio to transcription and output the reply to audio
def processConversation():
    new_message = transcribeAudio("output.wav")
    print(new_message.text)
    completed_message = addContextValues(new_message.text)
    reply = sendMessage(completed_message)
    print(reply)
    frames = empty_frames
    # wavfile.write(WAVE_OUTPUT_FILENAME, RATE, np.array(empty_frames))
    audio = AudioFile(text_to_wav("nl-NL-Wavenet-C", reply))
    audio.play()
    audio.close()


#Add background data to the prompt sent to ChatGPT (Fransje)
def addContextValues(message):
    global enable_emotion, enable_timedate, enable_weather, enable_pois
    
    complete_message = message + "\n\n Achtergrondinformatie:"
    if enable_emotion == True:
        complete_message += "\n - Gebruiker kijkt:" + str(last_emotion)

    if enable_timedate == True:
        complete_message += "\n - Het is:" + str(time.ctime())
    
    if enable_weather == True:
        complete_message += "\n - Temperatuur buiten:" + str(round(current_w_temp)) + " graden, " + str(current_w_stat)

    if enable_pois == True:
        complete_message += "\n - Plekken in de buurt: " + points_of_interest

    return complete_message


#Get current weather 
def get_current_weather():
    global current_w_temp, current_w_stat
    owm = OWM('2428a542349599e762eae9c08c9981b4')
    mgr = owm.weather_manager()

    # Search for current weather in London (Great Britain) and get details
    observation = mgr.weather_at_place('Eindhoven,NL')
    w = observation.weather

    status = w.detailed_status         # 'clouds'
    current_w_stat = status
    # w.wind()                  # {'speed': 4.6, 'deg': 330}
    temp = w.temperature('celsius')  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}
    current_w_temp = temp['temp']
    rain = w.rain                    # {}

    return temp, rain, status


#Send message to ChatGPT (Fransje)
def sendMessage(message_text):
    global message_history
    if enable_memory == True:
        message_history.append({"role": "user", "content": message_text})
        send_messages = message_history
        print(message_history)
    else:
        clean_history = conversation_start
        clean_history.append({"role": "user", "content": message_text})
        send_messages = clean_history
        print(clean_history)

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=send_messages
    )
    message_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    processed_message = clean(completion.choices[0].message.content, no_emoji=True)
    return processed_message


#Transcribe audio and return string
def transcribeAudio(audio_path):
    audio_file = open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcript


#Search for most occurring object in list
def most_frequent(List):
    return max(set(List), key = List.count)


#Initiate the process of emotion recognition that takes the average emotion every 2.5s
def initEmotionRecognition():
    global last_emotion
    all_emotions = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    break_while = False
    
    while True:
        start_time = time.time()
        elapsed_time = 0
        while elapsed_time < 2.5:
            current_time = time.time()
            elapsed_time = current_time - start_time
            ret,frame = cap.read()
            result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False, silent = True)

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

            emotion = result[0]['dominant_emotion']
            all_emotions.append(emotion)
            
            txt = str(emotion)

            cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            # cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break_while = True
                break

        last_emotion = most_frequent(all_emotions)
        print(last_emotion)
        all_emotions.clear()
        if break_while == True:
            break

    cap.release()
    cv2.destroyAllWindows()


# Gather the top 4 points of interest nearby in a radius of 2000m
def getPois():
    global points_of_interest, maps_client, adress, use_true_location

    search_radius = "2000"
    top = 4

    if use_true_location == True:
        location = maps_client.geolocate()
        lat = location['location']['lat']
        lng = location['location']['lng']

    else:
        geocode_result = maps_client.geocode(adress)
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']

    # print(lat, lng)
    place = maps_client.places_nearby(location=str(lat) + ", " + str(lng), radius=search_radius) 
    all_places=place["results"]
    # print(all_places)

    pois = pd.read_json(json.dumps(all_places))
    # print(pois['name'])
    # print(list(pois.columns.values))
    pois = pois.loc[:, pois.columns.intersection(['name','types','user_ratings_total'])]
    pois = pois.nlargest(top, 'user_ratings_total')
    pois.loc[:, 'types'] = pois.types.map(lambda x: x[0])
    pois = pois.drop(columns=['user_ratings_total'])
    pois = pois.reset_index()
    pois = pois.drop(columns=['index'])
    # print(list(pois.columns.values))
    # print(pois)

    points_of_interest = ""
    for index, row in pois.iterrows():
        points_of_interest += str(row['name']) + ", " + str(row['types'] + ". \n")

    return points_of_interest






# BEGIN ACTUAL CODE RUNNING
# --------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    #Set up the GUI
    app = QApplication(sys.argv)
    fransje_gui = FransjeGui()
    fransje_gui.show()

    started = False
    stream = None

    # Get points of interest around location
    print(getPois())

    #Get weather data of current location
    print(get_current_weather())

    #Set up audiorecorder
    audiorecorder = threading.Thread(target=recordAudio)
    audiorecorder.start()

    #Start watching the keyboard for pressing 'r' key
    keyboardlistener = threading.Thread(target=startKeyboardListener)
    keyboardlistener.start()

    #Set up emotion recognition
    emotions = threading.Thread(target=initEmotionRecognition)
    emotions.start()

    #Set up talking animation
    startTalking = threading.Thread(target=startTalkAnimation)
    startTalking.start()

    #Set up idle animation
    startIdle = threading.Thread(target=startIdleAnimation)
    startIdle.start()
    
    #Open GUI loop
    sys.exit(app.exec())
