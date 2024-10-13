import cv2
import mediapipe as mp
import pyglet
import numpy as np

# Constants
wCam, hCam = 800, 600  # Camera resolution
w, h = 30, 150  # White key dimensions
w_black, h_black = 20, 100  # Black key dimensions

# Updated Playlist for each key sound
playlist = [
    './tones/C.wav', './tones/C#.wav', './tones/D.wav', './tones/D#.wav', './tones/E.wav', './tones/F.wav',
    './tones/F#.wav', './tones/G.wav', './tones/G#.wav', './tones/A.wav', './tones/A#.wav', './tones/B.wav',
    './tones/C.wav', './tones/C#.wav', './tones/D.wav', './tones/D#.wav', './tones/E.wav', './tones/F.wav',
    './tones/F#.wav', './tones/G.wav', './tones/G#.wav', './tones/A.wav', './tones/A#.wav', './tones/B.wav'
]

# White and black key positions
white_key_positions = [(i * 40 + 30, 0) for i in range(14)]
black_key_positions = [(i * 40 + 60, 0) if i % 7 != 2 and i % 7 != 6 else (-1, -1) for i in range(14)]
black_key_positions = [pos for pos in black_key_positions if pos != (-1, -1)]

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Load sound files as sources
songs = [pyglet.media.load(song) for song in playlist]
players = [pyglet.media.Player() for _ in songs]
for i, song in enumerate(songs):
    players[i].queue(song)

# Variable to store current player
current_player = None

def findHands(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img, results

def findPositions(img, results, draw=True):
    lmList = []
    if results.multi_hand_landmarks:
        for myHand in results.multi_hand_landmarks:
            xList, yList, lList = [], [], []
            for id, lm in enumerate(myHand.landmark):
                hi, wi, c = img.shape
                cx, cy = int(lm.x * wi), int(lm.y * hi)
                xList.append(cx)
                yList.append(cy)
                lList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            lmList.append(lList)
    return img, lmList

def playMusic(p1, p2):
    global current_player
    for i, (wx, wy) in enumerate(white_key_positions):
        if (wx < p1 < wx + w) and (wy < p2 < wy + h):
            cv2.rectangle(img, (wx, wy), (wx + w, wy + h), (255, 0, 255), -1)
            player = players[i % len(players)]
            if current_player and current_player.playing:
                current_player.pause()
            current_player = player
            current_player.play()

    for i, (bx, by) in enumerate(black_key_positions):
        if (bx < p1 < bx + w_black) and (by < p2 < by + h_black):
            cv2.rectangle(img, (bx, by), (bx + w_black, by + h_black), (0, 0, 0), -1)
            player = players[i + 12]
            if current_player and current_player.playing:
                current_player.pause()
            current_player = player
            current_player.play()

while True:
    success, img = cap.read()

    # Chroma key effect: draw piano keys on green screen
    piano_img = np.zeros_like(img)
    piano_img[:] = [0, 255, 0]  # Green background

    # Draw white keys
    for wx, wy in white_key_positions:
        cv2.rectangle(piano_img, (wx, wy), (wx + w, wy + h), (255, 255, 255), -1)

    # Draw black keys
    for bx, by in black_key_positions:
        cv2.rectangle(piano_img, (bx, by), (bx + w_black, by + h_black), (0, 0, 0), -1)

    # Chroma key effect to remove green screen
    hsv_piano = cv2.cvtColor(piano_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_piano, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    piano_without_green = cv2.bitwise_and(piano_img, piano_img, mask=mask_inv)
    camera_bg = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.add(piano_without_green, camera_bg)

    # Detect hands and play music
    img, results = findHands(img)
    img, lmlist = findPositions(img, results)
    if len(lmlist) > 0:
        for hand in lmlist:
            p1, p2 = hand[8][1], hand[8][2]  # Index finger tip
            p3, p4 = hand[12][1], hand[12][2]  # Middle finger tip
            playMusic(p1, p2)
            playMusic(p3, p4)

    # Show the final image
    cv2.imshow("Virtual Piano", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
