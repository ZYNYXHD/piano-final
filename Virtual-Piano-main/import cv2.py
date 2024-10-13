import cv2
import mediapipe as mp
import pyglet
import time

# Constants
wCam, hCam = 800, 600  # Increased camera resolution for better visibility
w, h = 30, 150  # Width and height of white keys
w_black, h_black = 20, 100  # Width and height of black keys

# Updated Playlist for each key sound (now using .wav format with key names)
playlist = [
    './tones/C.wav',        # C
    './tones/C#.wav',       # C#
    './tones/D.wav',        # D
    './tones/D#.wav',       # D#
    './tones/E.wav',        # E
    './tones/F.wav',        # F
    './tones/F#.wav',       # F#
    './tones/G.wav',        # G
    './tones/G#.wav',       # G#
    './tones/A.wav',        # A
    './tones/A#.wav',       # A#
    './tones/B.wav',        # B
    './tones/C.wav',       # C (next octave)
    './tones/C#.wav',      # C# (next octave)
    './tones/D.wav',       # D (next octave)
    './tones/D#.wav',      # D# (next octave)
    './tones/E.wav',       # E (next octave)
    './tones/F.wav',       # F (next octave)
    './tones/F#.wav',      # F# (next octave)
    './tones/G.wav',       # G (next octave)
    './tones/G#.wav',      # G# (next octave)
    './tones/A.wav',       # A (next octave)
    './tones/A#.wav',      # A# (next octave)
    './tones/B.wav'        # B (next octave)
]

# White keys (14 white keys for 2 octaves: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
white_key_positions = [(i * 40 + 30, 0) for i in range(14)]

# Black keys positions (10 black keys for 2 octaves)
black_key_positions = [
    (i * 40 + 60, 0) if i % 7 != 2 and i % 7 != 6 else (-1, -1)  # Skip positions for E-F and B-C
    for i in range(14)
]
# Remove invalid black key positions
black_key_positions = [pos for pos in black_key_positions if pos != (-1, -1)]

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Load all songs as sources
songs = [pyglet.media.load(song) for song in playlist]

# Create player objects for each sound
players = [pyglet.media.Player() for _ in songs]

# Add each source to its corresponding player
for i, song in enumerate(songs):
    players[i].queue(song)

# Variable to store the currently playing player
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
            xList = []
            yList = []
            lList = []

            for id, lm in enumerate(myHand.landmark):
                hi, wi, c = img.shape
                cx, cy = int(lm.x * wi), int(lm.y * hi)
                xList.append(cx)
                yList.append(cy)
                lList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            lmList.append(lList)

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

    return img, lmList


def playMusic(p1, p2):
    """Plays music based on hand position on the virtual piano."""
    global current_player  # Use the global variable for current player
    for i, (wx, wy) in enumerate(white_key_positions):
        if (wx < p1 < wx + w) and (wy < p2 < wy + h):  # Check if within bounds
            cv2.rectangle(img, (wx, wy), (wx + w, wy + h), (255, 0, 255), -1)
            player = players[i % len(players)]  # Map to white keys
            
            # Stop current player if it's playing
            if current_player and current_player.playing:
                current_player.pause()  # Pause current player
            current_player = player  # Set new player to current
            current_player.play()  # Play new sound

    for i, (bx, by) in enumerate(black_key_positions):
        if (bx < p1 < bx + w_black) and (by < p2 < by + h_black):  # Check if within bounds
            cv2.rectangle(img, (bx, by), (bx + w_black, by + h_black), (0, 0, 0), -1)
            player = players[i + 12]  # Map to black keys (adjust the index accordingly)
            
            # Stop current player if it's playing
            if current_player and current_player.playing:
                current_player.pause()  # Pause current player
            current_player = player  # Set new player to current
            current_player.play()  # Play new sound


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the camera feed
    img, results = findHands(img)
    img, lmlist = findPositions(img, results)

    # Draw white keys filled
    for wx, wy in white_key_positions:
        cv2.rectangle(img, (wx, wy), (wx + w, wy + h), (255, 255, 255), -1)  # Fill white keys

    # Draw black keys filled
    for bx, by in black_key_positions:
        cv2.rectangle(img, (bx, by), (bx + w_black, by + h_black), (0, 0, 0), -1)  # Fill black keys

    # Play music based on hand positions
    if len(lmlist) > 0:
        for hand in lmlist:
            p1, p2 = hand[8][1], hand[8][2]  # Index finger tip
            p3, p4 = hand[12][1], hand[12][2]  # Middle finger tip
            
            # Play music for both fingers
            playMusic(p1, p2)
            playMusic(p3, p4)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Break on Escape key
        break

cap.release()
cv2.destroyAllWindows()
