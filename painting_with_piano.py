# This program makes the keyboard act a keys on a piano. It also paints the live video stream when east key is pressed.

import pyaudio, struct
import numpy as np
from math import sin, cos, pi
from scipy import signal
from random import randint
import cv2

BLOCKLEN   = 2400        # Number of frames per block
WIDTH       = 2         # Bytes per sample
CHANNELS    = 1         # Mono
RATE        = 8000      # Frames per second

MAXVALUE = 2**15 - 1  # Maximum allowed output signal value (because WIDTH = 2)

# Parameters
Ta = 1      # Decay time (seconds)

f0 = 659.255
f1 = 587.330
f2 = 523.251
f3 = 783.99
f4 = 2**(4/12) * 440
f5 = 2**(5/12) * 440
f6 = 2**(6/12) * 440
f7 = 2**(7/12) * 440
f8 = 2**(8/12) * 440
f9 = 2**(9/12) * 440
f10 = 2**(10/12) * 440
f11 = 2**(11/12) * 440
f12 = 2**(12/12) * 440

# Pole radius and angle
r = 0.01**( 1.0 / ( Ta * RATE ) )       # 0.01 for 1 percent amplitude

om = 2.0 * pi * float(f0)/RATE
om1 = 2.0 * pi * float(f1)/RATE
om2 = 2.0 * pi * float(f2)/RATE
om3 = 2.0 * pi * float(f3)/RATE
om4 = 2.0 * pi * float(f4)/RATE
om5 = 2.0 * pi * float(f5)/RATE
om6 = 2.0 * pi * float(f6)/RATE
om7 = 2.0 * pi * float(f7)/RATE
om8 = 2.0 * pi * float(f8)/RATE
om9 = 2.0 * pi * float(f9)/RATE
om10 = 2.0 * pi * float(f10)/RATE
om11 = 2.0 * pi * float(f11)/RATE
om12 = 2.0 * pi * float(f12)/RATE

ORDER = 2
# Filter coefficients (second-order IIR)

a = [1, -2*r*cos(om), r**2]
a1 = [1, -2*r*cos(om1), r**2]
a2 = [1, -2*r*cos(om2), r**2]
a3 = [1, -2*r*cos(om3), r**2]
a4 = [1, -2*r*cos(om4), r**2]
a5 = [1, -2*r*cos(om5), r**2]
a6 = [1, -2*r*cos(om6), r**2]
a7 = [1, -2*r*cos(om7), r**2]
a8 = [1, -2*r*cos(om8), r**2]
a9 = [1, -2*r*cos(om9), r**2]
a10 = [1, -2*r*cos(om10), r**2]
a11 = [1, -2*r*cos(om11), r**2]
a12 = [1, -2*r*cos(om12), r**2]

b = [r*sin(om)]
b1 = [r*sin(om1)]
b2 = [r*sin(om2)]
b3 = [r*sin(om3)]
b4 = [r*sin(om4)]
b5 = [r*sin(om5)]
b6 = [r*sin(om6)]
b7 = [r*sin(om7)]
b8 = [r*sin(om8)]
b9 = [r*sin(om9)]
b10 = [r*sin(om10)]
b11 = [r*sin(om11)]
b12 = [r*sin(om12)]
b0 = np.sin(om1)

states0 = np.zeros(ORDER)
states1 = np.zeros(ORDER)
states2 = np.zeros(ORDER)
states3 = np.zeros(ORDER)
states4 = np.zeros(ORDER)
states5 = np.zeros(ORDER)
states6 = np.zeros(ORDER)
states7 = np.zeros(ORDER)
states8 = np.zeros(ORDER)
states9 = np.zeros(ORDER)
states10 = np.zeros(ORDER)
states11 = np.zeros(ORDER)
states12 = np.zeros(ORDER)

x = np.zeros(BLOCKLEN)
x1 = np.zeros(BLOCKLEN)
x2 = np.zeros(BLOCKLEN)
x3 = np.zeros(BLOCKLEN)
x4 = np.zeros(BLOCKLEN)
x5 = np.zeros(BLOCKLEN)
x6 = np.zeros(BLOCKLEN)
x7 = np.zeros(BLOCKLEN)
x8 = np.zeros(BLOCKLEN)
x9 = np.zeros(BLOCKLEN)
x10 = np.zeros(BLOCKLEN)
x11 = np.zeros(BLOCKLEN)
x12 = np.zeros(BLOCKLEN)

# Open the audio output stream
p = pyaudio.PyAudio()
PA_FORMAT = pyaudio.paInt16
stream = p.open(
        format      = PA_FORMAT,
        channels    = CHANNELS,
        rate        = RATE,
        input       = False,
        output      = True,
        frames_per_buffer = 64)
# specify low frames_per_buffer to reduce latency

print("Switch to video window. Then press 'q' to quit")
print('Press keys [ "e", "d", "c", "g", "y", "u", "i", "o", "p", "a", "s", "w", "f" ] for sound.')

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

cap.set(10,150)

myColorValues = []

#generating random colors
for i in range(7):
    red = randint(0,255)
    blue = randint(0,255)
    green = randint(0,255)
    color = [red, blue, green]
    myColorValues.append(color)



myPoints = []
count = 0
x_coor = 0
y_coor = 0
length = 0
color = 0
size = randint(30,50)

#function generates the color of paint and the location to paint
def getcolor(img, myColorValues):
    global count
    global x_coor
    global y_coor
    global length
    global color
    
    #get the location to start painting and which color paint with
    if count == 0:
        x_coor = randint(0, 640)
        y_coor = randint(0, 480)
        color = len(myColorValues) - 1
        color = randint(0, color)
        count += 1
    
    #get next location to paint
    elif x_coor < 640 and y_coor < 480 and count <= 6:
        direction = randint(1,2) #random number dictates if the program is moving in the vertical or horizontal posiiton
        if direction == 1:
            x_coor = x_coor+20
        else:
            y_coor = y_coor+20
        count += 1
    
    #makes sure the program is always painting within the video frame
    elif x_coor > 640 and count <= 6:
         x_coor = randint(0, 640)
         count += 1
    elif y_coor > 480 and count <= 6:
         y_coor = randint(0, 480)
         count += 1
        
    #creates the new point
    newPoints = []
    cv2.circle(imgResult, (x_coor,y_coor), 10, myColorValues[color], cv2.FILLED)
    if x_coor != 0 and y_coor != 0:
        newPoints.append([x_coor,y_coor,color])
        
    #after painting a general location six times, rested the count to zero so a new location is chosen
    if count > 6:
        count = 0
    return newPoints
    
#function tranfers to the new points and the color used to another array
def addpoints(newPoints, myPoints):
    if len(newPoints)!= 0:
        for newP in newPoints:
            myPoints.append(newP)
    
#this function paints the video based on the locations points and specific colors used
def drawOnCanvas(myPoints, myColorValues, size):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), size, myColorValues[point[2]], cv2.FILLED)
        
            

while cap.isOpened():
    
    [ok, frame] = cap.read()   # Read one video frame (ok is true if it works)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25)
    
    success, img = cap.read()
    imgResult = img.copy()

    
    key = cv2.waitKey(1)
    
    if key == -1:
        x[0] = 0.0
    elif key == ord("q"):
        # User pressed 'q', so quit
        break
    elif key == ord('e'):
        x[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('d'):
        x1[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('c'):
        x2[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('g'):
        x3[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('y'):
        x4[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('u'):
        x5[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('i'):
        x6[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('o'):
        x7[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('p'):
        x8[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('a'):
        x9[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('s'):
        x10[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('w'):
        x11[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    elif key == ord('f'):
        x12[0] = 15000.0
        newPoints = getcolor(img, myColorValues)
        addpoints(newPoints, myPoints)
    
    if len(myPoints)!= 0:
        drawOnCanvas(myPoints, myColorValues, size)
    
    cv2.imshow("Live Video", imgResult)

    [y0, states0] = signal.lfilter(b, a, x, zi = states0)
    [y1, states1] = signal.lfilter(b1, a1, x1, zi = states1)
    [y2, states2] = signal.lfilter(b2, a2, x2, zi = states2)
    [y3, states3] = signal.lfilter(b3, a3, x3, zi = states3)
    [y4, states4] = signal.lfilter(b4, a4, x4, zi = states4)
    [y5, states5] = signal.lfilter(b5, a5, x5, zi = states5)
    [y6, states6] = signal.lfilter(b6, a6, x6, zi = states6)
    [y7, states7] = signal.lfilter(b7, a7, x7, zi = states7)
    [y8, states8] = signal.lfilter(b8, a8, x8, zi = states8)
    [y9, states9] = signal.lfilter(b9, a9, x9, zi = states9)
    [y10, states10] = signal.lfilter(b10, a10, x10, zi = states10)
    [y11, states11] = signal.lfilter(b11, a11, x11, zi = states11)
    [y12, states12] = signal.lfilter(b12, a12, x12, zi = states12)
    
    x[0] = 0.0
    x1[0] = 0.0
    x2[0] = 0.0
    x3[0] = 0.0
    x4[0] = 0.0
    x5[0] = 0.0
    x6[0] = 0.0
    x7[0] = 0.0
    x8[0] = 0.0
    x9[0] = 0.0
    x10[0] = 0.0
    x11[0] = 0.0
    x12[0] = 0.0

    y0 = np.clip(y0.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y1 = np.clip(y1.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y2 = np.clip(y2.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y3 = np.clip(y3.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y4 = np.clip(y4.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y5 = np.clip(y5.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y6 = np.clip(y6.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y7 = np.clip(y7.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y8 = np.clip(y8.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y9 = np.clip(y9.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y10 = np.clip(y10.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y11 = np.clip(y11.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    y12 = np.clip(y12.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    
    ytotal = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12
    ytotal = np.clip(ytotal.astype(int), -MAXVALUE, MAXVALUE)     # Clipping
    
    # Convert numeric list to binary data
    data = struct.pack('h' * BLOCKLEN, *ytotal);

    # Write binary data to audio output stream
    stream.write(data, BLOCKLEN)

print('* Done *')

# Close audio stream
stream.stop_stream()
stream.close()
cap.release()
p.terminate()
cv2.destroyAllWindows()
