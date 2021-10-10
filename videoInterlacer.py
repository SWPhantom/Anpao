import cv2
import numpy as np
import time

# gets next frame of video. Returns frame if success. Returns None if failure after 10 tries
def getNextFrame(video):
  blank = 60
  ret, frame = video.read()
  while not ret:
    print(f'{ret} returned')
    blank -= 1
    if blank <= 0:
      print('blanks are 0!!')
      return None
    ret, frame = video.read()
  
  return frame


def interlace(imgL, imgR, h, w):
    inter = np.empty((h, w, 3), imgL.dtype)
    inter[0:h:2, :w, :] = imgL[0:h:2, :w, :]
    inter[1:h:2, :w, :] = imgR[1:h:2, :w, :]
    print()
    return inter

v1 = cv2.VideoCapture('newL.MP4')
v2 = cv2.VideoCapture('newR.MP4')
fps = v1.get(cv2.CAP_PROP_FPS)


out = cv2.VideoWriter('666interlace10.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (int(v1.get(3)), int(v1.get(4))))
frames = 1000

frame1a = np.zeros((1920, 1080, 3), np.uint8)
frame2a = np.zeros((1920, 1080, 3), np.uint8)

# synchronize
while (v1.isOpened() and v2.isOpened()):
  # Capture frame-by-frame
  user = input("l for left, r for right, b for both (or first), c for continue")
  
  if user == 'b':
    frame1a = getNextFrame(v1)
    frame2a = getNextFrame(v2)
  elif user == 'l':
    frame1a = getNextFrame(v1)
  elif user == 'r':
    frame2a = getNextFrame(v2)
  elif user == 'c':
    break
  else:
    break

  if frame1a is None or frame2a is None:
    break

  print('printing new frames')
  cv2.imshow('left', frame1a)
  cv2.imshow('right', frame2a)
  cv2.waitKey(10)

blank = 0
# Loop until the end of the video
while (frames > 0 and v1.isOpened() and v2.isOpened()):
    # Capture frame-by-frame
    ret1, frame1a = v1.read()
    # ret2, frame1b = v1.read()

    ret3, frame2a = v2.read()
   # ret4, frame2b = v2.read()

    #if(not ret1 and not ret2 and not ret3 and not ret4):
    
    if(not ret1 and not ret3):
      blank += 1
      if(blank > 10):
        break

    blank = 0
    #if(not ret1 or not ret2 or not ret3 or not ret4):
    if(not ret1 or not ret3):
      continue

    # frames -= 1

    #frame1a = cv2.resize(frame1a, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    #frame1b = cv2.resize(frame1b, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    #frame2a = cv2.resize(frame2a, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    #frame2b = cv2.resize(frame2b, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)   

    # Display the resulting frame
    # cv2.imshow('f1', frame1a)
    # cv2.imshow('f2', frame1b)

    inter = interlace(frame1a, frame2a, 1080, 1920)
    cv2.imshow("test", inter)
    out.write(inter)
    #out.write(frame1b)
    #out.write(frame2a)
    #out.write(frame2b)
 
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
# release the video capture object
v1.release()
v2.release()
out.release()

# Closes all the windows currently opened.
cv2.destroyAllWindows()

