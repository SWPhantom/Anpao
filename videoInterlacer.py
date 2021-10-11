import cv2
import numpy as np
import time
from datetime import datetime

N = 60
# gets next frame of video. Returns frame if success. Returns None if failure after N tries tries
def getNextFrame(video):
  blank = N

  ret, frame = video.read()
  while not ret:
    print(f'{ret} returned')
    blank -= 1
    if blank <= 0:
      print('blanks are 0!!')
      return None
    ret, frame = video.read()
  
  return frame

# Interlaces images row by row.
# Takes in two images, outputs an image of w*h size with alternating input image pixel rows
def interlace(imgL, imgR, h, w):
    inter = np.empty((h, w, 3), imgL.dtype)
    inter[0:h:2, :w, :] = imgL[0:h:2, :w, :]
    inter[1:h:2, :w, :] = imgR[1:h:2, :w, :]
    print()
    return inter


inputFile1 = './input/fixedMove1.MP4'
inputFile2 = './input/fixedMove2.MP4'
outputFile = './output/' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + 'interlace.mp4'

def main():
  v1 = cv2.VideoCapture(inputFile1)
  v2 = cv2.VideoCapture(inputFile2)
  fps = v1.get(cv2.CAP_PROP_FPS)

  print(f'writing to {outputFile}')
  out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (int(v1.get(3)), int(v1.get(4))))
  frames = 1000

  frame1a = np.zeros((1920, 1080, 3), np.uint8)
  frame2a = np.zeros((1920, 1080, 3), np.uint8)

  # Synchronization
  # Temporal
  # Currenly, allows user to advance one or both of the video sources until synchronization is found.
  # TODO: Add sync metadata file that captures the video sources and the frame offset to skip this manual sync
  # TODO: Add way to auto-sync, either with jitter matching OR by using a known sync visual signal (LED flash/color pattern match, etc)
  while (v1.isOpened() and v2.isOpened()):
    # Capture frame-by-frame
    user = input("l for left, r for right, b for both (or first), c for continue:")
    
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

  cv2.destroyAllWindows()

  # Positional
  # Here goes the manual alignment of the frames, to keep the center overlapped
  # Temporary, as alignment should be done per frame
  horizontalTranslation = 0
  verticalTranslation = 0
  while (v1.isOpened() and v2.isOpened()):
    translate1 = np.float32([[1, 0, horizontalTranslation],[0, 1, verticalTranslation]])
    translate2 = np.float32([[1, 0, -horizontalTranslation],[0, 1, -verticalTranslation]])
    inter = interlace(cv2.warpAffine(frame1a, translate1, (frame1a.shape[1], frame1a.shape[0])), cv2.warpAffine(frame2a, translate2, (frame2a.shape[1], frame2a.shape[0])), 1080, 1920)
    
    cv2.imshow('alignment', inter)
    user = input("h for horizonal alignment, v for vertical alignment, b for both new frames. c for continue:")
    user2 = input("amount:")
    if user == 'h':
      horizontalTranslation += int(user2)
    elif user == 'v':
      verticalTranslation += int(user2)
    elif user == 'b':
      frame1a = getNextFrame(v1)
      frame2a = getNextFrame(v2)
    elif user == 'c':
      break
    else:
      print("wrong!")
      continue

    cv2.waitKey(10)
    

  # Loop until the end of the video and interlace the images
  while (v1.isOpened() and v2.isOpened()):
    translate1 = np.float32([[1, 0, horizontalTranslation],[0, 1, verticalTranslation]])
    translate2 = np.float32([[1, 0, -horizontalTranslation],[0, 1, -verticalTranslation]])

    frame1a = getNextFrame(v1)
    frame2a = getNextFrame(v2)

    if frame1a is None or frame2a is None:
      break

    #frame1a = cv2.resize(frame1a, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
    #frame1b = cv2.resize(frame1b, (1920, 1080), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

    inter = interlace(cv2.warpAffine(frame1a, translate1, (frame1a.shape[1], frame1a.shape[0])), cv2.warpAffine(frame2a, translate2, (frame2a.shape[1], frame2a.shape[0])), 1080, 1920)

    cv2.imshow("test", inter)
    out.write(inter)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

  # release the video capture object
  v1.release()
  v2.release()
  out.release()

  # Closes all the windows currently opened.
  cv2.destroyAllWindows()

if __name__=="__main__":
  main()