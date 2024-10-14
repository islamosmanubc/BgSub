import cv2
vidcap = cv2.VideoCapture('sample_videos/water_foam.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("sample_videos/frames/4/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1