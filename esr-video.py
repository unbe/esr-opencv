import numpy as np
import argparse
import cv2
import tesseract
import re
 
def moduloTenRecursive(number):
    lut = [0, 9, 4, 6, 8, 2, 7, 1, 3, 5];
    carryover = 0;
    for i in str(number):
        t = carryover + int(i)
        carryover = lut[t % 10];
    return str((10 - carryover) % 10)

def generateCodeline(bc, chf, rappen, help1, referenceNumber, help2, participantNumber, help3):
    chf = str(chf)
    rappen = str(rappen)
    if len(chf) < 8:  # check if amount has less than eight chars
        chf = (8-len(chf))*"0" + chf
        
    # dynamic, check digit for bc and value (calculated with modulo 10 recursive)
    p1 = moduloTenRecursive(bc + chf + rappen);

    # dynamic, check digit for referenceNumber (calculated with modulo 10 recursive)
    p2 = moduloTenRecursive(referenceNumber);

    # dynamic, check digit for participantNumber (calculated with modulo 10 recursive)
    p3 = moduloTenRecursive(participantNumber);
    return bc + chf + rappen + p1 + help1 + referenceNumber + p2 + help2 + " " + participantNumber + p3 + help3;


def fixEsr(text):
  text = re.sub(r'[^0-9+>]', '', text)
  m = re.match('^([0-9]+)>([0-9]+)[+]([0-9]+)>?$', text)
  if m is None:
    return None
  group1 = m.group(1)
  if len(group1) < 13:
    return None
  bc = group1[0:2]
  chf = group1[2:10]
  rappen = group1[10:12]
  check1 = group1[12]
  
  group2 = m.group(2)
  if len(group2) < 2:
    return None
  ref = group2[0:-1]
  check2 = group2[-1]

  group3 = m.group(3)
  if len(group3) < 2:
    return None
  account = group3[0:-1]
  check3 = group3[-1]

  print bc, chf, rappen, ref, account
  if (check1 != moduloTenRecursive(bc + chf + rappen) or
      check2 != moduloTenRecursive(ref) or
      check3 != moduloTenRecursive(account)):
    return None
  return generateCodeline(bc, chf, rappen, '>', ref, '+', account, '>')

print fixEsr('0100000120000>9901433580614622+ 010010930>')

class Param(object):
  keys = {}
  names = {}

  def __init__(self, name, key, initial, minimum = None, maximum = None):
    self.name = name
    self.value = initial
    if minimum is None:
      self.toggle = True
    else:
      self.toggle = False
      self.minimum = minimum
      self.maximum = maximum
      self.step = 1
    self.index = len(Param.names)
    self.key = key
    Param.keys[key] = self
    Param.names[name] = self

  @staticmethod
  def Value(name):
    return Param.names[name].value

  @staticmethod
  def Processkey(key):
    chrkey = chr(key).lower()
    if not (chrkey in Param.keys):
      return
    instance = Param.keys[chrkey]
    instance.OnKeyPressed(chr(key).islower())
    Param.DisplayAll()

  @staticmethod
  def DisplayAll(image = None):
    for key, instance in Param.keys.iteritems():
      instance.Display(image)

  def OnKeyPressed(self, up = True):
    if self.toggle:
      self.value = not self.value
    else:
      if up:
        self.value += self.step
      else:
        self.value -= self.step
    
      if self.value > self.maximum:
        self.value = self.minimum
      if self.value < self.minimum:
        self.value = self.maximum
    
  def Display(self, image = None):
    descr = "[%s] %s: %s" % (self.key, self.name, str(self.value))
    if image is not None:
      cv2.putText(image, descr, (0, 10 + self.index * 14), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
      cv2.putText(image, descr, (1, 11 + self.index * 14), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    else:
      print descr


Param("bilateral", "b", True)
Param("contours", "c", True)
Param("all_contours", "a", False)
Param("lines_p", "p", True)
Param("lines", "l", False)
Param("ocr", "o", True)
Param("line_filter", "f", False)
Param("show_bad_lines", "x", False)
Param("lines_threshold", "t", 100, 0, 255)
Param("lines_minlength", "m", 60, 0, 255)
Param("lines_maxgap", "g", 5, 0, 255)
Param("dilate", "d", 1, 0, 3)
Param("erode", "e", 1, 0, 3)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", action='store_true', help = "Use camera")
ap.add_argument("-i", "--image", help = "Use image")
args = vars(ap.parse_args())

def geoClosestPoint(p, seg, testSegmentEnds=True):
    y3,x3 = p
    (y1,x1),(y2,x2) = seg

    dx21 = (x2-x1)
    dy21 = (y2-y1)
    
    lensq21 = dx21*dx21 + dy21*dy21
    if lensq21 == 0:
        dy = y3-y1
        dx = x3-x1
        return np.sqrt( dx*dx + dy*dy )  # return point to point distance

    u = (x3-x1)*dx21 + (y3-y1)*dy21
    u = u / float(lensq21)


    x = x1+ u * dx21
    y = y1+ u * dy21    

    if testSegmentEnds:
        if u < 0:
            x,y = x1,y1
        elif u >1:
            x,y = x2,y2
    return (y, x)    

def geoPointLineDist(p, seg, testSegmentEnds=True):
    """
    Minimum Distance between a Point and a Line
    Written by Paul Bourke,    October 1988
    http://astronomy.swin.edu.au/~pbourke/geometry/pointline/
    """
    (y, x) = geoClosestPoint(p, seg, testSegmentEnds)
    y3,x3 = p
    dx30 = x3-x
    dy30 = y3-y
    return np.sqrt( dx30*dx30 + dy30*dy30 )


def display(imgs):
  nx = int(np.ceil(np.sqrt(len(imgs))))
  ny = np.ceil((len(imgs) + 0.0)/nx)
  rows = []
  for y in range(int(ny)):
    irow = imgs[y * nx : (y + 1) * nx]
    while len(irow) < nx:
      irow.append(irow[-1])
    for i in range(len(irow)):
      if len(irow[i].shape) != 3 or irow[i].shape[2] != 3:
        irow[i] = cv2.cvtColor(irow[i], cv2.COLOR_GRAY2BGR)
    row = np.concatenate(irow, axis=1)
    rows.append(row)
  return np.concatenate(rows, axis=0)

tess_api = None
def process(image):
  global tess_api
  sz = 500
  ratio = (sz + 0.0) / image.shape[1]
  dim = (sz, int(image.shape[0] * ratio))
  orig = image
  image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  if Param.Value("bilateral"):
    bila = cv2.bilateralFilter(image, 11, 17, 17)
  else:
    bila = image
  gray = cv2.cvtColor(bila, cv2.COLOR_BGR2GRAY)

  canny = cv2.Canny(gray, 25, 50)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  if Param.Value("dilate"):
    canny = cv2.dilate(canny, kernel, iterations = Param.Value("dilate"))
  if Param.Value("erode"):
    canny = cv2.erode(canny, kernel, iterations = Param.Value("erode"))
  
  goodCnts = []
  badCnts = []
  if Param.Value('contours'):
    (cnts, hier) = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # loop over our contours
    for c in cnts:
      if cv2.contourArea(c) < 1000:
        continue
      # approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.01 * peri, True)
  #    print len(approx)
   
      good = False
      # if our approximated contour has four points, then
      # we can assume that we have found our screen
      if len(approx) == 4:
        d01 = np.sum((approx[0]-approx[1])**2)
        d12 = np.sum((approx[1]-approx[2])**2)
        wh_ratio = np.sqrt(np.divide(float(d01), d12))
        if wh_ratio > 1.5 and wh_ratio < 3:
          good = True
        elif 1/wh_ratio > 1.5 and 1/wh_ratio < 3:
          approx = np.roll(approx, 1, axis=0)
          good = True

      if good:
        goodCnts.append(approx)
      else:
        badCnts.append(approx)

    cv2.drawContours(image, goodCnts, -1, (0, 255, 0), 2)
    if Param.Value('all_contours'):
      cv2.drawContours(image, badCnts, -1, (128, 128, 0), 1)

      
  warped = None
    
  if Param.Value('lines_p') and len(goodCnts) > 0:
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, Param.Value('lines_threshold'), minLineLength = Param.Value('lines_minlength'), maxLineGap = Param.Value('lines_maxgap'))
    if lines is None:
      lines = []
    else:
      lines = lines[0]
    goodLines = []
    goodBoxes = []
    badLines = []
    for line in lines:        
      line = np.array([[line[0], line[1]], [line[2], line[3]]])
      length2 = np.sum((line[0]-line[1])**2)
      ok = False
      for cnt in goodCnts:
        minLength2 = np.sum((cnt[0]-cnt[3])**2) * 2
        if length2 < minLength2:
          continue
        d = [geoPointLineDist(point[0], line) for point in cnt]
        if d[0] < 10 and d[3] < 10:
          goodLines.append(line)
          goodBoxes.append(cnt)
        elif d[1] < 10 and d[2] < 10:
          goodLines.append(line)
          goodBoxes.append(np.roll(cnt, 2, axis=0))
        elif Param.Value('show_bad_lines'):
          badLines.append(line)

    if Param.Value('show_bad_lines'):
      for line in goodLines:
        cv2.line(image, line[0], line[1], (0,0,255), 1)
      for line in badLines:
        cv2.line(image, line[0], line[1], (0,0,255), 1)

    if len(goodLines) > 0:
      g = np.array(goodLines)
      longestIndex = np.argmax(np.sum((g[:,1:2,:] - g[:,0:1,:])**2, axis=2))
      longestLine = goodLines[longestIndex]
      cv2.line(image, tuple(longestLine[0]), tuple(longestLine[1]), (200, 200, 0), 2)
      box = goodBoxes[longestIndex]
      tl = np.array(geoClosestPoint(box[0][0], longestLine, False))
      bl = np.array(geoClosestPoint(box[3][0], longestLine, False))
      tr = box[1][0]
      br = box[2][0]

      h = bl - tl
      tl = tl + h * 3.1
      bl = tl + h
      h = br - tr
      tr = tr + h * 3.1
      br = tr + h

      w = tr - tl
      tl = tl - w * 1.1
      tr = tr + w * 0.6
      w = br - bl
      bl = bl - w * 1.1
      br = br + w * 0.6

      tl = tl/ratio
      tr = tr/ratio
      bl = bl/ratio
      br = br/ratio

      width = int(np.sqrt(np.sum((tl-tr)**2)))
      height = int(np.sqrt(np.sum((tl-bl)**2)))

      dst = np.array([
          [0, 0],
          [width - 1, 0],
          [width - 1, height - 1],
          [0, height - 1]], dtype = "float32")

      rect = np.array([tl, tr, br, bl], dtype = "float32")
      M = cv2.getPerspectiveTransform(rect, dst)
      warped = cv2.warpPerspective(orig, M, (width, height))
      
  if Param.Value('lines'):
    lines = cv2.HoughLines(canny, 1, np.pi/180, Param.Value('lines_threshold'))
    for rho,theta in (lines[0] if lines is not None else []):
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
      y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
      x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
      y2 = int(y0 - 1000*(a))
      cv2.line(image, (x1,y1), (x2,y2), (0,128,128), 2)

  Param.DisplayAll(image)
  cv2.imshow("Image", display([image, bila, gray, canny]))
  if warped is not None:
    if Param.Value('ocr'):
      warped = cv2.bilateralFilter(warped, 11, 17, 17)
      if tess_api is None:
        tess_api = tesseract.TessBaseAPI()
        tess_api.Init(".","eng",tesseract.OEM_DEFAULT)
        tess_api.SetVariable("tessedit_char_whitelist", "0123456789+>")
        tess_api.SetPageSegMode(tesseract.PSM_AUTO)
      cv2.imwrite("ocr.png", warped)   # SetCvImage segfauls. Fuck it.
      pixImage=tesseract.pixRead("ocr.png")
      tess_api.SetImage(pixImage)
      outText=tess_api.GetUTF8Text()
      for line in outText.split("\n"):
        if len(line) > 40:
          print line
          esr = fixEsr(line)
          if esr is None:
            esr = fixEsr(re.sub(r'\d (\d)', '+ \\1', line))
          print esr
          if esr is not None:
            cv2.putText(warped, esr, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 128), 2)
            cv2.imshow("Image2", warped)

if not args.get("camera", False):
  image = cv2.imread(args["image"])
  while True:
    process(image)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"): break
    Param.Processkey(key)

  exit(0)
else:
  camera = cv2.VideoCapture(0)

Param.DisplayAll()

while True:
  (grabbed, frame) = camera.read()
  process(frame)

  key = cv2.waitKey(1) & 0xFF
  # if the 'q' key is pressed, stop the loop
  if key == ord("q"): break
  Param.Processkey(key)

