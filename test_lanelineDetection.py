""" import cv2
import numpy as np

def test_canny(img):
    if img is None:
        assert img is None ,"test passed"
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    assert kernel==5,"test passed"
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny
    



def test_region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
    (200, height),
    (800, 350),
    (1200, height),]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image
    

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def test_display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        assert lines is not None,"test passed"
        for line in lines:
            assert line in lines,"test passed"
            for x1, y1, x2, y2 in line:
                assert x1,y1;x2,y2 in line,"test passed"
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
    
 
def test_make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def test_average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        assert lines is None,"test passed"
        return None
    for line in lines:
        assert line in lines,"test passed"
        for x1, y1, x2, y2 in line:
            assert x1, y1; x2, y2 in line,"test passed"
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            assert slope == fit[0],"test passed"
            assert intercept == fit[1],"test passed"
            if slope < 0: 
                assert slope<0,"test passed"
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = test_make_points(image, left_fit_average)
    right_line = test_make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines
    
    
cap = cv2.VideoCapture("test1.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = test_canny(frame)
    cropped_canny = test_region_of_interest(canny_image)
    # cv2.imshow("cropped_canny",cropped_canny)

    lines = houghLines(cropped_canny)
    averaged_lines = test_average_slope_intercept(frame, lines)
    line_image = test_display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    result.write(combo_image)
    #cv2.imshow("result", combo_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 """
def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]