from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("./Videos/mixvehicle.mp4")
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO('../yolo-weights/yolov5n6u.pt')
#mask = cv2.imread("mask.png")

#global variables
gologicd2_seconds = 0 #duration of go of logic device 2
fvehicledetected_sec = 0 #number of seconds first vehicle detected
frames_sufvehicle = 0 # number of frames that has sufficient vehicles

#SAMPLE TRAFFIC SIGNAL
logic_device1 = False
logic_device2 = True

#timings (adjustable) constant
GOANDSTOP_TIMING = 30
YELLOW_TIMING = 15
#timings for exception condition
EXCEPTION_TIMING = 40

#number of vehicles
LIM_BVEHICLE = 2 # trigger value big vehicles
LIM_SVEHICLE = 3 # trigger value small vehicles
LIM_TVEHICLE = 3 # trigger value total vehicles

#adjust required frames to trigger true in check vehicle function
FRAMES_REQUIRED = 5


def check_vehicle(count_bvehicle, count_svehicle):
    global frames_sufvehicle
    global fvehicledetected_sec
    totalVehicles = count_bvehicle + count_svehicle
    if(frames_sufvehicle>=FRAMES_REQUIRED or fvehicledetected_sec >= EXCEPTION_TIMING):
        frames_sufvehicle = frames_sufvehicle * 0
        fvehicledetected_sec = fvehicledetected_sec * 0
        return True
    if (totalVehicles>= LIM_TVEHICLE):
        frames_sufvehicle = frames_sufvehicle + 1
        print("FRAMES HAVE SUFFC: %s" % frames_sufvehicle)
        return False
    elif(count_bvehicle >= LIM_BVEHICLE ): # big vehicle is greater than or equal to 3
        frames_sufvehicle = frames_sufvehicle + 1
        return False
    elif(count_svehicle >= LIM_SVEHICLE): # small vehicle is greater than or equal to 4
        frames_sufvehicle = frames_sufvehicle + 1
        return False
    elif(totalVehicles!=0): #check if not empty only if 50 seconds pass
        fvehicledetected_sec = fvehicledetected_sec + 1
        return False
    else:
        frames_sufvehicle = frames_sufvehicle * 0
        fvehicledetected_sec = fvehicledetected_sec * 0
        return False

def yellow_signal():
    global logic_device1
    if (logic_device1 == True):
        print("Traffic Light 1: YELLOW")
        print("Traffic Light 2: YELLOW RED")
    else:
        print("Traffic Light 1: YELLOW RED")
        print("Traffic Light 2: YELLOW")
    return

def go_logicdevice1():
    global logic_device1
    global logic_device2
    print("Traffic Light 1: %s" % logic_device1)
    print("Traffic Light 2: %s" % logic_device2)
    # green duration of logic device 1
    if (logic_device1 == True):
        for sec in range(GOANDSTOP_TIMING):
            print("Go: %s" % (sec+1))
            time.sleep(1)
        yellow_signal()
        for sec in range(YELLOW_TIMING):
            print("Yellow: %s" % (sec+1))
            time.sleep(1)
        logic_device1 = not logic_device1
        logic_device2 = not logic_device2
    return
def signal_change(check_vehicle): # TRUE=GO FALSE=STOP
    global logic_device1
    global logic_device2
    global gologicd2_seconds
    if(check_vehicle==True and gologicd2_seconds>=GOANDSTOP_TIMING-1): #CHANGE SIGNAL STATUS
        yellow_signal()
        for sec in range(YELLOW_TIMING): #15 seconds yellow
            print("Yellow: %s" % (sec+1))
            time.sleep(1)
        #signal change
        logic_device1= not logic_device1
        logic_device2= not logic_device2
        gologicd2_seconds = gologicd2_seconds * 0
        go_logicdevice1()
        print("Traffic Light 1: %s" % logic_device1)
        print("Traffic Light 2: %s" % logic_device2)
    else:
        print("Traffic Light 1: %s" %logic_device1)
        print("Traffic Light 2: %s" %logic_device2)
    gologicd2_seconds = gologicd2_seconds + 1
    return
def main():
    while True:
        success, img = cap.read()
        #image_region = cv2.bitwise_and(img, mask)
        results = model(img, stream=True)
        #Number of Objects Detected
        vehicle_class1 = 0 #big vehicles(Tricycle, cars, truck)
        vehicle_class2 = 0 #small vehicles(motor, ebike)
        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"
                      ]

        for r in results:
            vehicle_class1 = vehicle_class1 * 0
            vehicle_class2 = vehicle_class2 * 0
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_detected = classNames[int(box.cls[0])]
                if(confidence >= 0.40 and  class_detected == "car" or class_detected == "bus" or class_detected == "truck" or  class_detected == "motorbike"):
                    # cvzone.cornerRect(img, (x1, y1, (x2 - x1), (y2 - y1)), l=5)
                    # cvzone.putTextRect(img, f'{classNames[int(box.cls[0])]} {confidence}', (max(0, x1), max(40, y1 - 20)), scale=0.6, thickness=1, offset=5)
                    if(class_detected == "car" or class_detected == "bus" or class_detected == "truck"):
                        vehicle_class1 = vehicle_class1 + 1
                    else:
                        vehicle_class2 = vehicle_class2 + 1
                # print(confidence)
                # print(x1, y1, x2, y2)
            check_v = check_vehicle(vehicle_class1, vehicle_class2)
            print("Big vehicle: %s  Small vehicle: %s" %(vehicle_class1,vehicle_class2))
            print("Trigger Signal Change: %s" %(check_v))
            #signal_change(check_vehicle(vehicle_class1, vehicle_class2))
            signal_change(check_v)
            print("Go duration(logic device 2): %s"%gologicd2_seconds)
         # cv2.imshow("Video", img)
         # cv2.waitKey(1)

if __name__ == "__main__":
    main()


