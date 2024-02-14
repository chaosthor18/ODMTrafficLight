import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

cap = cv2.VideoCapture("./Videos/6.mp4")
# cap = cv2.VideoCapture(1)
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO("../yolo-weights/VehiclesPH.pt")
#model = YOLO('../yolo-weights/PHVehicle.pt')
#mask = cv2.imread("mask.png")

#TRACKER
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# LINES
line1=[0,300,1500,300]
thick = 400
#Number of vehicle
vehicle_countingbox= []

#variable philsin tl
gosecs_ld2 = 0
MINSEC_GOSECSLD2 = 30+1 #min seconds to detect on ld1
#variable callejon
fvehicledet_hits = 0

#vehicle limit
VEHICLE_LIM = 3
#MINIMUM HITS TO TRIGGER CHANGE SIGNAL
MIN_HITS = 6
MIN_HITSEXP = 15 #if condition is not satisfied then this exception executes
curr_hits = 0

#timings (adjustable) constant
GOANDSTOP_TIMING = 30
READY_TIMING = 15

#GET MILLISECONDS
gold2start_millisec = 0

def counting_vehicle(cx,cy,id):
    if (line1[0] < cx < line1[2] and line1[1] - thick < cy < line1[1] + thick):  # line2 1
        if vehicle_countingbox.count(id) == 0:
            vehicle_countingbox.append(id)

def traffic_signal():
    global curr_hits
    global gosecs_ld2
    global fvehicledet_hits
    if (len(vehicle_countingbox)>=VEHICLE_LIM and gosecs_ld2 >= MINSEC_GOSECSLD2):
        curr_hits+=1
        if(curr_hits >= MIN_HITS):
            gosecs_ld2 = 0
            curr_hits =  0
            fvehicledet_hits = 0
            vehicle_countingbox.clear()
            go_logicd1()
        else:
            go_logicd2()
    elif (len(vehicle_countingbox) != 0 and gosecs_ld2 >= MINSEC_GOSECSLD2):
        curr_hits = 0
        fvehicledet_hits+=1
        if (fvehicledet_hits >= MIN_HITSEXP):
            gosecs_ld2 = 0
            curr_hits = 0
            fvehicledet_hits = 0
            vehicle_countingbox.clear()
            go_logicd1()
        else:
            go_logicd2()
    else:
        fvehicledet_hits = 0
        curr_hits = 0
        go_logicd2()


def go_logicd1(): #CALLEJON
    global gold2start_millisec
    gold2start_millisec=0
    for sec in range(READY_TIMING):
        print("READY (YELLOW RED LD1 AND RED LD2): %s" %(sec+1))
        time.sleep(1)
    for sec in range(GOANDSTOP_TIMING):
        print("GO/STOP (GREEN LD1 AND RED LD2): %s" %(sec+1))
        time.sleep(1)
    for sec in range(READY_TIMING):
        print("READY (RED LD1 AND YELLOW RED LD2): %s" %(sec+1))
        time.sleep(1)

def go_logicd2(): # PHILSIN
    global gosecs_ld2
    global gold2start_millisec
    if(gold2start_millisec == 0):
        gold2start_millisec = time.perf_counter()
    curr_millisec = time.perf_counter()
    gosecs_ld2 = round(curr_millisec-gold2start_millisec)
    print("GO/STOP (RED LD1 AND GREEN LD2): %s" %gosecs_ld2)

def main():
    while True:
        success, img = cap.read()
        #image_region = cv2.bitwise_and(img, mask)
        results = model(img, stream=True)
        classNames = ['Bus', 'Jeepney', 'Motorcycle', 'Tricycle', 'Van', 'cars', 'truck']
        # classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        #               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        #               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        #               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        #               "baseball bat",
        #               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        #               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        #               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        #               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        #               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        #               "teddy bear", "hair drier", "toothbrush"
        #               ]
        #classNames = ['Bus', 'Jeepney', 'Motorcycle', 'Tricycle', 'Van', 'cars', 'truck']
        #vehicleTarget = ['Bus', 'Jeepney', 'Motorcycle', 'Tricycle', 'Van', 'cars', 'truck']
        #vehicleTarget = ['car', 'motorbike', 'bus', 'truck']
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_detected = classNames[int(box.cls[0])]
                if(confidence >= 0.40):
                    currentArray = np.array([x1,y1,x2,y2,confidence])
                    detections = np.vstack((detections,currentArray))
        resultsTracker = tracker.update(detections)
        cv2.line(img, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), thick)
        for result in resultsTracker:
            x1,y1,x2,y2,id = result
            x1,y1,x2,y2,id = int(x1), int(y1), int(x2) , int(y2),int(id)
            cvzone.cornerRect(img, (x1, y1, (x2 - x1), (y2 - y1)), l=9, rt=2, colorR=(255,0,0))
            cvzone.putTextRect(img, f'{id}', (max(0, x1), max(40, y1 - 20)), scale=2, thickness=2, offset=3)
            cx,cy = x1+(x2-x1)//2, y1+(y2-y1)//2
            cv2.circle(img, (cx,cy), 3,(255,0,0), cv2.FILLED)
            counting_vehicle(cx,cy,id)
        traffic_signal()
        print("Vehicle Count: %s" % len(vehicle_countingbox))
        vehicle_countingbox.clear()
        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()


