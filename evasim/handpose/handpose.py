import cv2
import time
import handpose.HandTrackingModule as htm
from sys import argv, path
# import paho.mqtt.client as mqtt
import argparse
from handpose.poses import poses, defined_poses, thumbs_orientation

import os.path

path.append(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-q', '--quiet', action='store_true')
# parser.add_argument('max_hands', type=int)
parser.add_argument('trigger_pose', default=None, nargs="?")

args = parser.parse_args()

DEBUG = args.debug
QUIET = args.quiet
MAX_HANDS = 10 #args.max_hands
TRIGGER_POSE = args.trigger_pose
if TRIGGER_POSE: TRIGGER_POSE = TRIGGER_POSE.upper()

if MAX_HANDS < 1:
    print("'max_hands' must be at least 1")
    exit()

if not TRIGGER_POSE in defined_poses:
    print("unknown 'trigger_pose'. proceeding as if it was not defined.")
    TRIGGER_POSE = None

broker_address = "localhost" #input("Broker address: ")

print("Attempting connection...")

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    pass
    # print("Connected with result code "+str(rc))

# client = mqtt.Client()
# client.on_connect = on_connect

# client.connect(broker_address, 1883, 60)

# print(f"Connected to {broker_address}:1883")

def indexOf(finger, distance=0):
    if finger < 1 or finger > 5 or distance < 0 or distance > 3: return -1
    return (finger*4)-distance

def dist(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
def is_closed(finger, lmList):
    if finger == 1:
        # d1 = tip of thumb to ring finger knuckle
        # d2 = tip of thumb to thumb knuckle
        d1 = dist(lmList[4][1], lmList[4][2], lmList[13][1], lmList[13][2])
        d2 = dist(lmList[4][1], lmList[4][2], lmList[2][1], lmList[2][2])
    else:
        # d1 = tip to wrist 
        # d2 = proximal joint to wrist
        d1 = dist(lmList[indexOf(finger)][1], lmList[indexOf(finger)][2], lmList[0][1], lmList[0][2])
        d2 = dist(lmList[indexOf(finger, 2)][1], lmList[indexOf(finger, 2)][2], lmList[0][1], lmList[0][2])
    return d1 < d2


detector = htm.handDetector(maxHands=MAX_HANDS, detectionCon=0.75)

__UNDEF = "undefined"

def getPose(fingers, lmList):
    totalFingers = fingers.count(True)
    for pose in poses[totalFingers]:
        if poses[totalFingers][pose] == fingers:
            if pose == "THUMB":
                return "THUMBS_" + thumbs_orientation(lmList)
            return pose
    return __UNDEF

hands = dict()
last_results = dict()
for i in range(MAX_HANDS):
    hands[i] = dict()
    hands[i][__UNDEF] = 0
    for p in poses:
        for k in p:
            hands[i][k] = 0
    hands[i]["THUMBS_UP"] = 0
    hands[i]["THUMBS_DOWN"] = 0
    last_results[i] = __UNDEF

def reset_hand(handNo):
    hand = hands[handNo]
    for pose in hand:
        hand[pose] = 0
    hand["THUMBS_UP"] = 0
    hand["THUMBS_DOWN"] = 0

def evaluate(g):
    max_key = max(g, key=g.get)
    return max_key

def read_hand(img, handNo=0):
    
    lmList = detector.findPosition(img, handNo=handNo, draw=False)

    if len(lmList) != 0:
        
        fingers = []
        for i in range(1,6):
            fingers.append(not is_closed(i, lmList))
        hands[handNo][getPose(fingers, lmList)] += 1
        #print(fingers, getPose(fingers))

def publish_result(result):
    # client.publish("handpose_recog", result)
    if not QUIET:
        print(f"-> Published '{result}'")
    last_results[i] = result


def run():

    print("Teste")

    wCam, hCam = 640, 480

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0

    INTERVAL = 3
        
    timer = time.time()
    run = True
    response = None
    try:
        
        while run:

            # client.loop(timeout=0.05)

            success, img = cap.read()
            numHands, img = detector.findHands(img)

            for i in range(numHands):
                read_hand(img, handNo=i)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            t = time.time() - timer
            
            if t > INTERVAL:

                is_trigger_up = False
                trigger_hand = -1

                for i in range(numHands):
                    result = evaluate(hands[i])
                    if result == TRIGGER_POSE and not is_trigger_up:
                        is_trigger_up = True
                        trigger_hand = i
                        reset_hand(i)
                        break

                for i in range(numHands):
                    if i == trigger_hand:
                        continue
                    hand = hands[i]
                    result = evaluate(hand)
                    # if debug:
                    #     print(is_trigger_up, result)
                    if result != __UNDEF:
                        if not TRIGGER_POSE or (TRIGGER_POSE and is_trigger_up):
                            # publish_result(result)
                            response = result
                            run = False
                    reset_hand(i)
                    timer += t
                
            img=cv2.flip(img,1)

            cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
                break

    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()
    return response

if __name__ == "__main__":
    run()