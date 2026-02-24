import mediapipe as mp
import numpy as np
import cv2
import threading
import time
import struct
import base64
import global_vars

# Kalman filter
class KalmanPoint:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.is_initialized = False

    def filter(self, x, y):
        # If we receive (0,0) ignore them it's a junk
        if x == 0 and y == 0:
            return 0, 0

        measured = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.is_initialized:
            # Initialization both states: current and predicted
            initial_state = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.kf.statePost = initial_state
            self.kf.statePre = initial_state
            self.is_initialized = True
            return float(x), float(y)

        self.kf.correct(measured)
        predict = self.kf.predict()
        return float(predict[0]), float(predict[1])

    def reset(self):
        self.is_initialized = False

class CaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.ret = False
        self.frame = None
        self.isRunning = False

    def run(self):
        self.cap = cv2.VideoCapture(global_vars.WEBCAM_INDEX)
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_vars.HEIGHT)

        while not global_vars.KILL_THREADS:
            self.ret, self.frame = self.cap.read()
            self.isRunning = True
        self.cap.release()

class HandsThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.pipe = None
        self.filters = {"Left": KalmanPoint(), "Right": KalmanPoint()}

    def run(self):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        capture = CaptureThread()
        capture.start()

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            while not global_vars.KILL_THREADS and not capture.isRunning:
                time.sleep(0.5)

            while not global_vars.KILL_THREADS and capture.cap.isOpened():
                if not capture.ret: continue

                image = capture.frame.copy()
                image = cv2.flip(image, 1)
                h, w, _ = image.shape
                
                # MediaPipe processing
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                data_string = ""
                
                if results.multi_hand_landmarks:
                    for i, (hand_lms, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                        label = hand_info.classification[0].label # "Left" или "Right"
                        
                        # 1. Processing of the skeleton (HAND_VISUALIZE)
                        if getattr(global_vars, 'HAND_VISUALIZE', False):
                            nodes = []
                            for lm in hand_lms.landmark:
                                # Collect x,y for each of 21 pieces
                                nodes.append(f"{lm.x:.4f},{lm.y:.4f}")
                            data_string += f"HAND_NODES|{label}|{';'.join(nodes)}\n"
                        
                        
                        # Filter for hand mode from global_vars
                        if global_vars.DETECT_HANDS_MODE != "BOTH":
                            if label.upper() != global_vars.DETECT_HANDS_MODE:
                                continue

                        # Points for Thumb and Middle finger
                        t = hand_lms.landmark[4]
                        idx = hand_lms.landmark[8]
                        
                        dist = np.hypot(t.x - idx.x, t.y - idx.y)
                        is_pinched = dist < global_vars.PINCH_THRESHOLD

                        if is_pinched:
                            # Claculate center and making filter
                            raw_x, raw_y = (t.x + idx.x) / 2, (t.y + idx.y) / 2
                            sm_x, sm_y = self.filters[label].filter(raw_x * w, raw_y * h)
                            
                            # Build string for Unity: PINCH|HAND|X|Y
                            data_string += f"PINCH|{label}|{sm_x/w}|{sm_y/h}\n"

                        if global_vars.DEBUG:
                            mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
                            if is_pinched:
                                cv2.circle(image, (int(sm_x), int(sm_y)), 15, (0, 255, 0), -1)
                else:
                    # If there are no hands then reset filters
                    for f in self.filters.values():
                        f.reset()

                # Image adding to the pipe
                if global_vars.DEBUG:
                    ret, jpeg = cv2.imencode('.jpg', image)
                    if ret:
                        img_b64 = base64.b64encode(jpeg).decode('utf-8')
                        data_string += f"IMAGE|{img_b64}\n"
                    
                    # Image as itself. Comment it if you don't need it
                    cv2.imshow('Hands Tracking Pipe', image)
                    cv2.waitKey(1)

                self.send_to_unity(data_string)

        cv2.destroyAllWindows()

    def send_to_unity(self, message):
        if not message: return
        
        if self.pipe is None:
            try:
                self.pipe = open(r'\\.\pipe\UnityMediaPipeHands', 'wb', 0)
            except FileNotFoundError:
                self.pipe = None
                return

        try:
            s = message.encode('utf-8')
            self.pipe.write(struct.pack('I', len(s)) + s)
            self.pipe.seek(0)
        except Exception:
            self.pipe = None

if __name__ == "__main__":
    thread = HandsThread()
    thread.start()