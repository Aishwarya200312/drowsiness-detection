import cv2
import mediapipe as mp
import numpy as np
import winsound  # for alert sound

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices
MOUTH = [13, 14, 78, 308]

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate MAR (Mouth Aspect Ratio)
def calculate_mar(mouth):
    vertical = np.linalg.norm(mouth[0] - mouth[1])
    horizontal = np.linalg.norm(mouth[2] - mouth[3])
    return vertical / horizontal

# Thresholds
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
MAR_THRESHOLD = 0.6

counter = 0
yawn_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            left_eye = []
            right_eye = []
            mouth = []

            # LEFT EYE
            for idx in LEFT_EYE:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                left_eye.append([x, y])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # RIGHT EYE
            for idx in RIGHT_EYE:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                right_eye.append([x, y])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # MOUTH
            for idx in MOUTH:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                mouth.append([x, y])
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Convert to numpy arrays
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            mouth = np.array(mouth)

            # Calculate EAR & MAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = calculate_mar(mouth)

            # Display EAR & MAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # DROWSINESS DETECTION
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(1000, 1000)
            else:
                counter = 0

            # YAWNING DETECTION
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter > 15:
                    cv2.putText(frame, "YAWNING DETECTED!", (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                    winsound.Beep(1500, 1000)
            else:
                yawn_counter = 0

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()