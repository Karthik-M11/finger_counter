import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
base_options = python.BaseOptions(model_asset_path=r"hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# the tip ids for the fingers (thumb, index, middle, ring, pinky)
tip_ids = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()

    if not ret:
        # if camera fails, try again instead of crashing
        print("Camera failed, retrying...")
        continue  

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # the total number of fingers up
        total = 0

        result = detector.detect(mp_image)
        
        # if hand is found, then loop through the landmarks and check which fingers are up
        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                h, w, _ = frame.shape
                lm_list = []

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                # if we don't have all 21 landmarks, skip this hand
                if len(lm_list) < 21:
                    continue

                fingers = []

                # get the hand label (left or right)
                hand_label = result.handedness[idx][0].category_name

                # thumb
                if hand_label == "Right":
                    fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
                else:
                    fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)

                # threshold for finger being up (distance between tip and pip)
                threshold = 20

                for i in range(1, 5):
                    if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1] - threshold:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total += fingers.count(1)

        cv2.putText(frame, f'Fingers: {total}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    except Exception as e:
        print("Frame error:", e)
        continue   # don't crash, skip bad frame

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()