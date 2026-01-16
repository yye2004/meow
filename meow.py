import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cam = cv2.VideoCapture(0)

# --- Tune these thresholds for YOUR camera/face distance ---
EYE_WIDE_TH = 0.028      # eyes wide
EYE_SHALLOW_TH = 0.018   # eyes shallow (squint-ish)
MOUTH_OPEN_TH = 0.030    # mouth open
MOUTH_CLOSE_TH = 0.016   # mouth closed (use < this)


def eye_opening(face):
    # eyelid points (FaceMesh)
    l_top, l_bot = face.landmark[159], face.landmark[145]
    r_top, r_bot = face.landmark[386], face.landmark[374]
    return (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0


def mouth_opening(face):
    top_lip = face.landmark[13]
    bottom_lip = face.landmark[14]
    return abs(top_lip.y - bottom_lip.y)


def is_index_finger_up(hand_lms):
    """
    Simple ☝🏻 detector:
    - index finger extended (tip above pip above mcp)
    - other fingers folded (tips below their pips)
    Note: y goes DOWN as value increases; "up" means smaller y.
    """
    lm = hand_lms.landmark

    # Index extended
    index_up = (lm[8].y < lm[6].y < lm[5].y)

    # Other fingers folded (tip below pip)
    middle_folded = (lm[12].y > lm[10].y)
    ring_folded   = (lm[16].y > lm[14].y)
    pinky_folded  = (lm[20].y > lm[18].y)

    return index_up and middle_folded and ring_folded and pinky_folded


def main():
    while True:
        ret, image = cam.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_result = face_mesh.process(rgb)
        hand_result = hands.process(rgb)

        face_list = face_result.multi_face_landmarks
        hand_list = hand_result.multi_hand_landmarks

        # Default cat if nothing detected
        cat_image = "assets/cat-default.jpeg"

        face = None
        if face_list:
            face = face_list[0]

        warning_gesture = False
        if hand_list:
            warning_gesture = is_index_finger_up(hand_list[0])

        if face:
            e = eye_opening(face)
            m = mouth_opening(face)

            # Priority order:
            # 1) warning (requires face + ☝🏻)
            # 2) shock (eyes wide + mouth open)
            # 3) shut (eyes big + mouth closed)
            # 4) default (mouth open + eyes shallow)
            if warning_gesture:
                cat_image = "assets/cat-warning.jpeg"
            elif (e > EYE_WIDE_TH) and (m > MOUTH_OPEN_TH):
                cat_image = "assets/cat-shock.jpeg"
            elif (e > EYE_WIDE_TH) and (m < MOUTH_CLOSE_TH):
                cat_image = "assets/cat-shut.jpeg"
            elif (m > MOUTH_OPEN_TH) and (e < EYE_SHALLOW_TH):
                cat_image = "assets/cat-default.jpeg"
            else:
                cat_image = "assets/cat-default.jpeg"

            # Optional: draw face landmarks
            for lm in face.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(image, (x, y), 1, (0, 100, 0), -1)

            # Optional: show live measurements to help tune thresholds
            cv2.putText(image, f"eye:{e:.4f} mouth:{m:.4f} warn:{int(warning_gesture)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Face Detection", image)

        cat = cv2.imread(cat_image)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
            cv2.imshow("Cat Image", cat)
        else:
            blank = image * 0
            cv2.putText(blank, f"Missing: {cat_image}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cat Image", blank)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
