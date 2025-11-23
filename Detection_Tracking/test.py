import cv2
import mediapipe as mp

# Initialize MediaPipe's solution for holistic body and pose detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Path to your input video file
INPUT_VIDEO_PATH = "Sequence 1.mp4"

# Use MediaPipe Holistic to detect people (it detects a pose which defines a person)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Convert the BGR image to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get results
        results = holistic.process(image)

        # Convert the image color back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks (which implies a detected person)
        mp_drawing.draw_landmarks(image, 
                                  results.pose_landmarks, 
                                  mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        # NOTE: MediaPipe's primary focus is landmarks/pose. 
        # For a simple rectangular bounding box, the first OpenCV method is better.

        cv2.imshow('MediaPipe Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()