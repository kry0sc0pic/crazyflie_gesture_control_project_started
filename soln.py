import tensorflow as tf
import cv2
import numpy as np
from collections import namedtuple
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp import init_drivers

# Constants
HEIGHT_LIMIT = 2.0 # maximum height of the drone
PREDICTION_THRESHOLD = 0.3 # confidence threshold for keypoints
UPPER_DIFFERENCE = 0.1 # maximum difference between the wrist & hip
LOWER_DIFFERENCE = 0.1 # minimum difference between the wrist & hip
URI = "radio://0/90/2M/E7E7E7E7E5"
CRAZYFLIE_ENABLED = False

# Movenet Keypoints
# order of the keypoints matches the order in the presentation diagram.
# used a named tuple makes the code easier to read since we can access the keypoints by name instead of using the index.
MOVENET_POINTS = [
"nose", 
"left_eye", 
"right_eye", 
"left_ear", 
"right_ear",
"left_shoulder",
"right_shoulder",
"left_elbow",
"right_elbow",
"left_wrist",
"right_wrist",
"left_hip",
"right_hip",
"left_knee",
"right_knee",
"left_ankle",
"right_ankle",
]
MovenetKeypoints = namedtuple('MovenetKeypoints',MOVENET_POINTS)

# Variables for KeyPoints
right_wrist = (0,0)
left_wrist = (0,0)
right_hip = (0,0)
left_hip = (0,0)


# Initialize the drivers
if CRAZYFLIE_ENABLED:
    init_drivers()


# Initialize the AI Model
interpreter = tf.lite.Interpreter(model_path='models/singlepose-lightning/3.tflite')
interpreter.allocate_tensors()
def movenet(input_image):
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

# Setup Camera Stream
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

success, frame = video_capture.read()
if not success:
    print("Error: Could not read frame.")
    exit()

# Dimensions of the frame
dim_y , dim_x, _ = frame.shape

# Connect to Crazyflie
if CRAZYFLIE_ENABLED:
    # Fill in here
    scf = SyncCrazyflie(URI)
    scf.open_link()
    commander = scf.cf.high_level_commander
    print("Connected to Crazyflie")
    print("Starting up the motors")
    commander.takeoff(0.5,2.0)

try:
    while success:
        tf_img = cv2.resize(frame,(192,192)) # Resize the frame to 192x192
        tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
        tf_img = np.asarray(tf_img) # Convert the frame to a numpy array
        tf_img = np.expand_dims(tf_img,axis=0) # Add a batch dimension
        image = tf.cast(tf_img,dtype=tf.int32) # Cast the image to int32

        output = movenet(image) # Get the keypoints from the model
        points = output[0][0] # Get the keypoints from the output
        movenet_keypoints = MovenetKeypoints(*points) # Create a named tuple for the keypoints


        # Check if the keypoints are above the prediction threshold
        # If the keypoints are above the prediction threshold, update the variables for the keypoints
        if movenet_keypoints.right_wrist[2] > PREDICTION_THRESHOLD:
            right_wrist = (movenet_keypoints.right_wrist[0], movenet_keypoints.right_wrist[1])

        if movenet_keypoints.left_wrist[2] > PREDICTION_THRESHOLD:
            left_wrist = (movenet_keypoints.left_wrist[0], movenet_keypoints.left_wrist[1])

        if movenet_keypoints.right_hip[2] > PREDICTION_THRESHOLD:
            right_hip = (movenet_keypoints.right_hip[0], movenet_keypoints.right_hip[1])

        if movenet_keypoints.left_hip[2] > PREDICTION_THRESHOLD:
            left_hip = (movenet_keypoints.left_hip[0], movenet_keypoints.left_hip[1])

        # Calculate the distance between the right wrist and the right hip
        right_diff = abs(right_hip[0] - right_wrist[0])

        # Calculate the distance between the left wrist and the left hip
        left_diff = abs(left_hip[0] - left_wrist[0])

        # Average both
        average_diff = (right_diff + left_diff) / 2
        
        # Add text to the output visual
        cv2.putText(frame,f"Av. Diff: {average_diff}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Constrain the average_diff to be between LOWER_DIFFERENCE and UPPER_DIFFERENCE
        average_diff = max(average_diff, LOWER_DIFFERENCE) # limit average_diff to be at least LOWER_DIFFERENCE
        average_diff = min(average_diff, UPPER_DIFFERENCE) # limit average_diff to be at most UPPER_DIFFERENCE

        # scale the average_diff to be between 0 and 1
        average_diff = average_diff / (UPPER_DIFFERENCE - LOWER_DIFFERENCE)

        # calculate desired height
        desired_height = max(0.5, HEIGHT_LIMIT * average_diff) # limit desired_height to be at least 0.5
        
        if CRAZYFLIE_ENABLED:
            # Fill in here
            commander.go_to(0,0,desired_height,0)
            pass


        # Add text to the output visual
        cv2.putText(frame,f"Desired Height: {desired_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        # Draw the keypoints on the frame
        for i,k in enumerate(points):

            # Checks confidence for keypoint
            if k[2] > PREDICTION_THRESHOLD:
                # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
                yc = int(k[0] * dim_y)
                xc = int(k[1] * dim_x)

                # Draws a circle on the image for each keypoint
                frame = cv2.circle(frame, (xc, yc), 2, (0, 255, 0), 5) # draw green circle at the keypoint   
            
        # Display the frame
        cv2.imshow("Pose Estimation",frame)
        
        if cv2.waitKey(1) == ord('q'):
            print("Exiting")
            break

        success, frame = video_capture.read()
except Exception as e:
    print(e)

finally:
    if CRAZYFLIE_ENABLED:
        # Fill in here
        print("Landing the drone")
        commander.land(0.0,2.0)
        scf.close_link()

video_capture.release()