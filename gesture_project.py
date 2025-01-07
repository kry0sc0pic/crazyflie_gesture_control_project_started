import tensorflow as tf
import cv2
import numpy as np

HEIGHT_LIMIT = 2.0
PREDICTION_THRESHOLD = 0.3

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


while success:
    tf_img = cv2.resize(frame,(192,192)) # Resize the frame to 192x192
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
    tf_img = np.asarray(tf_img) # Convert the frame to a numpy array
    tf_img = np.expand_dims(tf_img,axis=0) # Add a batch dimension
    image = tf.cast(tf_img,dtype=tf.int32) # Cast the image to int32

    output = movenet(image) # Get the keypoints from the model

    for i,k in enumerate(output[0,0,:,:]):

        # Checks confidence for keypoint
        if k[2] > PREDICTION_THRESHOLD:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * dim_y)
            xc = int(k[1] * dim_x)

            # Draws a circle on the image for each keypoint
            frame = cv2.circle(frame, (xc, yc), 2, (0, 255, 0), 5)
         
        
    # Display the frame
    cv2.imshow("Pose Estimation",frame)
    
    if cv2.waitKey(1) == ord('q'):
        print("Exiting")
        break

    success, frame = video_capture.read()

video_capture.release()