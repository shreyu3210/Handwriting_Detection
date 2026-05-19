import cv2

# The URL that we KNOW works from your web browser test
url = "http://admin:sis9724123@192.168.29.5/cgi-bin/mjpg/video.cgi?channel=1&subtype=1"

print(f"Attempting to connect to: {url.replace('sis9724123', '***')}")

# Notice we DO NOT use cv2.CAP_FFMPEG here. 
# We let OpenCV automatically choose the best HTTP reader.
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: OpenCV could not open the MJPEG stream.")
else:
    print("Success! Stream opened.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Stream dropped.")
            break
        
        cv2.imshow('CP Plus Feed', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()