import cv2

Face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

original_image = cv2.imread("Smiledetection.jpg")

# resize image
resizeimage = cv2.resize(original_image,(500,500))
# gray
grayimage = cv2.cvtColor(resizeimage,cv2.COLOR_BGR2GRAY)

Faces = Face_cascade.detectMultiScale(grayimage,scaleFactor=1.1,minNeighbors=5)

for(x,y,w,h) in Faces:
    roi_gray = grayimage[y:y + h, x:x + w]  # Region of Interest (Face)
    roi_color = resizeimage[y:y + h, x:x + w]

    # Detect smiles inside face region
    smiles = Smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.7,  # Reduce scale factor for better detection sensitivity
        minNeighbors=20,  # Reduce minNeighbors to detect subtle smiles
        minSize=(20, 20)  # Smaller size to detect small smiles
    )

    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

cv2.imshow("Smile Detection",resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()