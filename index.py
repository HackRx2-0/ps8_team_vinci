import cv2
import emotion_detection

def checkBlur(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	var = cv2.Laplacian(gray, cv2.CV_64F).var()

	if var>100:
		return "The Image is Blurred"

	else:
		return "The Image is Not Blurred"



def detectFace(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	return faces


def countFaces(faces):
	count = 0
	for (x,y,w,h) in faces:
		count+=1

	if count>1:
		return "There are more than one people"

	else:
		return "Only one face (Image Accepted)"



def cropFace(faces, image):
	for (x,y,w,h) in faces:
		roi_color = image[y:y + h, x:x + w]
		cv2.imwrite("Thumnail1.jpg", roi_color)
		return "Thumnail Created and Saved to Local Directory"


def calFaceArea(faces,image):
	for (x,y,w,h) in faces:
		faceArea = w * h

	h1, w1, _ = image.shape

	imgArea = h1 * w1

	# print("FaceArea:",faceArea)
	# print("ImageArea:", imgArea)

	percent = ((imgArea - faceArea) / imgArea) * 100

	if percent < 20:
		return "Face area is less than 20 percent (Upload Better Photo)"

	else:
		return "Face area greater than 20 percent (Test Case Passed)"


def mouthHindarance(image, faces):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mouth_cascade = cv2.CascadeClassifier('F://HackrX//Project//mouth.xml')
	mouth_feature = mouth_cascade.detectMultiScale(gray, 1.5, 5)
	for (x,y,w,h) in faces:
		if(len(mouth_feature)==0):
			return "Face Region is Not Shown Completely or Hindered"

		else:
			for (mx, my, mw, mh) in mouth_feature:
				if(y < my < y + h):
					return "Face Region is Clear (Test Case Passed)"



def faceEmotion(path):
	label = emotion_detection.emotion(path)
	if label =="Happy" or label == "Neutral":
		case = label + "  (Experssion Accepted)"
		return case

	else:
		return label + "  (Inappropriate Experssion)"










path = "faceWithMask.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)


faces = detectFace(img)
noOfFacetest = countFaces(faces)


# thumnail = cropFace(faces, img)

regionArea = calFaceArea(faces, img)
print(regionArea)

mouthArea = mouthHindarance(img, faces)
print(mouthArea)

# emotion = faceEmotion(path)
# print(emotion)


# print(noOfFacetest)

# blurTest = checkBlur(img)
# print(blurTest)
