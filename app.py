from flask import Flask, request, render_template 
import cv2
import index
import dl_models

app = Flask(__name__)   


@app.route('/', methods =["GET", "POST"])
def home():
	result = "Test Case Result"
	return render_template("index.html", result=result)
 


@app.route('/test', methods =["GET", "POST"])
def test1():
	result = "Test Case Result"
	if request.method == "POST":
		path = request.form["path"]
		tc = request.form["testcase"]

		img = cv2.imread(path, cv2.IMREAD_COLOR)
		faces = index.detectFace(img)

		if tc == "blur":
			img = cv2.imread(path, cv2.IMREAD_COLOR)
			result = index.checkBlur(img)
			return render_template("index.html", result = result)

		if tc == "count":
			faces = index.detectFace(img)
			result = index.countFaces(faces)
			return render_template("index.html", result = result)


		if tc == "thumnail":
			result = index.cropFace(faces, img)
			return render_template("index.html", result = result)

		if tc == "area":
			result = index.calFaceArea(faces,img)
			return render_template("index.html", result = result)


		if tc == "mouth":
			result =  index.mouthHindarance(img, faces)
			return render_template("index.html", result = result)

		if tc == "exp":
			result = index.faceEmotion(path)
			return render_template("index.html", result = result)

		if tc == "car":
			predict = dl_models.get_results(path)
			ans = predict[0]

			if ans == "real":
				result = "The Image is not a Cartoon. Its a Real Image"
			
			else:
				result = "The Face Image is a Cartoon Image"

			return render_template("index.html", result = result)


		if tc == "wat":
			predict = dl_models.get_results(path)
			ans = predict[1]

			if ans == "watermarked":
				result = "The Image is Watermarked (Rejected)"
			
			else:
				result = "No Watermarks Identified"

			return render_template("index.html", result = result)

		if tc == "liv":
			predict = dl_models.get_liveliness(path)
			ans = predict

			if ans == "lively":
				result = "The Image is of a Real Person"
			
			else:
				result = "The image is not lively."

			return render_template("index.html", result = result)









	return render_template("index.html", result=result)
if __name__=='__main__':
   app.run(threaded=False)