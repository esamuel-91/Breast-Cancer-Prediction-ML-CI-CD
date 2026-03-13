from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from src.exception import CustomException
from src.logger import logger
import sys

application = Flask(__name__)
app = application

@app.route("/")
def homepage():
    """
    This will return the homepage for the current project.
    """
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error("Exception occured while trying to return the homepage.")
        raise CustomException(e,sys)
    
@app.route("/predict",methods=["GET","POST"])
def predict():
    """
    This function is used to make prediction.
    """
    if request.method == "GET":
            return render_template("form.html")
    
    else:
        try:
            # Mean features
            radius_mean = float(request.form.get("radius_mean"))
            texture_mean = float(request.form.get("texture_mean"))
            perimeter_mean = float(request.form.get("perimeter_mean"))
            area_mean = float(request.form.get("area_mean"))
            smoothness_mean = float(request.form.get("smoothness_mean"))
            compactness_mean = float(request.form.get("compactness_mean"))
            concavity_mean = float(request.form.get("concavity_mean"))
            concave_points_mean = float(request.form.get("concave_points_mean"))
            symmetry_mean = float(request.form.get("symmetry_mean"))
            fractal_dimension_mean = float(request.form.get("fractal_dimension_mean"))

            # Standard error features
            radius_se = float(request.form.get("radius_se"))
            texture_se = float(request.form.get("texture_se"))
            perimeter_se = float(request.form.get("perimeter_se"))
            area_se = float(request.form.get("area_se"))
            smoothness_se = float(request.form.get("smoothness_se"))
            compactness_se = float(request.form.get("compactness_se"))
            concavity_se = float(request.form.get("concavity_se"))
            concave_points_se = float(request.form.get("concave_points_se"))
            symmetry_se = float(request.form.get("symmetry_se"))
            fractal_dimension_se = float(request.form.get("fractal_dimension_se"))

            # Worst features
            radius_worst = float(request.form.get("radius_worst"))
            texture_worst = float(request.form.get("texture_worst"))
            perimeter_worst = float(request.form.get("perimeter_worst"))
            area_worst = float(request.form.get("area_worst"))
            smoothness_worst = float(request.form.get("smoothness_worst"))
            compactness_worst = float(request.form.get("compactness_worst"))
            concavity_worst = float(request.form.get("concavity_worst"))
            concave_points_worst = float(request.form.get("concave_points_worst"))
            symmetry_worst = float(request.form.get("symmetry_worst"))
            fractal_dimension_worst = float(request.form.get("fractal_dimension_worst"))

            data = CustomData(
                 radius_mean, 
                 texture_mean, 
                 perimeter_mean, 
                 area_mean, 
                 smoothness_mean, 
                 compactness_mean, 
                 concavity_mean, 
                 concave_points_mean, 
                 symmetry_mean, 
                 fractal_dimension_mean, 
                 radius_se, 
                 texture_se, 
                 perimeter_se, 
                 area_se, 
                 smoothness_se, 
                 compactness_se, 
                 concavity_se, 
                 concave_points_se, 
                 symmetry_se, 
                 fractal_dimension_se, 
                 radius_worst, 
                 texture_worst, 
                 perimeter_worst, 
                 area_worst, 
                 smoothness_worst, 
                 compactness_worst, 
                 concavity_worst, 
                 concave_points_worst, 
                 symmetry_worst, 
                 fractal_dimension_worst
            )

            pred_df = data.gather_data_as_dataframe()

            ##Prediction Pipeline
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(pred_df)

            result = "Malignant Tumor" if prediction[0] == 0 else "Benign Tumor"

            return render_template("result.html", final_result=result)

        except Exception as e:
            logger.error(f"Exception occured while making preiction through Flask app: {e}")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)