import pytest
import os
from app import app


@pytest.fixture
def client():
    app.config["Testing"] = True
    ## Disable logging of errors during testing to keep the output clean
    app.config["DEBUG"] = False
    with app.test_client() as client:
        yield client


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200


def artifacts_are_valid():
    model_path = "artifacts/model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        return os.path.getsize(model_path) > 1024 * 1024
    return False


@pytest.mark.skipif(not artifacts_are_valid(), reason="Real Model Files Not Found")
def test_prediction_endpoint(client):
    test_data = {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001,
        "mean_smoothness": 0.118,
        "mean_compactness": 0.278,
        "mean_concavity": 0.300,
        "mean_concave_points": 0.147,
        "mean_symmetry": 0.242,
        "mean_fractal_dimension": 0.078,
        "radius_se": 1.095,
        "texture_se": 0.9053,
        "perimeter_se": 8.589,
        "area_se": 153.4,
        "smoothness_se": 0.006399,
        "compactness_se": 0.04904,
        "concavity_se": 0.05373,
        "concave_points_se": 0.01587,
        "symmetry_se": 0.03003,
        "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38,
        "texture_worst": 17.33,
        "perimeter_worst": 184.6,
        "area_worst": 2019,
        "smoothness_worst": 0.1622,
        "compactness_worst": 0.6656,
        "concavity_worst": 0.7119,
        "concave_points_worst": 0.2654,
        "symmetry_worst": 0.4601,
        "fractal_dimension_worst": 0.1189,
    }
    response = client.post("/predict", data=test_data, follow_redirects=True)
    assert response.status_code == 200
