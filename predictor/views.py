from django.shortcuts import render
import joblib, os
import numpy as np

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "forest_fire_model.pkl")
model = joblib.load(model_path)

def home(request):
    result = ""
    if request.method == "POST":
        # get values from form
        temp = float(request.POST.get("temp"))
        humidity = float(request.POST.get("humidity"))
        wind = float(request.POST.get("wind"))
        rain = float(request.POST.get("rain"))

        # arrange input as 2D array for model
        input_data = np.array([[temp, humidity, wind, rain]])

        # make prediction
        prediction = model.predict(input_data)[0]

        # interpret result
        if prediction == 1:
            result = "🔥 High Risk of Forest Fire"
        else:
            result = "✅ Low Risk of Forest Fire"

    return render(request, "predictor/home.html", {"result": result})
