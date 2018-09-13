from flask import Flask, abort, render_template, jsonify, request
from api import prophet_forecast_weight, nutritional_breakdown, get_features, example, df
from api import yummly_df, recipe_cosine_similarity, get_closest_recipes, food_recommendations, flask_output
import random

app = Flask('WeightLossApp')


@app.route('/predict', methods=['POST'])
def do_prediction():
    if not request.json:
        abort(400)
    data = request.json

    converted_data = {key: float(value) for key, value in data.items() if key != 'Date'}
    converted_data['Date'] = data['Date']

    fat_breakdown, protein_breakdown, carb_breakdown, calorie_breakdown = nutritional_breakdown(
        df, converted_data)

    features = get_features(converted_data, fat_breakdown, protein_breakdown,
                            carb_breakdown, calorie_breakdown)

    response = flask_output(features, df, yummly_df, converted_data, periods=90)

    return jsonify(response)


@app.route('/')
def index():
    return render_template('index.html')


app.run(debug=True)
