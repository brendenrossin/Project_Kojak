from flask import Flask, abort, render_template, jsonify, request
from api import prophet_forecast_weight, nutritional_breakdown, get_features, example, df

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

    response = prophet_forecast_weight(features, df, periods=90, date=converted_data['Date'])

    return jsonify(response)


@app.route('/')
def index():
    return render_template('index.html')


app.run(debug=True)
