from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
app.test_client()
# Load your trained model
with open('coin_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        price = float(request.form['price'])
        volume = float(request.form['volume'])
        mkt_cap = float(request.form['mkt_cap'])
        change_1h = float(request.form['change_1h'])
        price_change_24h = float(request.form['change_24h'])
        price_change_7d = float(request.form['change_7d'])

        # Feature engineering
        mkt_cap_to_volume_ratio = mkt_cap / volume
        moving_average_mkt_cap_7d = mkt_cap
        std_dev_price_7d = price
        # Final features (order must match training)
        features = [[

            price,
            mkt_cap_to_volume_ratio,
            price_change_24h,
            price_change_7d,
            moving_average_mkt_cap_7d,
            std_dev_price_7d,
            # liquidity_ratio
        ]]
        # features = np.array(features).reshape(1, -1)
        # Predict
        prediction = model.predict(features)
        output = prediction[0] * 100

        return render_template('index.html', prediction_text=f'Predicted Value: {output:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
