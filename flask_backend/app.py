from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from indicators import *

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

@app.route('/api/price', methods=['GET'])
def get_price():
    ticker = request.args.get('ticker')
    dates, close_prices = get_prices(ticker)
    return {"dates": dates, "closePrices": close_prices}

@app.route('/api/ma', methods=['GET'])
def get_ma():
    ticker = request.args.get('ticker')
    close, high, low = get_stock_info(ticker)
    ma = ma(close, 20)
    return {"MA": ma}

@app.route('/api/ema', methods=['GET'])
def get_ema():
    ticker = request.args.get('ticker')
    close, high, low = get_stock_info(ticker)
    ema = ema(close, 20)
    return {"EMA": ema}
    

if __name__ == '__main__':
    app.run(debug=True)