from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
swagger = Swagger(app)

# Load the trained model
model = joblib.load('predict_purchase_model.pkl')

@app.route('/')
def predict_purchase_home_page():
    """
    Example endpoint returning a simple greeting
    ---
    responses:
      200:
        description: A simple greeting message
        examples:
          text: Welcome message.
    """
    return 'Welcome to purchase probability prediction!'

@app.route('/purchase_predict', methods=['POST'])
def predict():
    """
    Predict if a user will buy a product.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            income:
              type: float
              description: Customer income
            savings:
              type: float
              description: Customer savings
            subscribed_to_marketing:
              type: integer
              description: Whether a customer subscribed to marketing or not
            age:
              type: integer
              description: Customer's age
            work_experience_years:
              type: integer
              description: Customer's work experience
            credit_score:
              type: integer
              description: Customer's credit score
          example:
            income: 58951
            savings: 60670
            subscribed_to_marketing: 1
            age: 36
            work_experience_years: 10
            credit_score: 700
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: string
              description: Prediction result
              example: "Purchase"
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message
              example: "Error message"
    """
    try:
        data = request.get_json(force=True)
        features = pd.DataFrame([{
            'income': data['income'],
            'savings': data['savings'],
            'subscribed_to_marketing': data['subscribed_to_marketing'],
            'age': data['age'],
            'work_experience_years': data['work_experience_years'],
            'credit_score': data['credit_score'],
            'subscribed_to_marketing': data['subscribed_to_marketing']
        }])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        prediction_label = "Purchase" if int(prediction[0]) == 1 else "No Purchase"
        return jsonify({'prediction': prediction_label, 'probability': probability})
    except KeyError as e:
        return jsonify({'error': f"Missing parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
