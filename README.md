# Building an API for a Machine Learning model with Swagger

<img width="1433" alt="image" src="https://github.com/MNCEDISIMNCWABE/Machine-Learning-Model-API-with-Swagger/assets/67195600/f1f743a9-f3e5-4b19-acd8-2bdeef8b904c">


Your Machine Learning Model is Accurate, Now What?. 
The ability to deploy these models as APIs that end users can interact with is just as crucial. In this repo we explore the process of creating an API for a machine learning model using [Swagger,](https://swagger.io/) demonstrating how to share and integrate this endpoint into a production environment so that your model doesn't just end in a Juputer notebook. We'll also test the API and highlight the benefits of such an approach.


### Scenario: Predicting Purchase Probability
Let's consider a scenario where we have built a machine learning model to predict the probability of a customer making a purchase of a product. We'll generate synthetic data for this example, create the model, and then build an API for it. In this case, you can think of it as a model that helps marketing teams to identify potential customers who are more likely to make a purchase, allowing them to target their campaigns more effectively. In the front-end of the API, end-users such as marketing teams can provide values for customer features like income, savings, age, credit score, and whether a customer subscribed to a marketing campaign before. The model will then return the probabilities of a customer making a purchase.
The target variblae is ```purchase = 1 for a customer who makes a purchase, 0 for No purchase.```

#### 1: Generate Synthetic Data
First, let's generate synthetic data:

```
import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 4500

income = np.random.normal(50000, 15000, num_samples)
savings = np.random.normal(10000, 5000, num_samples)
subscribed_to_marketing = np.random.choice([0, 1], num_samples)
age = np.random.randint(20, 60, num_samples)
work_experience_years = np.random.randint(1, 20, num_samples)
credit_score = np.random.normal(700, 50, num_samples).astype(int)

# Target variable
purchase = (income + savings + np.random.randn(num_samples) * 10000 + 5000 * subscribed_to_marketing) > 70000
purchase = purchase.astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'income': income,
    'savings': savings,
    'subscribed_to_marketing': subscribed_to_marketing,
    'age': age,
    'work_experience_years': work_experience_years,
    'credit_score': credit_score,
    'purchase': purchase
})

data.head()
 ```

<img width="782" alt="image" src="https://github.com/MNCEDISIMNCWABE/Machine-Learning-Model-API-with-Swagger/assets/67195600/4fc0b008-07a8-4068-9b96-0fa04f359bcd">

#### 2: Build a Simple Logistic Regression model

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def model_application(X_train, y_train):
    lr_clf = LogisticRegression(max_iter=100)
    lr_clf.fit(X_train, y_train)

    return lr_clf

lr_clf = model_application(X_train, y_train)
                 
def generate_perfomance_metrics(y_test,y_pred):
    rf_model_score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', rf_model_score)
    return print('Classification Report:\n', classification_report(y_test, y_pred))

generate_perfomance_metrics(y_test,y_pred)
```
<img width="436" alt="image" src="https://github.com/MNCEDISIMNCWABE/Machine-Learning-Model-API-with-Swagger/assets/67195600/d3a671ad-8283-45c0-941f-253682efb636">

### 3: Save the trained logistic model as a pickle file using the code below:
```
def save_model(model, filename='predict_purchase_model.pkl', folder='/path/to/save/model/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    file_path = os.path.join(folder, filename)
    joblib.dump(model, file_path)
    print(f'Model saved to {file_path}')

# Save the logistic regression model
save_model(lr_clf)
```

### 4: Build an API for the model using Flask and Swagger
- Install ```flassger``` and ```flask``` using ```pip install flasgger flask```.
- Create a file named ```app.py``` in your project directory and add the following code: This file calls your trained model ```predict_purchase_model.pkl``` and use it to predict probability of a customer to make a purchase.
```
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
    Precict purchase home page.
    ---
    responses:
      200:
        description: A simple welcome message.
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
```
### 5: Run the API
In your project directory where ```app.py``` is saved, run : ```python app.py```

When the app is running you can access the Swagger API endpoint at http://127.0.0.1:5009/apidocs/ to test the API and see the documentation.
Example:
<img width="1433" alt="image" src="https://github.com/MNCEDISIMNCWABE/Machine-Learning-Model-API-with-Swagger/assets/67195600/587a241c-5ef4-4d0c-bf37-8a8436355d06">

Here you can then provice values for customer features like income, savings, age, credit score, and whether a customer subscribed to a marketing campaign before, to predict their probability of purchasing a product.
