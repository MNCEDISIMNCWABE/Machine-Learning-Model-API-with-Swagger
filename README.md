# Building an API for a Machine Learning model with Swagger

Your Machine Learning Model is Accurate, Now What?. 
The ability to deploy these models as APIs that end users can interact with is just as crucial. In this repo we explore the process of creating an API for a machine learning model using [Swagger,](https://swagger.io/) demonstrating how to share and integrate this endpoint into a production environment so that your model doesn't just end in a Juputer notebook. We'll also test the API and highlight the benefits of such an approach.


### Scenario: Predicting Purchase Probability
Let's consider a scenario where we have built a machine learning model to predict the probability of a customer making a purchase of a product. We'll generate synthetic data for this example, create the model, and then build an API for it. In this case, you can think of it as a model that helps marketing teams to identify potential customers who are more likely to make a purchase, allowing them to target their campaigns more effectively. In the front-end of the API, end-users such as marketing teams can provide values for customer features like income, savings, age, credit score, and whether a customer subscribed to a marketing campaign before. The model will then return the probabilities of a customer making a purchase.

#### Step 1: Generate Synthetic Data
First, let's generate synthetic data:


``` # Generate synthetic data
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

