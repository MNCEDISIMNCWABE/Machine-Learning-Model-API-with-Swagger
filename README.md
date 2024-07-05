# Building an API for a Machine Learning model with Swagger

Your Machine Learning Model is Accurate, Now What?. 
The ability to deploy these models as APIs that end users can interact with is just as crucial. In this repo we explore the process of creating an API for a machine learning model using [Swagger,](https://swagger.io/) demonstrating how to share and integrate this endpoint into a production environment so that your model doesn't just end in a Juputer notebook. We'll also test the API and highlight the benefits of such an approach.


### Scenario: Predicting Purchase Probability
Let's consider a scenario where we have built a machine learning model to predict the probability of a customer making a purchase of a product. We'll generate synthetic data for this example, create the model, and then build an API for it. In this case, you can think of it as a model that helps marketing teams to identify potential customers who are more likely to make a purchase, allowing them to target their campaigns more effectively. In the front-end of the API, end-users such as marketing teams can provide values for customer features like income, savings, age, credit score, and whether a customer subscribed to a marketing campaign before. The model will then return the probabilities of a customer making a purchase.
