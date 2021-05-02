# Cus-Be-An
A service that can perform customer churn and behavior prediction and help companies save millions in revenue

#See the live demo at:https://cus-bean.herokuapp.com/

# Objective
In E-Commerce or startups, the relationship between businesses and customers are non-contractual relationship. In the non-contractual world, customers do go away, but they do so silently; they have no need to tell us they are leaving. This makes for a much trickier CLV calculation. We have to look at the time since a customer’s last transaction and ask a question: is the customer alive but dormant, or is the customer “dead” (“alive” means customers interact with us, “die” means they become inactive as customers)?

When it comes to finding out who your best customers are, the old RFM matrix principle is the best. RFM stands for Recency, Frequency and Monetary. It is a customer segmentation technique that uses past purchase behavior to divide customers into groups.

Can you develop a model of machine learning that can predict customers who will leave the company?

# MODELS:
# .BG-NBD Model for Customer Base Analysis
# .RFM MODEL 
RFM Score Calculations
RECENCY (R): Days since last purchase
FREQUENCY (F): Total number of purchases
MONETARY VALUE (M): Total money this customer spent
#STEPS OF K-MEAN ALGORITHM

Step 1: Choose the number of clusters k

Step 2: Select k random points from the data as centroids

Step 3: Assign all the points to the closest cluster centroid

Step 4: Recompute the centroids of newly formed clusters

Step 5: Repeat steps 3 and 4

Stopping Criteria for K-Means Clustering

There are essentially three stopping criteria that can be adopted to stop the K-means algorithm:

1.Centroids of newly formed clusters do not change

2.Points remain in the same cluster

3.Maximum number of iterations are reached

# Details about the dataset:
It consists of 10000 observations and 12 variables. Independent variables contain information about customers. Dependent variable refers to customer abandonment status.

#Tool(s) Used - 
Python (Jupyter Notebook)
Numpy
matplotlib 
chart_studio
pandas
seaborn 
streamlit 
plotly 
numpy 
scikit_learn
datetime
catboost
xgboost
lightgbm


