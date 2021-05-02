import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import chart_studio as cs
import plotly.offline as po
import plotly.graph_objs as gobj
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


header = st.beta_container()
data = st.beta_container()


header1 = st.beta_container()
data1 = st.beta_container()




with header:
    st.title('Welcome to Customer Lifetime Value')
    st.image('clv.png')


	


with data:

	# try:
	#     with open(filename) as input:
	#         st.text(input.read())
	# except FileNotFoundError:
	#     st.error('File not found.')
    st.header('Online Reatil Data')
    st.subheader('Customer Lifetime Value (CLV) = how much a company expects to earn from an average customer in a life time')
    st.text('''CLV allows to benchmark customers and identify how much money the company can afford to spend on customer acquisition.
Historical CLV = the sum of revenues of all customer transactions multiplied by average or product-level profit margin
Problems with historical CLV:
doesn't account for customer tenure, retention or churn (e.g. if the company is growing its customer base, historical CLV will be deflated due to short tenure) doesn't account for new customers and their future revenue Basic CLV = Average Revenue Profit Margin Average Lifespan (where Average Lifespan is e.g. average time before customer churn)
Granular CLV = (Average Revenue per Transaction Average Frequency Profit Margin) * Average Lifespan (where Avg Frequency is within the certain timeframe, e.g. a month) accounts for each transaction
Traditional CLV = (Average Revenue Profit Margin) Retention Rate / Churn Rate (where Churn = 1 - Retention Rate) Retention/Churn - a proxy of expected length of customer lifespan with the company account for customer loyalty assumes that churn is final and customers do not return (especially critical for non-contractual business models)
Because we don't have profit margin, we will calculate revenue-based CLV.
But before calculating Basic, Granular and Traditional CLV, we will load, explore the data and then calculate retention rates using cohort analysis. We will need retention rates and churn rates for calculating Traditional CLV later.''')
    st.subheader('CLV Model Definition')
    st.text('''For the CLV models, the following nomenclature is used:
Frequency represents the number of repeat purchases the customer has made. This means that it’s one less than the total number of purchases.
T represents the age of the customer in whatever time units chosen (daily, in our dataset). This is equal to the duration between a customer’s first purchase and the end of the period under study.
Recency represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer’s first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.) ''')
    st.image('clv2.png')
    st.text_input('Enter file path:')
    st.text('I found the dataset on kaggle')
    Rtl_data = pd.read_csv('data/OnlineRetail.csv', encoding = 'unicode_escape')
    st.write(Rtl_data.head())

    country_cust_data=Rtl_data[['Country','CustomerID']].drop_duplicates()
    countries = country_cust_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
    st.text('The data is concentrated around UK')
    dist = pd.DataFrame(country_cust_data['Country'].value_counts())
    st.bar_chart(dist)

    #Keep only United Kingdom data
    Rtl_data = Rtl_data.query("Country=='United Kingdom'").reset_index(drop=True)

    #Check for missing values in the dataset
    Rtl_data.isnull().sum(axis=0)

    #Remove missing values from CustomerID column, can ignore missing values in description column
    Rtl_data = Rtl_data[pd.notnull(Rtl_data['CustomerID'])]

    #Validate if there are any negative values in Quantity column
    Rtl_data.Quantity.min()

    #Validate if there are any negative values in UnitPrice column
    Rtl_data.UnitPrice.min()

    #Filter out records with negative values
    Rtl_data = Rtl_data[(Rtl_data['Quantity']>0)]

    #Convert the string date field to datetime
    Rtl_data['InvoiceDate'] = pd.to_datetime(Rtl_data['InvoiceDate'])

    #Add new column depicting total amount
    Rtl_data['TotalAmount'] = Rtl_data['Quantity'] * Rtl_data['UnitPrice']
    st.text('After Cleaning the data and taking only UK data')

    st.write(Rtl_data.head())

    st.header('RFM Modelling')
    st.image('yo.png')

    #Recency = Latest Date - Last Inovice Data, Frequency = count of invoice no. of transaction(s), Monetary = Sum of Total 
    #Amount for each customer

    #Set Latest date 2011-12-10 as last invoice date was 2011-12-09. This is to calculate the number of days from recent purchase
    Latest_Date = dt.datetime(2011,12,10)

    #Create RFM Modelling scores for each customer
    RFMScores = Rtl_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})

    #Convert Invoice Date into type int
    RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)

    #Rename column names to Recency, Frequency and Monetary
    RFMScores.rename(columns={'InvoiceDate': 'Recency', 
                            'InvoiceNo': 'Frequency', 
                            'TotalAmount': 'Monetary'}, inplace=True)

    st.write(RFMScores.reset_index().head())
    
    #Recency distribution plot
    x = RFMScores['Recency']
    sns.distplot(x)
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    st.subheader('Recency distribution Plot')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.text('Frequency distribution plot, taking observations which have frequency less than 1000')
    x = RFMScores.query('Frequency < 1000')['Frequency']
    sns.distplot(x)
    st.pyplot()

    st.text('Monateray distribution plot, taking observations which have monetary value less than 10000')
    x = RFMScores.query('Monetary < 10000')['Monetary']
    sns.distplot(x)
    st.pyplot()

    #Split into four segments using quantiles
    quantiles = RFMScores.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()

    st.write(quantiles)

    #Functions to create R, F and M segments
    def RScoring(x,p,d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]: 
            return 3
        else:
            return 4
        
    def FnMScoring(x,p,d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]: 
            return 2
        else:
            return 1
    
    st.text('Calculate Add R, F and M segment value columns in the existing dataset to show R, F and M segment values')
    RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
    RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
    RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
    st.write(RFMScores.head())

    #Calculate and Add RFMGroup value column showing combined concatenated score of RFM
    RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)

    st.text('Calculate and Add RFMScore value column showing total sum of RFMGroup values')
    RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
    st.write(RFMScores.head())

    st.text('#Assign Loyalty Level to each customer')
    Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
    Score_cuts = pd.qcut(RFMScores.RFMScore, q = 4, labels = Loyalty_Level)
    RFMScores['RFM_Loyalty_Level'] = Score_cuts.values
    mod1 = RFMScores.head()
    # st.write(mod1)

    st.text('#Validate the data for RFMGroup = 111')
    mod2 = RFMScores[RFMScores['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(10)
    # st.write(mod2)


    st.text('#Recency Vs Frequency')
    graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

    plot_data = [
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
            mode='markers',
            name='Bronze',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
            mode='markers',
            name='Silver',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
            mode='markers',
            name='Gold',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
            mode='markers',
            name='Platinum',
            marker= dict(size= 13,
                line= dict(width=1),
                color= 'black',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = gobj.Layout(
            yaxis= {'title': "Frequency"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig = gobj.Figure(data=plot_data, layout=plot_layout)
    # po.iplot(fig)
    st.plotly_chart(fig)
    st.text('#Frequency Vs Monetary')
    graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

    plot_data = [
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
            y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
            mode='markers',
            name='Bronze',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
            y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
            mode='markers',
            name='Silver',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
            y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
            mode='markers',
            name='Gold',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
            y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
            mode='markers',
            name='Platinum',
            marker= dict(size= 13,
                line= dict(width=1),
                color= 'black',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = gobj.Layout(
            yaxis= {'title': "Monetary"},
            xaxis= {'title': "Frequency"},
            title='Segments'
        )
    fig = gobj.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.text('#Recency Vs Monetary')
    graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

    plot_data = [
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
            mode='markers',
            name='Bronze',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
            mode='markers',
            name='Silver',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
            mode='markers',
            name='Gold',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
        gobj.Scatter(
            x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
            y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
            mode='markers',
            name='Platinum',
            marker= dict(size= 13,
                line= dict(width=1),
                color= 'black',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = gobj.Layout(
            yaxis= {'title': "Monetary"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    fig = gobj.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.header('K-Means Clustering')
    #Handle negative and zero values so as to handle infinite numbers during log transformation
    def handle_neg_n_zero(num):
        if num <= 0:
            return 1
        else:
            return num
    #Apply handle_neg_n_zero function to Recency and Monetary columns 
    RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
    RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]

    st.text('#Perform Log transformation to bring data into normal or near normal distribution')
    Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)

    Recency_Plot = Log_Tfd_Data['Recency']
    ax = sns.distplot(Recency_Plot)
    st.pyplot()

    st.text('#Data distribution after data normalization for Frequency')
    Frequency_Plot = Log_Tfd_Data.query('Frequency < 1000')['Frequency']
    ax = sns.distplot(Frequency_Plot)
    st.pyplot()

    st.text('#Data distribution after data normalization for Monetary')
    Monetary_Plot = Log_Tfd_Data.query('Monetary < 10000')['Monetary']
    ax = sns.distplot(Monetary_Plot)
    st.pyplot()




    #Bring the data on same scale
    scaleobj = StandardScaler()
    Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

    #Transform it back to dataframe
    Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScores.index, columns = Log_Tfd_Data.columns)

    sum_of_sq_dist = {}
    for k in range(1,15):
        km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
        km = km.fit(Scaled_Data)
        sum_of_sq_dist[k] = km.inertia_
        
    st.text('#Plot the graph for the sum of square distance values and Number of Clusters')
    sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
    plt.xlabel('Number of Clusters(k)')
    plt.ylabel('Sum of Square Distances')
    plt.title('Elbow Method For Optimal k')
    st.pyplot()

    KMean_clust = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 1000)
    KMean_clust.fit(Scaled_Data)

    st.text('#Find the clusters for the observation given in the dataset')
    RFMScores['Cluster'] = KMean_clust.labels_
    # st.write(RFMScores.head())

    plt.figure(figsize=(7,7))

    st.text('##Scatter Plot Frequency Vs Recency')
    Colors = ["red", "green", "blue"]
    RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
    ax = RFMScores.plot(    
        kind="scatter", 
        x="Recency", y="Frequency",
        figsize=(10,8),
        c = RFMScores['Color']
    )
    st.pyplot()


with header1:
	st.title("Customer Churn Prediction")
	st.image('download.png')



with data1:
	df = pd.read_csv("data/churn.csv", index_col=0)
	df.columns = map(str.lower, df.columns)
	st.text('The first 5 observation units of the data set were accessed.')
	st.write(df.head())
	st.text('The size of the data set was examined. It consists of 10000 observation units and 13 variables.')
	st.write(df.shape)
	st.text('Feature information')
	st.write(df.info())
	st.text('Descriptive statistics of the data set accessed.')
	st.write(df.describe().T)
	st.text('The average of the age variable was taken according to the dependent variable.')
	st.write(df.groupby("exited").agg("mean"))
	st.text('The average of the age variable according to the gender variable was examined.')
	st.write(df.groupby("gender").agg({"age": "mean"}))
	st.text('The average of the dependent variable according to the gender variable was examined.')
	st.write(df.groupby("gender").agg({"exited": "mean"}))
	st.text('The average of the dependent variable according to the geography variable was examined.')
	st.write(df.groupby("geography").agg({"exited": "mean"}))
	st.text('The frequency of the dependent variable has been reached')
	st.write(df["exited"].value_counts())
	churn = df[df["exited"] == 1]
	st.text('The first 5 observation units were reached.')
	st.write(churn.head())
	st.text('The most commonly used surname was examined and observed to be non-multiplexing.')
	st.write(df[df["surname"] == "Smith"])
	st.text('The distribution of the dependent variable in the dataset is plotted as pie and columns graphs')
	f,ax=plt.subplots(1,2,figsize=(18,8))
	df['exited'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
	ax[0].set_title('dağılım')
	ax[0].set_ylabel('')
	sns.countplot('exited',data=df,ax=ax[1])
	ax[1].set_title('exited')
	# plt.show()
	st.pyplot()

	st.text('Plotted the categorical variables on the basis of the graph of the column according to the dependent variable.')


	fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
	sns.countplot(x='geography', hue = 'exited',data = df, ax=axarr[0][0])
	sns.countplot(x='gender', hue = 'exited',data = df, ax=axarr[0][1])
	sns.countplot(x='hascrcard', hue = 'exited',data = df, ax=axarr[1][0])
	sns.countplot(x='isactivemember', hue = 'exited',data = df, ax=axarr[1][1])
	st.pyplot()

	st.text('Correlation Matrix')
	f, ax = plt.subplots(figsize= [20,15])
	sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
	ax.set_title("Correlation Matrix", fontsize=20)
	# 
	st.pyplot()


	st.text('Boxplot graph for outlier observation analysis')
	fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
	sns.boxplot(y='creditscore',x = 'exited', hue = 'exited',data = df, ax=axarr[0][0])
	sns.boxplot(y='age',x = 'exited', hue = 'exited',data = df , ax=axarr[0][1])
	sns.boxplot(y='tenure',x = 'exited', hue = 'exited',data = df, ax=axarr[1][0])
	sns.boxplot(y='balance',x = 'exited', hue = 'exited',data = df, ax=axarr[1][1])
	sns.boxplot(y='numofproducts',x = 'exited', hue = 'exited',data = df, ax=axarr[2][0])
	sns.boxplot(y='estimatedsalary',x = 'exited', hue = 'exited',data = df, ax=axarr[2][1])
	st.pyplot()


	st.subheader('''Report
		''')
	st.text('''The aim of this study was to create classification models for the churn dataset and to predict whether a person abandons us by creating models and to obtain maximum accuracy score in the established models. The work done is as follows:

1) Churn Data Set read.

2) With Exploratory Data Analysis; The data set's structural data were checked. The types of variables in the dataset were examined. Size information of the dataset was accessed. Descriptive statistics of the data set were examined. It was concluded that there were no missing observations and outliers in the data set.

4) During Model Building; Logistic Regression, KNN, SVM, CART, Random Forests, XGBoost, LightGBM, CatBoost like using machine learning models Accuracy Score were calculated. Later XGBoost, LightGBM, CatBoost hyperparameter optimizations optimized to increase Accuracy score.

5) Result; The model created as a result of LightGBM hyperparameter optimization became the model with the maxium Accuracy Score. (0.9116)     ''')
	

	
	
	
	

