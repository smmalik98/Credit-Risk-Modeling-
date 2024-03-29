# Credit-Risk-Modeling
Data Cleaning &amp; Processing of a dataset to understand the likelihood and the financial impact of defaulting lender.

## Part 1: Data Cleaning And Processing

### What is Credit Risk Analysis
<img width="250" alt="image" src="https://user-images.githubusercontent.com/97324716/205785618-764032c0-ad98-4793-acee-6ddf32cb12f7.png">
Credit risk analysis is a method of determining the likelihood that a customer will fail to make a payment before any credit is extended to them. To assess a customer's creditworthiness, businesses establish their history of timely payment and their ability to continue doing so. High risk customers are inversely proportional to cashflow since even if a small proportion of these customers default/lag their payment, it will directly hit the cashflow of the firm. Credit risk analysis is not a new concept but it's amalgamation with Machine Learning/Statitics has been upcoming and valued after a few successful deployment of these models. But these model require more things than just the payment history, they require almost all details linked to the customer.

### Defining Clean Data
<img width="250" alt="image" src="https://user-images.githubusercontent.com/97324716/205785778-bd781b83-913d-4ac7-9fa4-e0a92c37172d.png">
Here I am trying to clean the data which is required for credit risk analysis. The definition of clean data is relative and varies depending on it's usage. For me, I am defining a clean data as a data which can be used for predicting the probability of the customers and a data which is machine readable. But I will also be providing with a clean data which was defined according to a paper written by Wickham (which is human readable data)

### Cleaning Life-Cycle
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97324716/205786332-c74601e1-c4ae-4b24-8f17-d3ee22e0b508.png">
Every person working with data craves for an ideal data that fulfills each and every target (Model Development, Visualization etc). To achieve their goal these people spend eternity to reach that goal. This meme clearly expresses what I am trying to tell and what I felt while cleaning this data. This is a huge dataset which has got 75 rows and ~900k rows. Simply putting this data into a model won't make sense since this data is currently not even in its human-readable. Our first priority here is to convert it into a human-readable format and then transforming it into a machine readable format. There were many columns that had to be converted into a digestible formats like converting dates from 'obj' to 'datetime'.
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97324716/205788860-ca3e19ca-0c3b-4bae-880b-c11ffb800cef.png">
But everything always pointed towards one thing i.e., "DROPPING COLUMNS". My first step towards it constitutes of getting an overview of the data and removing columns that cannot be used (like titles, url etc). Moving forward I look at the missing values in the column, if there are rows with more than 30% of missing values, drop them or try to find patterns. Replacing large amounts of missing cells makes the data bias which is why I try to avoid replacing values.

### Feature Selection and Engineering
There are different methods when it comes to feature selection like 'SelectKBest', 'ChiSquare Method' etc but every method has its own purpose. This is where we introduce the concepts of Weight of Evidence and Information Value.

#### Weight of Evidence: 
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97324716/205807842-51d848ce-6013-45a8-83cc-5db2ebd42ee8.png">
This meme best explains the concept of WoE. It can be seen that each witness is describing the appearance of the perpretor. Since the appearance which is being described matches the appearance of Frank, his lawayer is therefore informing him that the weight of evidence is against him.

#### Information Value
The meme indirectly is also talking about information value. The information provided by each witness has some value and this value adds up to form the information value. 

<img width="500" alt="Screen Shot 2023-01-17 at 9 30 25 AM" src="https://user-images.githubusercontent.com/97324716/212807506-0ce16b00-dd5a-4432-83a9-0265cd0c9f5c.png">
The information can be weak and incapable of targetting the desired information or the information can be too goo to be true. The desired value of range can be found in the upper table.

Moving forward we will be looking at the information value, the weight of evidence and the number of observations, but this will be limited to only categorical variable. 
![image](https://user-images.githubusercontent.com/97324716/205811639-5c6b36be-5ff2-460d-9a82-1faf25e19dfd.png)
When we plot the weight of evidences of each element of a column, we get something like. But what should we be looking at? 
All grades have a linear relationship with weight of average. With 'A' having a very high WOE it raises suspiscion but if we look at it's 'n_obs', it seems to have a high value which removes our suspiscion. 'E' and 'D' have similar WOE but when we look at their 'n_obs', there is a huge difference, hence they cannot be merged. We are going to keep each of them in their original form, there is no need to merge them.

But there are cases where it's all same like this, what strategy should we follow?
![image](https://user-images.githubusercontent.com/97324716/205811960-2cec927e-9eef-44a6-bccc-a83ee4a06fdd.png)

Here it is very difficult to find out which attributes can be grouped or merged. The strategy for grouping these elements will include looking at the n_obs and WOE like before with a twist. Since the WOE of elements are very similar we will keep grouping elements looking at their WOE first. When we see that the n_obs of the succeding element decreases, the group ends there and a new group starts. There are cases where there is only a single element in a group. The n_obs difference and the WOE difference must have been too large or the previous strategy might be isolating some elements. Some elements have their n_obs very low i.e. >2000, they have been grouped together to have a comparable n_obs with rest of the groups.

On the basis of the strategy described, here are the final groups:

IA, HI, ID, ME, WY, DC, NE, ND, ME

NV, AL

NY

VA, OK, LA, FL

UT, SD, NM, NC

CA

MD, NJ, AZ, TN, IN, PA, MI, MO, MN

AR, OH

DE, KY, MA

RI

WA, OR

TX, WI

AK, GA

KS, MT, IL

When it comes to numerical variables, we are simply using correlation plot to eliminate some highly correlated variables.

## Part 2: Model Development
### Hyper-parameter Tuning using Bayesian Hyperopt
Bayesian hyperopt is a way to find the best set of parameters for a model. It works by testing different combinations of parameters and using the results to predict which set of parameters will work best. It uses a method called Bayesian optimization, which makes educated guesses about which parameter combinations to try next based on the previous results. This process continues until the algorithm finds the best set of parameters for the model. Essentially, it's like trying different recipes and using the feedback to refine the ingredients and cooking process until you create the perfect dish.

### Model Results 
![image](https://user-images.githubusercontent.com/97324716/235721099-c75b697b-4a57-4157-af6d-1099f1cb576c.png)

It is important to understand the business context in which the model will be used. For instance, what is the impact of false positives and false negatives? If a false positive results in a good customer being denied credit, it can lead to loss of business and customer dissatisfaction. Similarly, if a false negative results in a bad loan being approved, it can lead to default and financial losses for the lender. Thus, the business objectives and acceptable levels of risk need to be considered.

In summary, while an F1 score of 0.97 is impressive, it is important to consider the business context, data quality, interpretability, and performance in production before deciding whether the model is ready for deployment.


# References:
1. SinHacker. (2020, March 9). Model? or do you mean weight of evidence (WOE) and information value (iv)? Medium. Retrieved November 30, 2022, from https://towardsdatascience.com/model-or-do-you-mean-weight-of-evidence-woe-and-information-value-iv-331499f6fc2
2. Weight of evidence - definition and meaning. Market Business News. (2018, August 17). Retrieved November 30, 2022, from https://marketbusinessnews.com/financial-glossary/weight-evidence-definition-meaning/
3. Credit risk assessment: Allianz Trade in USA. Corporate. (n.d.). Retrieved November 30, 2022, from https://www.allianz-trade.com/en_US/insights/how-to-improve-credit-risk-analysis.html#:~:text=What%20is%20Credit%20Risk%20Analysis,to%20continue%20to%20do%20so.
4. Krishnan, S. (2019, December 20). Weight of evidence and information value using python. Medium. Retrieved November 30, 2022, from https://sundarstyles89.medium.com/weight-of-evidence-and-information-value-using-python-6f05072e83eb
5. Team, T. I. (2022, September 20). Credit risk: Definition, role of ratings, and examples. Investopedia. Retrieved November 30, 2022, from https://www.investopedia.com/terms/c/creditrisk.asp 
6. Numpy.dtype.kind¶. numpy.dtype.kind - NumPy v1.9 Manual. (n.d.). Retrieved December 5, 2022, from https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.dtype.kind.html 
7. Weight of evidence - definition and meaning. Market Business News. (2018, August 17). Retrieved December 5, 2022, from https://marketbusinessnews.com/financial-glossary/weight-evidence-definition-meaning/ 
