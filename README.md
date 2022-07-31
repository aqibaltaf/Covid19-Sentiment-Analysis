# COVID’19 Tweets - Text Sentiment Analysis

## INTRODUCTION

Since last year the world has changed drastically due to the coronavirus outbreak which slowed down the fast-paced world. Many countries had to declare complete lockdown in order to contain it. During this period, people used social media to spread information about the pandemic and through their messages.
In this project we are going to use diﬀerent techniques to diﬀerentiate whether the information that was conveyed and the tone of that information is positive, negative or neutral. The data that we are going to use are tweets mentioning corona virus extracted from twitter. We have used Naïve Bayes and boosting to predict whether the tweets are positive or negative.


## DATA UNDERSTANDING

The data has been taken from Kaggle Competition.

https://www.kaggle.com/datatattle/covid-19-nlp-text-classiﬁcation/tasks?taskId=2011

The dataset provided consists of 44,955 tweets and has been divided into testing and training data where each sample in the train and test set have the following information:

•	The original text of the tweet

•	The location of the tweet

•	The date of the tweet
 
<strong>Columns</strong>

•	UserName – a code has been given to uniquely identify the names and also for privacy concerns.
•	ScreenName – the code of the PC/phone used to post the tweet.

•	Location – where the tweet was tweeted.

•	TweetAt – the date on which the tweet was posted.

•	OriginalTweet – the whole tweet that the user posted.

•	Sentiment – shows if the tweet was positive, neutral or negative.

Our goal is to predict whether people have negative or positive sentiments during the pandemic from tweets.
 
## DATA PREPARATION

In the data preparation, we clean the data for training it with diﬀerent models. We will train the models using training data which has 41157 tweets and test it by using testing data which has 3798 tweets.
Because tweets are textual data and text data is unstructured and full of unwanted patterns, it is extremely diﬃcult to work with, especially when it comes to machine learning. For optimizing the machine learning model, we handled null values, stop words and the sentences which are full of special characters such as, @ and # in the tweets in the following ways:
<h4>Handling the null values</h4>

Upon testing the data without removing null values, we concurred that it was aﬀecting the eﬀectiveness of the model therefore, we handled that by ﬁnding the columns with most null values and removing them. The column that we removed was the location. Since we are doing sentiment analysis, location is not an important factor in deﬁning the sentiments of people and had 9424 values in total, removing it was the best way of handling the null values.

![image](https://user-images.githubusercontent.com/61707240/182018549-f0930915-1c34-4b56-949d-76c3b27981a1.png)
 
<h4>Removing the symbols and short words </h4>

Data with symbols and short words make the classiﬁcation process diﬃcult. The symbols which are common and redundant such as hashtags and asterisks as well as short words do not hold any value for ﬁnding the sentiment and for the essence of the sentence, thus, removing it helps optimize the model. We have removed such symbols and words whose length is less than 2.

![image](https://user-images.githubusercontent.com/61707240/182018589-a51fd93a-f0bb-4d1f-acd8-5aabb66f3c29.png)

![image](https://user-images.githubusercontent.com/61707240/182018600-ccae884d-dcb8-4080-a2cf-4607ff105e6f.png)

<h4>Removing the stop words</h4>

Stop words only help in connecting the sentence and like symbols do not hold any importance in ﬁnding out the essence of the sentence and thus, we have removed it.

![image](https://user-images.githubusercontent.com/61707240/182018609-76eb024f-2025-4bc8-b51a-16804a968e65.png)

<h4>Tokenization and Normalization</h4>

After removing unwanted patterns, words and symbols, we tokenize and normalize the data. We tokenized the data to break text into words to help the model in understanding the context by analyzing the sequence of the words and normalized it to maintain the general distribution of the data.

![image](https://user-images.githubusercontent.com/61707240/182018625-c74e69aa-24e5-4e39-bae7-258213c6777c.png)

<h4>Stemming</h4>

For cleaning the data further, we converted words into their base forms. For this we tried both stemming and lemmatization but got a better accuracy from stemming and hence used that before vectorization.

![image](https://user-images.githubusercontent.com/61707240/182018640-526d84f1-09b3-4a45-8e16-c53656d90451.png)

![image](https://user-images.githubusercontent.com/61707240/182018656-d2ee0501-d5d2-443b-81d6-8be108adfe9f.png)


These steps made the data much cleaner to work with.

<h4>Vectorising</h4>

Machine learning models can only take in integer values and that is why we used the vectorizer. A document term matrix was created and it indicated where the words were used in a sentence (document). We have used this method to convert words into integers through which we have then trained and tested the model with.
When vectorising, following were the parameters:

●	Max DF = 0.90

●	Min DF = 3

●	Max features= 3000

With this, we will have 3000 columns and for a word to become a feature, it will have to appear at least in 3 tweets.

![image](https://user-images.githubusercontent.com/61707240/182018481-cdd0c58d-8b86-49da-bc6e-30c49a9b4c0d.png)

Following is the vectorised matrix of words

![image](https://user-images.githubusercontent.com/61707240/182018496-a20b3095-7769-4167-b617-b23b47e0391b.png)


<h4>Labelling</h4>

We did mapping to give sentiments a number for optimizing the model. Here we have grouped extremely negative and negative together and have labeled them as zero, neutral is one and positive and extremely positive are labeled two. By grouping similar sentiments we were able to achieve a higher accuracy than giving each one a separate number.

![image](https://user-images.githubusercontent.com/61707240/182018508-3fe70104-3319-44b3-9319-6419959cc7d7.png)

Following is the output of all the pre-processing steps we performed above:

![image](https://user-images.githubusercontent.com/61707240/182018516-2aa24e5a-8a8d-4d33-9aa5-ecad23cacd5b.png)
 
## MODEL BUILDING AND EVALUATION

<h4>Libraries used </h4>

![image](https://user-images.githubusercontent.com/61707240/182018324-2f786360-e5d7-4bce-9b03-4d0993769b47.png)


<h4>Models used:</h4>

●	Naive Bayes

●	Extreme Gradient Boosting Tree

<h3>Naive Bayes</h3>

We extracted the data from DTM after vectorising it. Then we split the data into training set which was 66% and testing set which was 33%, the model that we used ﬁrst for the sentiment analysis was Naive Bayes. We were able to get 69% accuracy with this model after preparing the data.

![image](https://user-images.githubusercontent.com/61707240/182018349-1ecee82f-00da-4954-a9a2-1ea298491b49.png)

![image](https://user-images.githubusercontent.com/61707240/182018380-8a27d3c5-a780-47a6-8727-5287157f6439.png)

<h3>Extreme Gradient Boosted Tree (XGBT)</h3>

We extracted the data from DTM after vectorising it. Then we split the data into training set which was 75% and testing set which was 25%, the model that we used next for the sentiment analysis was Extreme Gradient Boosted Tree. At ﬁrst, the accuracy of the model was 63% which we then improved by optimizing the algorithm. For that, we changed the parameters several times and tuned it for the best result. We were able to get 83% accuracy with this model after preparing the data.

![image](https://user-images.githubusercontent.com/61707240/182018399-b8f7a617-474c-4623-a5d7-d8b19c32b61a.png)
 

The accuracy can be increased if we increase the number of models but there is no drastic change between the accuracies. Also after increasing the models, there eventually comes a point that the accuracy starts to decrease.
This model has a higher accuracy rate than the naive bayes one which means that it works better.

![image](https://user-images.githubusercontent.com/61707240/182018419-60c1b23f-05a4-43b2-afe4-90e1bf382a45.png)

For both models it is important to take into consideration that modelling was easy and accuracy was actually improved by the cleaning process and cleaning process was more tedious and time taking which is common for problems which have unstructured data.
 

## FINDINGS

The dataset helped us to predict whether a tweet posted during the time of Covid-19 was a positive tweet or a negative tweet. It was a way through which we could analyse the sentiments of the people during the pandemic. Although the ratio of positive tweets was higher than that of the negative tweets, there is still an ongoing negative array which is quite strong and given the conditions it is natural.

Sentiment analysis of tweets related to Covid-19 can also help governments and administrative units with the distribution of covid vaccine at scale and with tackling the needs and problems people might be facing due to the pandemic.

Moreover, these tweets can also show the businesses the demand for various things and if those products, their availability and placement currently is inculcating positive or negative sentiments among people and how to improve this situation.

![image](https://user-images.githubusercontent.com/61707240/182018199-a65de3d5-6e53-4956-90a3-c188d5ef7a28.png)

The words used to write these tweets were used to predict whether a given tweet falls in the positive class or the negative.
However, there is a limitation which is that there exists a high similarity in the words of both negative and positive tweets as can be seen by the word cloud and could be a reason why the models did not perform as well.
 

<h3>Negative Word Cloud</h3>

![image](https://user-images.githubusercontent.com/61707240/182018062-4ee9bac2-1615-409c-b821-2a8de7812f16.png)


<h3>Positive Word Cloud</h3>

![image](https://user-images.githubusercontent.com/61707240/182018057-4d8a990b-f355-4152-aff1-c873d65a8294.png)


This problem could be solved by oversampling the data of the class which is in minority and undersampling the data of the majority class.

One more limitation was that there existed a high diversity of words which was affecting the predictions of the models. This problem was solved by us by increasing the minimum document frequency which ended up dropping data which could have been meaningful.
 
Furthermore, machine learning models not being able to differentiate between sentiments such as extremely positive and positive is also a limitation. This problem can be solved by labeling them differently or amalgamating similar ones together. For example: right and more right will be labelled right.

Our suggestion for organizations are that:
Data collection should be collected by organizations in a way which makes sure that all variables that will affect the outcome of the predictions are included and the irrelevant data should not be collected such as the location column in our data..

Processes should be performed in a way which minimizes the noise in data, prepares data, excludes columns which do not make any difference on the outcomes.

Using these variables and processes can further improve and optimize the analysis.

Moreover, the model would expire when Covid-19 ends as it is helping us predict the sentiments of the people through the tweets posted during pandemic and would be of little use after the pandemic is over.
