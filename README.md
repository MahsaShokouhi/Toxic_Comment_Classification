# Toxic Comment Classification¶

[Link_to_notebook](https://github.com/MahsaShokouhi/Toxic_Comment_Classification/blob/master/Toxic_Comment_Classification.ipynb)

<br>

## Project Description

It is challenging for platforms to effectively identify toxic conversations due to the high volume of conversations. Automatic detection of toxic conversations can therefore help platforms to identify these types of conversations efficiently. The aim of this project is to develop a NLP model to label a comment as toxic or non-toxic. 

<br>

## Dataset

The dataset includes Wikipedia comments (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) labeled using the following 6 toxicity types:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

Note that a comment can belong to multiple toxic groups.

The train and tests set include 159571 and 153164 samples, respectively. Among the test samples only 63978 were labelled.

<br>

## Exploratory Data Analysis

Among all 159571 comments in the train set, 143346 are non-toxic and 16225 belong to at least one of the toxic groups, hence a very imbalanced data. 

Here's the breakdown of all 16225 toxic comments:


![Table 1](/figures/tab1.png)


As can be seen, 94% of toxic comments at least belong to the general 'toxic' subgroup. The other major subgroups are 'obscene' and 'insult' types, representing 52% and 49% of all toxic comments. 'threat' subgroup represents 3% of toxic comments. 

There's a considerable overlap between some of these toxic subgroups, as shown by the following correlation plot, indicating strong correlation between 'obscene' and 'insult' subgroups, for example.

![Figure 1](/figures/fig1.png)

The average length of comments in each subgroup also varies, as can be seen from the following figure. Note that groups represent different combinations of the toxic labels (the one-hot encoded representation follows the label orders shown at the bottom of the figure). For example, the first bar corresponds to non-toxic comments. The 5th bar, with a hight average comment length, represents the comments that are labeled as both 'insult' and 'threat'.

![Figure 2](/figures/fig2.png)


It should be noted however, that the average comment length could be misleading due to a highly imbalanced dataset. 


The following figure shows the results of comments sentiment analysis within each subgroup. As expected, the average sentiment score is only positive for non-toxic comments. Also, the subgroups in the right side of the plot, which represent most types of toxic comment, have the lowest sentiment scores.

![Figure 3](/figures/fig3.png)

<br>

## Data Preprocessing

Comment text was vectorized using TF-IDF vectorizer. The number of features for TF-IDF vectorizer was selected using grid-search validation. The number of features was set to 10,000 which achieved the lowest hamming loss and highest accuracy score.

The sentiment score was also added as an additional feature, resulting in a total of 10001 features.

<br>

## Baseline Model

For comparison, a dummy classifier with stratified strategy (to account for the data imbalance) was trained on the train set, which achieved the average score of 0.04 for precision, recall, and f1-score on the test-set.

<br>

## Model Selection

Classifier chains algorithm was used for multi-label classification. As the name suggests, this method involves a series of classifiers (base model) where each classifier predicts one label at a time (the total number of classifiers is therefore the same as the number of labels, i.e. 6 in this case.). The first classifier is trained on the input features to predict the 1st label. Each successive classifier then takes the features and all the label predictions of the previous classifiers as the input to predict one of the remaining labels. This way, the correlation between labels is also accounted for.

Classifier chains algorithm needs a base classification model to perform the classification. For this project a range of models, including logistic regression and ensemble tree-based model were evaluated and compared, before selecting the best model. The evaluation was performed using 3-fold iterative, stratified cross-validation, with accuracy, hamming loss, and f1-scores as metrics. The following table compares the performances of different models on train and validation sets, estimated using cross-validation.

![Table 2](/figures/tab2.png)

Compared to logistic regression and gradient boosting, random forest takes much longer and is overfitting as well. Logistic regression and gradient boosting show similar results and were chosen for evaluation using the test set.

<br>

## Prediction on Test Set

To account for data imbalance, a cost-sensitive logistic regression model (which adjusts the class weights so that minority classes get higher weights for loss calculation) was also examined. Here's the evaluation metrics on test set:

![Table 3](/figures/tab3.png)


The balanced (weighted) logistic regression model achieves higher recall but lower precision and f1-score.

The table below shows the percentages of true negative (TN), false positive (FP), false negative (FN), and true positive (TP) within each class with logistic regression, suggesting that among all labels, 'toxic' label has the highest misclassification rates, followed by the 'insult' label.

![Table 4](/figures/tab4.png)
