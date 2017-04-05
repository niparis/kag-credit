1. Tell us how you validate your model, which, and why did you chose such evaluation technique(s).

I use cross-validation. In that specific case, i've used Shuffle Cross validation with 5 splits & 20% test size, meaning that we train our model 5 times using 80% of the data, and test it against the remaining 20%. We use AUC for scoring.

Shuffle Cross validation seems more appropriate than KFold, as we have a rare event to predict, and we don't know if those events (default) are evenly distributed in our dataset. So by using random shuffling, we avoid the risk of having almost all defaults or none in the training set. That's for scikit learn models. XGBoost uses its own CV.

The purpose of the cross validation is to avoid overfitting on our training data set & of the model's hyperparameters, so we're looking for a low standard deviation on our choosen evaluation measure. Actual measure used are the average `test-auc-std` and `test-auc-mean` shown when executing the train command


2. What is AUC? Why do you think AUC was used as the evaluation metric for such a problem? What are other metrics that you think would also be suitable for this competition?

- AUC - Area Under the precision-recall Curve. It's a plot of Precision (y-axis) & recall (x-axis). High Precision show a low false positive rate, while a high recall show a low false negative rate.
- A high AUC means we have both high recall & high precision. As a result, having a high AUC means having a low false negative rate - that's the ultimate objective here.
Besides, accuracy is quite useless for this problem. We are trying to predict a are event (default). Less than 7% of the sample size has defaulted. So with a stupid model (nobody defaults), we'd get a seemingly excellent accuracy (93%+), while the model would be useless (fails to predict any default)
- False Negative rate is the most important metric from the bank's perspective, as they'd want to avoid extending credit to a future defaultee. So we could optimize based on recall.

3. What insight(s) do you have from your model? What is your preliminarily analysis of the given dataset?

The following features are strong predictors of default:
- Age. Lower age is more likely to default, and people above retirement age default less. The effect is linear until retirement age.
- DebtRatio: Having a debt ratio around 1 (between 0.8 and 2). Surprisingly little different within that range.
- Past Due: Getting past due is highly predictive of default - event just once ! Some special codes (96 & 98) seem to be identifying some especially risky individuals
- Using

Another important point is that no single feature is an extremely strong predictor of default.

4. Can you get into top 100 of the private leaderboard, or even higher?

At the time of writing, my private AUC is 0.864174 which is ranking 412 (checking on my own, dont' seem to be able to actually appear in the LB) - using XGBClassifier.

Elements I'm looking at for improvement:
- I've started tuning hyper-params, but i think there's still some work
- I suspect having a model to predict Income rather than just using the median would help ()
- Haven't had time to really look at feature importance (output from model) & feature correlation (doesnt matter for XGBoost, but would be interesting to look at)
- XGBClassifier is a very powerful classifier (first time using it), but it's harder to optimize by feature engineering. The features I've created increase the perf (ROC AUC) of a log reg from 67.85% to 84.81% (pretty neat increase, still losing to XGBoost), while XGBoost isnt really helped: from 85.91% to 85.84% (auc, average on test set - private score in Kaggle is helped by feature engineering).  Nevertheless, there are probably some features with negative impact, so identifying them will help.

