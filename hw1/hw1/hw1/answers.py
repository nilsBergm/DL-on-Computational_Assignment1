r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:
 First Question: False. The test set (unseen data) allows us to estimate our out-of-sample error, or generalization error, which is the error rate on unseen data. 
 In contrast, the in-sample error (or training error) is the error rate on the data that was used to train the model.
 
 second Question: False. Not all train-test splits are equally useful. The performance of a machine learning model depends on the quality and representativeness of the data it is trained on. 
 Therefore, the choice of train-test split can have a significant impact on the performance of the model.
 A good train-test split should ensure that the distribution of the data in the training set is representative of the distribution of the data in the test set, and ideally also representative of the distribution of the data in the real world where the model will be used. 
 If the training set and test set are not representative of each other or of the real world, the model may overfit to the training data and perform poorly on the test data and/or real-world data.
 Therefore, it is important to carefully consider the choice of train-test split, and to use techniques such as stratification, cross-validation, and random shuffling to ensure that the data is split in a representative and unbiased manner.
 
 Third Question:True. The test set is meant to provide an unbiased evaluation of the final model after all hyperparameter tuning and model selection has been completed. 
 Using the test set during cross-validation would break this separation between training and evaluation, and would result in an overoptimistic estimate of the model's performance.
 The test set should be kept separate from the training and validation sets used during cross-validation, and should only be used once, at the end, to obtain an unbiased estimate of the final model's performance.
 
 Forth Question: True. In cross-validation, the data is split into multiple folds, and each fold is used as a validation set while the remaining data is used for training. 
 This process is repeated for each fold, with each fold serving as the validation set exactly once. The performance of the model is then averaged across all folds to obtain an estimate of its performance.
 The validation-set performance of each fold can be used as a proxy for the model's generalization error, as it provides an estimate of how well the model will perform on new and unseen data. 
 By averaging the validation-set performance across all folds, we obtain a more reliable estimate of the model's generalization performance than if we were to use a single train-test split.
 It's worth noting, however, that the performance of the model on the validation set is still only an estimate of its performance on new and unseen data. 
 Therefore, it's important to keep the test set separate and only use it once, at the end, to obtain an unbiased estimate of the model's generalization error.
 **
"""

part1_q2 ="""
**Your answer:
No, my friend's approach is not justified.
By evaluating the model on a disjoint test set and concluding that it has overfit the training set, my friend has essentially used the test set as a validation set. 
This can lead to optimistic estimates of the model's performance, as the test set has been used to guide the model selection and hyperparameter tuning process.
In addition, by selecting the value of the regularization parameter based on the performance on the test set, my friend has introduced a data leakage problem. 
The regularization parameter should be selected based on the performance on a validation set that is separate from both the training and test sets. 
By using the test set to select the regularization parameter, my friend has incorporated information from the test set into the training process, potentially leading to overly optimistic estimates of the model's performance on new and unseen data.
Instead, my friend should have used a validation set to select the value of the regularization parameter. 
This can be done by randomly splitting the training set into a training and validation set, and then selecting the value of the regularization parameter that produces the best performance on the validation set. 
Once the regularization parameter has been selected, the model should be trained on the entire training set (without using the validation set) and evaluated on the test set to obtain an unbiased estimate of its performance on new and unseen data.
**

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:
In general, increasing the value of k tends to improve the generalization of the kNN classifier for unseen data up to a certain point, beyond which the performance may start to degrade. This is because a larger value of k considers more neighbors, resulting in a smoother decision boundary and less sensitivity to noise and outliers.
On the other hand, a smaller value of k may overfit to the training set and have higher variance.
However, the optimal value of k depends on the specific problem and the characteristics of the data set. Increasing k too much can lead to underfitting and a loss of discrimination power, as the decision boundary may become too simple and fail to capture the underlying patterns in the data. 
Additionally, increasing k also leads to an increase in computational complexity and memory usage, which can become a bottleneck for large data sets.
For extreme values of k, such as k=1 or k=N (the total number of training examples), the performance of the kNN classifier can be problematic. 
When k=1, the decision boundary can become too complex and sensitive to noise, resulting in overfitting to the training data. When k=N, every test point will be classified with the most frequent class in the training data, resulting in a high bias and low variance but potentially poor performance on unseen data.
Therefore, it is important to choose an appropriate value of k based on the trade-off between bias and variance, as well as the computational constraints of the problem.**

"""

part2_q2 = r"""
**Your answer:
Using k-fold CV is better than the two methods mentioned because it provides a more reliable estimate of the model's performance on unseen data.
When selecting the best model based on train-set accuracy, there is a risk of overfitting, where the model may fit the noise in the data and perform poorly on unseen data. 
This is because the model is optimized based on the same data it was trained on.
Similarly, selecting the best model based on test-set accuracy may also lead to overfitting, especially when the test set is small. 
It is possible to try out different models on the test set until one gets the best performance, but this can lead to a biased estimate of the model's performance on unseen data.
In contrast, k-fold CV ensures that the model is evaluated on all parts of the dataset, and the average performance across all folds is a more reliable estimate of the model's generalization performance. 
It also reduces the risk of overfitting by training the model on a subset of the data and evaluating it on another subset.
Therefore, k-fold CV is a preferred method for model selection and hyperparameter tuning as it provides a more reliable estimate of the model's performance on unseen data.**
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:
The selection of Δ>0 is arbitrary for the SVM loss 𝐿(𝑾) as it is defined above because it only affects the specific numerical value of the loss, and not the overall optimization problem.
This is because the regularization term, which is dependent on the model parameters, dominates the loss function as the value of Δ becomes very large. 
As a result, the model parameters will be determined primarily by the regularization term and not by the hinge loss term, which means that the value of Δ does not have a significant impact on the final model.
In other words, the choice of Δ>0 is a hyperparameter that can be tuned to balance the importance of correctly classifying the training data with the desire to maintain a large margin between classes.
However, as long as Δ is set to a positive value, the optimization problem will have the same solution, which is to minimize the hinge loss while also minimizing the magnitude of the model parameters.**

"""

part3_q2 = r"""
1.Using the visualization, one can see that each row (class) of weights learns a representation of the shape of the corresponding label.
 This representation is learned from the training set and if a number in the test set looks very different from the number in the training set, it can lead to a misclassification.
In addition, the visualization shows very well that the numbers 0-3 are represented very well by the weights, whereas 6 and 9 in particular are difficult to recognize.
 This is also reflected in the test results. If a test sample represents a very inaccurate number, it is often wrongly classified as a 6.
 
 2:The big difference is that in KNN there is no real learning. 
 There, classification is simply based on the neighbors. 
 In the linear classifier, the neighboring samples play no role in the classification. There the shapes of the labels are really learned. 
 Based on this, the classification is done by similarity between the learned weights and the test samples.




"""

part3_q3 = r"""
**Your answer:**
**Learning rate: I would say it is good, because it is an efficient learning, so the loss reduction is fast.

When the learning rate is too high, the training process is unstable, and the loss may oscillate or diverge. 
The loss curve will have a jagged or irregular shape, and it may not converge to an optimal solution. The loss may jump around or increase, indicating that the optimizer overshoots the minimum and may not find a good solution. 
In this case, the model may not learn much from the training data, resulting in poor performance.

When the learning rate is too low, the training process is slow, and the loss reduction is gradual. 
The loss curve will have a gentle slope, and it may take many epochs to converge to an optimal solution. 
The loss may plateau at a relatively high value, indicating that the optimizer is stuck in a local minimum or saddle point. 
In this case, the model may not learn much from the training data, resulting in poor performance. **

**Second question: The model is highly overfitted to the training set, since the training accuracy is much higher, than the test accuracy. 
You can see that the training accuracy is continuously increasing while the test accuracy stagnates and stays at the same level.
So the model trains only very well on the training set but it doesn´t generalize so the test accuracy doesn´t increase
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**The ideal residual plot should be a random scatter of points that form a constant around the y-^y=0 horizontal line.
The trained model behaves quite well since the residual plots are roughly scattered points around the horizontal $y-\hat{y}=0$ line.
The plot for the top-5 features displays a dense spot of points between $\hat{y}=10$ and $\hat{y}=20$ but has many outliers in the extreme values of $\hat{y}$. The residual plot after CV shows an improvement in this respect, because it has less outliers. We can also observe that dashed line, which represents the dots' standard deviation, has shrunk closer to the $y-\hat{y}=0$ line.
In addition to these visible changes, we see that the mean-squared error has diminished ($27.20$ to $12.41$ (train) and $19.51$ (test)) while the $R^2$ score has soared from $0.68$ to $0.86$ (train) and $0.71$ (test).**



"""

part4_q2 = r"""
**1. Adding non-linear features to the data is like adding a feature to the dataset, it does not change the fact that we can use a linear regression model to predict the data.
The outcome is still predicted with a linear combination of the parameters.
2. For this reason, we can use a linear regression model to fit any non-linear function of the original features.
3. Since we have added features to the data, the decision boundary is now a hyperplane in a higher-dimensional space.**




"""

part4_q3 = r"""
**When doing the gridsearch in the cross-validation code, we want to find the best combination of its hyperparameters in respect to model performance. Choosing a logscale instead of a linear scale enables us to span a broader amount of values for $\lambda$. The logarithmic scale also ensures that the range is not too skewed towards high or low values, as could happen with a linear scale.

For each step of the k-folds cross-validation, the model was fitted k times, and here we used k=3.**

"""

# ==============
