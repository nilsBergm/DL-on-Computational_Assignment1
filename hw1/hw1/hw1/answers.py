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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
