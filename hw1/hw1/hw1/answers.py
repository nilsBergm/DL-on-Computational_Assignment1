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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
# Part 3 answers

part3_q1 = r"""
**Your answer:**


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
