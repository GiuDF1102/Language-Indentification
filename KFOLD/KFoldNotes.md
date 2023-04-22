# K-Fold
## Why K-Fold
The split, is standardly done this way: with the train set, we can estimate the parameters, and with the Validation set we test their accuracy. The split is decided picking a fraction for the split.

But using this technique, sometimes the validation set doesn't contain enough samples. About the training set, the more samples I have the better will be the estimation of my parameters.

## K-Fold Approach
It consists in dividing the dataset in k equal folds. The validation set is iteratively rotated.

I will train a model using the first k-1 folds as the training part and the kth for the validation. This is repeated k times, as each time we use a different fold as validation fold.

After computing the scores, I can extract the accuracy. However now I used all the dataset as training and validation.

After finding the right model, I can build it using the whole training set. 

This way I have:
- More robust estimate
- More robust training
