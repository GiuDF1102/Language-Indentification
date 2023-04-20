# MachineLearningProject2023
Select a candidate between more different possible alternatives (Best performance).

## Protocol (Compulsory)
- The Training data can be use to estimate model parameters. So the train data has to be used only to build the model.
- The Validation set can be taken ONLY from the training data (maybe using k-fold cross-validation). Should be left aside and used to evaluate the possible alternatives. 
- The Test set can be used EXCLUSIVELY for evaluation.

## Evaluation
Once we have chosena a model, we need the test set (evaluation in the document). You cannot estimate anything on the evaluation test. However the evaluation should depend on the target application.

## Post-evaluation
Show the differences between the selected model and the discarded ones.

## About "Target Application"
"Sometimes it is better to have more false positive guesses than false negative guesses."

## Projects
### Gender Identification
Consisting of gender identification from high-level features extracted from face images (we assume that the features have already been extracted and given as embeddings, fixed dimensional vector of numbers). The embeddings are synthetic data, they don't come from real samples.

#### Classes
Keep attention that CLASSES ARE NOT BALANCED. The samples come from threee different age groups, which may have different characteristics for male and female. Young males might be different old males, but the label is not given.

### Language Detection
Detect wheter an utterance is spoken in a target language (italian or not -> binary problem). The data is also already extracted into embeddings.

#### Classes
We have six components of each vector, classes are not balanced, the target class has significantly less samples than the other languages (25 possible non-target languages). We need to optimize on two target applications.

### Fingerprint spoofing
The goal is to detect wheter a fingerprint image is authentic or spoofed. We still work with embeddings, and classify them. 

#### Classes
There are 10 continous features, the classes are imbalanced (but not too much). We need to optimize on one only target application.


### Biometric identity verification
The goal is to detect wheter a pair of embeddings belongs to the same person. 

#### Classes
The training set is built to make this project as complex as the others, but check out the documentation. Classes are also slightly imbalanced.

Lalala
