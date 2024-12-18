# Sentiment Analysis Project

## Context / Challenge

### Problem Statement
The primary problem addressed in this project is **sentiment analysis**: determining whether labeled sentences convey a positive or negative sentiment. This task has broad applications in marketing, product development, and social media analysis, enabling companies to gauge customer opinions and preferences.

### Why It Is Challenging
1. **Ambiguity and Subtlety**: Language can be nuanced and context-dependent.
2. **Sarcasm and Tone**: Text often includes tones like sarcasm, making classification difficult.
3. **Contextual Variations**: Sentiment varies based on content and word usage.

### Solving Through Machine Learning
Machine learning can analyze large amounts of text data to detect patterns and classify sentiments. Techniques like Natural Language Processing (NLP) and various classification models allow machines to extract meaning from text at scale.

### Aspects Addressed
The project compares the effectiveness of three machine learning models for sentiment analysis, focusing on accuracy and efficiency.

### Relevance
Sentiment analysis is crucial for industries that rely on understanding public opinion and customer satisfaction. By comparing techniques, this project can guide the selection of optimal models for applications such as product feedback analysis or sentiment tracking in social media.

## Dataset

The project uses the **Sentiment Labeled Sentences Data Set** from the UCI Machine Learning Repository. It contains 3,000 sentences labeled as positive or negative, extracted from reviews on Amazon, IMDb, and Yelp. The entire dataset is utilized due to its manageable size, ensuring faster training and sufficient data for cross-validation.

## Proposed Solution

### Problem Type
This is a **predictive** problem: classifying new sentences as positive or negative based on learned patterns.

### Machine Learning Techniques
1. **Support Vector Machines (SVM)**
2. **Logistic Regression (LR)**
3. **Naive Bayes (NB)**

These models are chosen for their proven performance in sentiment analysis and their compatibility with different preprocessing techniques.

### Data Preprocessing
1. **Tokenization**
2. **Stopword Removal**
3. **Lemmatization**
4. **Vectorization** using TF-IDF

### Evaluation Strategy
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Cross-validation**: Ensures robustness against overfitting or underfitting.

## Libraries Used
- **Visualization**: `matplotlib`
- **NLP and Data Handling**: `NLTK`, `pandas`, `numpy`, `re`, `WordNetLemmatizer`, `sentence-transformers`
- **Modeling**: `scikit-learn`, `pickle`, `TensorFlow`
- **Evaluation**: `scikit-learn.metrics`

## Preprocessing Improvements


## Model Specifications

### Original Specifications
| Model Type         | Loss Function          | Epoch | Learning Rate | Batch Size | Notes                  |
|--------------------|------------------------|-------|---------------|------------|------------------------|
| Logistic Regression | Cross-Entropy          | 10,000 | 0.1           | Full       | Batch Gradient Descent |
| SVM                | Hinge Loss            | 10    | 0.0005        | Full       |                        |
| Naive Bayes        | Negative Log-likelihood | N/A   | N/A           | N/A        |                        |

### Improvements
- **Early Stopping**: Reduced training time by implementing thresholds for loss improvement.
- **Learning Rates**: Experimented with values to optimize convergence.
- **Vectorization Techniques Compared**:
  - Bag of Words (BoW)
  - TF-IDF
  - Sentence-BERT (SBERT)
- **Dimensionality Reduction**: Chi-Squared feature selection reduced the feature space from 4,535 to 2,100, improving Naive Bayes performance.

## Evaluation

### Bias-Variance Tradeoff
- **High Bias (Underfitting)**: Train error > 10%, difference between train and validation error < 5%.
- **High Variance (Overfitting)**: Train error < 5%, validation error > 15%.

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Results
#### Preliminary Results (Before Improvements)
| Model Type     | Accuracy | Precision | Recall | F1 Score | Train Error | Validation Error | Test Error |
|----------------|----------|-----------|--------|----------|-------------|------------------|------------|
| Logistic Regression | 79.33%  | 80.23%    | 79.33% | 79.22%   | 9.86%       | 22.99%           | 20.67%     |
| SVM            | 79.17%  | 79.31%    | 79.17% | 79.12%   | 14.09%      | 24.37%           | 20.83%     |
| Naive Bayes    | 82.67%  | 82.68%    | 82.67% | 82.67%   | 5.53%       | 20.65%           | 17.33%     |

#### Improved Results
SBERT vectorization significantly improved model performance:
| Model Type     | Accuracy | Precision | Recall | F1 Score | Train Error | Validation Error | Test Error |
|----------------|----------|-----------|--------|----------|-------------|------------------|------------|
| Logistic Regression (SBERT) | 94.17%  | 94.28%    | 94.17% | 94.17%   | 5.50%       | 6.17%            | 5.83%      |
| SVM (SBERT)    | 93.50%  | 93.59%    | 93.50% | 93.50%   | 7.23%       | 7.34%            | 6.50%      |
| NB (Chi Squared)    | 86.67%  | 86.73%    | 86.67% | 86.67%   | 7.44%       | 14.69%           | 13.33%      |

## Limitations
- **Dataset Size**: A larger dataset would improve model performance.
- **Sparse Matrices**: Preprocessing techniques like BoW and TF-IDF produce sparse matrices, making dimensionality reduction challenging.

## Conclusion
This project highlights the comparative strengths and weaknesses of different machine learning models and preprocessing techniques for sentiment analysis. Future work could focus on expanding the dataset and exploring advanced deep learning models.
