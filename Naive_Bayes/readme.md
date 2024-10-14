# Sentiment Analysis using Naive Bayes

This project performs sentiment analysis on a dataset of movie reviews using three variants of Naive Bayes classifiers: **GaussianNB**, **MultinomialNB**, and **BernoulliNB**. The dataset is preprocessed to clean the text, convert it into numerical features, and the models are trained using both traditional train-test splitting and **k-fold cross-validation** to ensure robustness. The performance is evaluated based on accuracy and confusion matrix visualizations.

## Project Structure

The core steps involved in this project include:
1. **Loading the Dataset**: Loading the IMDB dataset containing 50,000 movie reviews.
2. **Preprocessing the Data**: Text data is cleaned by:
   - Removing HTML tags
   - Converting to lowercase
   - Removing special characters
   - Removing stopwords
   - Stemming the words
3. **Feature Extraction**: The cleaned text is converted to numerical representation using a vectorizer.
4. **Modeling**: Three Naive Bayes models are used:
   - **GaussianNB**: Assumes features follow a normal distribution.
   - **MultinomialNB**: Suitable for discrete counts, commonly used for text classification.
   - **BernoulliNB**: Suitable for binary or boolean features.
5. **Evaluation**: Models are evaluated using accuracy and confusion matrices. Additionally, **k-fold cross-validation** is used to ensure model robustness.
6. **Visualizations**: The project includes various visualizations to understand data distribution and model performance.

## Getting Started

### Prerequisites

The following Python libraries are required to run the project:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk

These can be installed using `pip`.

### Dataset

The dataset used in this project is the **IMDB Dataset of 50K Movie Reviews**, which contains two columns:
- `review`: The text of the review.
- `sentiment`: Whether the review is `positive` or `negative`.

### Steps to Run the Code

1. **Clone the Repository**:
   - Clone the repository and navigate to the project directory.
   
2. **Run the Python Script/Notebook**:
   - Open the Jupyter notebook or Python script and run the code.

### Detailed Explanation of Each Section

#### 1. Data Loading and Exploration

The dataset is loaded into a Pandas DataFrame, and the first few rows are explored to understand its structure, containing movie reviews and their corresponding sentiments.

#### 2. Data Cleaning and Preprocessing

This section involves cleaning the text data to prepare it for machine learning. The cleaning process includes:
- **Removing HTML tags**: To eliminate any leftover HTML code from the text.
- **Converting to lowercase**: Ensures uniform text data.
- **Removing special characters**: Keeps only alphanumeric characters for simplicity.
- **Removing stopwords**: Removes common words like 'the', 'is', etc., which do not add significant value for sentiment analysis.
- **Stemming**: Reduces words to their root form, e.g., "loving" becomes "love".

#### 3. Feature Extraction

Once the data is cleaned, the next step is to convert the text into numerical features. This is done using a vectorizer, which converts each review into a vector of token counts.

#### 4. Modeling and Cross-Validation

Three types of Naive Bayes classifiers are trained on the dataset:
- **GaussianNB**: Best suited for normally distributed data.
- **MultinomialNB**: Effective for text data where features represent counts.
- **BernoulliNB**: Works well for binary or boolean features.

In addition to a regular train-test split, **k-fold cross-validation** is employed to provide a more robust evaluation of the models. The dataset is split into k (e.g., 5) subsets, and the model is trained and tested k times on different subsets to ensure it generalizes well.

#### 5. Visualizations

- **Sentiment Distribution**: A bar chart shows the distribution of positive and negative sentiments in the dataset.
- **Confusion Matrix**: Confusion matrices are plotted for each Naive Bayes model, showing the breakdown of true positives, true negatives, false positives, and false negatives.
- **Top 20 Frequent Words**: A bar chart visualizes the most frequent words in the dataset after cleaning, providing insights into the most common words used in the reviews.

## Results and Evaluation

The accuracy of each model on the test set and after k-fold cross-validation is reported:
- **Gaussian Naive Bayes**: Around 71% accuracy
- **Multinomial Naive Bayes**: Around 83% accuracy
- **Bernoulli Naive Bayes**: Around 83.5% accuracy

These results show that **Multinomial** and **Bernoulli Naive Bayes** are particularly well-suited for this type of text data.

## Conclusion

This project demonstrates how Naive Bayes models can be effectively applied to sentiment analysis tasks. By using cross-validation and detailed preprocessing, the models are made robust, and their generalization ability is improved. Visualizations further help in understanding the underlying data and model performance.

