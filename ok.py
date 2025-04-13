# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('C:/Users/hp/Desktop/twitter/Twitter Sentiments.csv')
print("Dataset loaded successfully.\n")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display datatype info
print("\nData types and non-null counts:")
df.info()

# Function to remove specific patterns from text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

print("\nCleaning tweets...")

# Remove Twitter handles
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

# Remove special characters, numbers, and punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

# Remove short words (less than 4 characters)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

# Tokenize tweets
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())

# Stemming
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

# Combine tokens back into string
df['clean_tweet'] = tokenized_tweet.apply(lambda x: " ".join(x))

print("Tweet cleaning completed.\n")

# Generate word cloud for entire corpus
print("Generating overall word cloud...")
all_words = " ".join([sentence for sentence in df['clean_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Overall WordCloud")
plt.show()

# Word cloud for non-racist/sexist tweets (label == 0)
print("Generating word cloud for non-racist/sexist tweets...")
all_words_pos = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words_pos)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Non-Racist/Sexist WordCloud")
plt.show()

# Word cloud for racist/sexist tweets (label == 1)
print("Generating word cloud for racist/sexist tweets...")
all_words_neg = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words_neg)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Racist/Sexist WordCloud")
plt.show()

# Extract hashtags
def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags    

print("\nExtracting hashtags...")
ht_positive = hashtag_extract(df['clean_tweet'][df['label'] == 0])
ht_negative = hashtag_extract(df['clean_tweet'][df['label'] == 1])

# Unnest lists
ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])

# Positive hashtags visualization
print("Most common hashtags in non-racist/sexist tweets:")
freq_pos = nltk.FreqDist(ht_positive)
df_pos = pd.DataFrame({'Hashtag': list(freq_pos.keys()), 'Count': list(freq_pos.values())})
df_pos = df_pos.nlargest(columns='Count', n=10)

plt.figure(figsize=(15,9))
sns.barplot(data=df_pos, x='Hashtag', y='Count')
plt.title("Top 10 Hashtags - Non-Racist/Sexist Tweets")
plt.show()

# Negative hashtags visualization
print("Most common hashtags in racist/sexist tweets:")
freq_neg = nltk.FreqDist(ht_negative)
df_neg = pd.DataFrame({'Hashtag': list(freq_neg.keys()), 'Count': list(freq_neg.values())})
df_neg = df_neg.nlargest(columns='Count', n=10)

plt.figure(figsize=(15,9))
sns.barplot(data=df_neg, x='Hashtag', y='Count')
plt.title("Top 10 Hashtags - Racist/Sexist Tweets")
plt.show()

# Bag of Words Feature Extraction
print("\nExtracting features using Bag of Words...")
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], test_size=0.25, random_state=42)

# Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(x_train, y_train)

# Predictions and evaluation
print("Evaluating model...")
pred = model.predict(x_test)

print("F1 Score (default threshold=0.5):", f1_score(y_test, pred))
print("Accuracy Score:", accuracy_score(y_test, pred))

# Using probability threshold = 0.3
pred_prob = model.predict_proba(x_test)
pred_custom = (pred_prob[:, 1] >= 0.3).astype(int)

print("\nUsing custom threshold (0.3):")
print("F1 Score:", f1_score(y_test, pred_custom))
print("Accuracy Score:", accuracy_score(y_test, pred_custom))
print("Example: Probability of first prediction =", pred_prob[0][1], "→ Predicted label =", pred_custom[0])
