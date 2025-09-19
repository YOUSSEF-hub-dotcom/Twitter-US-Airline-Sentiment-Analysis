import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.metrics import classification_report,accuracy_score
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm
import nlpaug.augmenter.word as naw

df = pd.read_csv(r"C:\Users\youss\OneDrive\python dataset\Tweets.csv")
pd.set_option('display.width',None)
print(df.head(11))

print('=======================>>> Analysis Function')
print("Information about Data:")
print(df.info())
print("Statistical Data Analysis:")
print(df.describe().round(2))
print('Number of rows and columns:')
print(df.shape)
print('Columns in Data')
print(df.columns)
print("Frequency of Rows in Data:")
print(df.duplicated().sum())      # 36

print("=======================>>> Cleaning Data:")
print("Remove Duplicates")
df = df.drop_duplicates(keep='first')
print(df.shape)

print("Missing Values Before Cleaning:")
print(df.isnull().sum())

missing_values = df.isnull().mean() * 100
print("The Percentage of missing values in data:\n",missing_values)

print(
"The percentage of missing Value in negativereason column is: 37% \n",
"The percentage of missing Value in negativereason_confidence column is: 28% \n",
"The percentage of missing Value in tweet_location column is: 32% \n",
"The percentage of missing Value in user_timezone column is: 32% \n",
"According to the percentage of missing value in latest columns we use fillna(Median & Mode)"
)

df['negativereason'] = df['negativereason'].fillna(df['negativereason'].mode()[0])
df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].median())
df['tweet_location'] = df['tweet_location'].fillna(df['tweet_location'].mode()[0])
df['user_timezone'] = df['user_timezone'].fillna(df['user_timezone'].mode()[0])

print("There is a missing Value in Data:")
print(df[['negativereason','negativereason_confidence','tweet_location']].isnull().sum())
print("---------------------------------------------------")

print(
"The Percentage of missing value in tweet_coord column is: 93% \n",
"The Percentage of missing value in negativereason_gold column is: 99% \n",
"The Percentage of missing value in airline_sentiment_gold column is: 99% \n",
"According to the percentage of missing value in latest columns we use drop "
)

df = df.drop(columns=['tweet_coord','negativereason_gold','airline_sentiment_gold'],axis=1)

print("Missing Values After Cleaning :")
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title('The missing value in Tweets Dataset ')
plt.show()

print("===================>>> EDA & Text Preprocessing")
#Index(['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence','negativereason', 'negativereason_confidence', 'airline', 'name'
#'retweet_count', 'text', 'tweet_created','tweet_location', 'user_timezone']

print("Convert tweet_created column to datetime")
df['tweet_created'] = pd.to_datetime(df['tweet_created'])
print(df.dtypes)

print("Distribution of Sentiments in the Dataset:")
print(df['airline_sentiment'].value_counts())

print("Number of Tweets per Airline:")
print(df['airline'].value_counts())

nltk.download('punkt')
nltk.download('stopwords')



df['lower_text'] = df['text'].str.lower()

df['tokenized_text'] = df['lower_text'].apply(nltk.word_tokenize)

df['no_specials'] = df['tokenized_text'].apply(lambda x: [re.sub(r'[^a-zA-Z]', '', word) for word in x])

stop_words = set(stopwords.words('english'))
df['no_stopwords'] = df['no_specials'].apply(lambda x: [word for word in x if word not in stop_words and word != ''])
stemmer = PorterStemmer()
df['stemmed_tokens'] = df['no_stopwords'].apply(lambda tokens:[stemmer.stem(word) for word in tokens])

lemmatizer = WordNetLemmatizer()
df['lemmatized_text'] = df['no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

df['cleaned_text'] = df['lemmatized_text'].apply(lambda x: " ".join(x))

print(df[['text', 'lower_text', 'tokenized_text', 'no_specials', 'no_stopwords', 'stemmed_tokens', 'cleaned_text']].head(5))

print("===================>>> Visualization")

plt.figure(figsize=(6,4))
sns.countplot(x="airline_sentiment", data=df, order=df['airline_sentiment'].value_counts().index, palette="Set2")
plt.title("Distribution of Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="airline", hue="airline_sentiment", data=df, palette="Set1")
plt.title("Sentiments per Airline")
plt.xlabel("Airline")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.grid()
plt.show()

positive_text = " ".join(df[df['airline_sentiment']=='positive']['cleaned_text'])
negative_text = " ".join(df[df['airline_sentiment']=='negative']['cleaned_text'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Positive Tweets")
plt.show()

wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap="Reds").generate(negative_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Negative Tweets")
plt.show()


sentiment_trend = df.groupby([df['tweet_created'].dt.date, 'airline_sentiment']).size().reset_index(name='counts')

plt.figure(figsize=(12,6))
sns.lineplot(data=sentiment_trend, x="tweet_created", y="counts", hue="airline_sentiment", palette="Set2")
plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.grid()
plt.show()

print("===========================>>>> Deep Learning (Beart)")
import nltk

nltk.download('averaged_perceptron_tagger_eng')

nltk.download('wordnet')
nltk.download('omw-1.4')

df['label'] = df['airline_sentiment'].map({'negative':0, 'neutral':1, 'positive':2})
df = df[['cleaned_text', 'label']]
print(df.head())

max_count = df['label'].value_counts().max()
aug = naw.SynonymAug(aug_src='wordnet')

augmented_texts, augmented_labels = [], []
for label in df['label'].unique():
    subset = df[df['label'] == label]
    current_count = len(subset)
    needed = max_count - current_count

    augmented_texts.extend(subset['cleaned_text'])
    augmented_labels.extend(subset['label'])

    generated_texts = set()
    while len(generated_texts) < needed:
        row = subset.sample(1).iloc[0]
        new_text = aug.augment(row['cleaned_text'])
        if isinstance(new_text, list):
            new_text = " ".join(new_text)
        if new_text not in generated_texts and new_text not in subset['cleaned_text'].values:
            generated_texts.add(new_text)

    augmented_texts.extend(list(generated_texts))
    augmented_labels.extend([label]*len(generated_texts))

balanced_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
print("Balanced labels:\n", balanced_df['label'].value_counts())

train_df, test_df = train_test_split(
    balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label']
)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=128, return_tensors='pt')

train_labels = torch.tensor(list(train_df['label']))
test_labels = torch.tensor(list(test_df['label']))

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    print(f"Epoch {epoch+1} finished")


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=['negative','neutral','positive']))

print('================================================================================================================')
# ✅ Save the model and vectorizer
model.save_pretrained("sentiment_model")

tokenizer.save_pretrained("sentiment_model")

print("Model and Vectorizer saved successfully!")