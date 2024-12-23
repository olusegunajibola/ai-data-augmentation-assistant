#%% md
# # Data Augmentation
# 
# Here we balance out the datasets using https://groq.com/ free API.
#%%
import os
import pandas as pd
from groq import Groq
#%%
# Set up your Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
print(client)
#%% md
# Read dataframe.
#%%
news = pd.read_csv('../data/financial_news.csv',
                   names=['sentiment', 'news'])
#%%
 # Select 10 positive news
positive_news = news[news['sentiment'] == 'positive'].sample(10)
# Select 5 negative news
negative_news = news[news['sentiment'] == 'negative'].sample(6)

# Combine the results
selected_news = pd.concat([positive_news, negative_news])
#%%
selected_news.to_csv('../data/selected_news.csv', index=False)
#%%
selected_news
#%% md
# Now we do data augmentation for the selected texts, by making the value of negative news equal that of the positive.
#%%
selected_news.sentiment.value_counts()
#%%
df = selected_news.copy()
#%%
# Number of rows we want for label negative
target_rows_label_negative = 10

# Find underrepresented rows (label == 0)
underrepresented_texts = df[df['sentiment'] == 'negative']['news'].tolist()

# Number of examples we currently have for label 0
current_rows_label_negative = len(underrepresented_texts)

# Number of additional examples we need
needed_examples = target_rows_label_negative - current_rows_label_negative
#%%
underrepresented_texts
#%%
# Augment the underrepresented class with new examples
augmented_texts = []
for i in range(needed_examples):
    # Select a random text from the underrepresented class to augment
    text = underrepresented_texts[i % current_rows_label_negative]  # Cycle through available texts if needed
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a data augmentation assistant."},
            {"role": "user", "content": f"Generate a concise headline similar to: {text} and reply with response only without quotes"},
        ],
        model="llama3-8b-8192"
    )
    
    # Get the augmented text from the response
    augmented_data = response.choices[0].message.content
    augmented_texts.append(augmented_data)
#%%
# Create a new DataFrame for the augmented data
augmented_df = pd.DataFrame({
    'news': augmented_texts,
    'sentiment': ['negative'] * needed_examples  # Label the new examples as 0
})

# Combine the original and augmented DataFrames
balanced_df = pd.concat([df, augmented_df], ignore_index=True)

# Display the balanced DataFrame
# print(balanced_df)
#%%
balanced_df
#%%
balanced_df.sentiment.value_counts()
#%%
balanced_df.to_csv('../data/balanced_news.csv', index=False)
#%% md
# This approach can be applied to other text based applications.
#%%
