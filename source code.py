!pip install pandas numpy scikit-learn openpyxl seaborn matplotlib surprise

!pip install numpy==1.24.4

from google.colab import files 
import pandas as pd
# Upload your Excel file
uploaded = files.upload()
# Read the file
file_path = next(iter(uploaded))
df = pd.read_excel(file_path)
# Preview
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine content features
df['combined_features'] = df['Category'].astype(str) + ' ' + df['Item_ID'].astype(str)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index map for items
indices = pd.Series(df.index, index=df['Item_ID']).drop_duplicates()

# Function to get content-based recommendations
def content_based_recommend(item_id, num_recommendations=10):
    if item_id not in indices:
        return f"Item_ID '{item_id}' not found."
    idx = indices[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    item_indices = [i[0] for i in sim_scores]
    return df[['Item_ID', 'Category']].iloc[item_indices]

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Use Surprise to prepare dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['User_ID', 'Item_ID', 'Rating']], reader)

# Split into training and testing
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Train SVD model
model = SVD()
model.fit(trainset)

# Test RMSE
predictions = model.test(testset)
rmse(predictions)

# Predict function
def predict_rating(user_id, item_id):
    return model.predict(user_id, item_id).est

def hybrid_recommend(user_id, item_id, top_n=10, weight_cb=0.5, weight_cf=0.5):
    if item_id not in indices:
        return f"Item_ID '{item_id}' not found."

    idx = indices[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n*2+1]

    hybrid_scores = []
    for i, score in sim_scores:
        candidate_id = df['Item_ID'].iloc[i]
        cb_score = score
        cf_score = predict_rating(user_id, candidate_id)
        final_score = (weight_cb * cb_score) + (weight_cf * (cf_score / 5))
        hybrid_scores.append((candidate_id, final_score))

    top_recommendations = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(top_recommendations, columns=['Recommended Item_ID', 'Score'])

# Content-based
print("Content-Based Recommendations:")
print(content_based_recommend('Item_52'))

# Predict individual rating
print("Collaborative Prediction for User_913 & Item_52:")
print(predict_rating('User_913', 'Item_52'))

# Hybrid
print("Hybrid Recommendations:")
print(hybrid_recommend('User_913', 'Item_52'))
 
