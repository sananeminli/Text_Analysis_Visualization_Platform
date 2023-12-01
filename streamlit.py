import streamlit as st
import requests
import googleapiclient.discovery
import json
import pandas as pd
from textblob import TextBlob 
from langdetect import detect  
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


load_dotenv()

youtube_api_key = os.getenv("YOUTUBE_API_KEY")

tagme_api_key = os.getenv("TAGME_API_KEY")

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)




def search_youtube(query, max_results=10):
    try:
       
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id",
            maxResults=max_results,
            eventType="completed"
        ).execute()

        # Extract video IDs from the search results
        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]

        return video_ids

    except googleapiclient.errors.HttpError as e:
        print(f"YouTube API error occurred: {e}")
        return []


def get_video_comments(video_id, max_results=100):
    try:
        # Call the commentThreads.list method to retrieve comments for a video
        comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=max_results
        ).execute()

        # Extract comments from the response
        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in
                    comment_response.get("items", [])]

        return comments

    except googleapiclient.errors.HttpError as e:
        error_details = e.error_details[0]
        if error_details["reason"] == "commentsDisabled":
            print(f"Comments are disabled for the video with ID: {video_id}")
            return []
        else:
            print(f"YouTube API error occurred: {e}")
            return []


# Updated TagMe linking function
def tagme_entity_linking(text):

   
    tagme_endpoint = "https://tagme.d4science.org/tagme/tag"

  
    params = {
        "text": text,
        "gcube-token": tagme_api_key,
        "lang": "en",
        "long_text":0
    }

    try:
        response = requests.get(tagme_endpoint, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
       
        

        # Extract entities from the TagMe response
        entities = []
        if "annotations" in result:
            for annotation in result["annotations"]:
                # Check for the existence of 'title' in the annotation
                if "title" in annotation and annotation["link_probability"]>0.3 and annotation["rho"]>0.2:
                    entities.append(annotation["title"])

        return entities

    except requests.exceptions.HTTPError as e:
        print(f"TagMe API HTTP error occurred: {e}")
        print(text)
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during the TagMe API request: {e}")
        return []
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding TagMe API JSON: {e}")
        return []

# Function for sentiment analysis
def analyze_sentiment(comment):
    blob = TextBlob(comment)
    return blob.sentiment.polarity

def detect_language(comment):
    try:
        language = detect(comment)
        return language
    except:
        return None
    

def link_entities(comment):
    entities = tagme_entity_linking(comment)
    return entities



st.title("Text Analysis App")
query = st.text_input("Enter YouTube search query:")
button_pressed = st.button("Start Analysis")

if 'wordcloud_image' not in st.session_state:
        # If not, initialize it
        st.session_state.wordcloud_image = []
if 'negative_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.negative_entities = []
if 'positive_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.positive_entities = []
if 'sentiment' not in st.session_state:
        # If not, initialize it
        st.session_state.sentiment = ''
def analyze():
    if len(st.session_state.wordcloud_image)>0:
        for path in st.session_state.wordcloud_image:
            os.remove(path)
    video_ids = search_youtube(query)
    wordcloud_images = []
    all_comments_data = {'Comment': [], 'Polarity': []}
    update = st.empty()
    update.text("💭 Getting comments from videos...") 
    # Get comments for each of the first four videos
    for video_id in video_ids:
        comments = get_video_comments(video_id)
        for comment in comments:
            polarity = analyze_sentiment(comment)
            # Detect the language of the comment
            language = detect_language(comment)
            # If the polarity is not zero and the language is English, add the comment and polarity to the DataFrame
            if language == 'en':
                all_comments_data['Comment'].append(comment)
                all_comments_data['Polarity'].append(polarity)
           
    update.text("👍 Done")
    comments_df = pd.DataFrame(all_comments_data)
    # Count the number of positive and negative comments
    positive_comments = comments_df[comments_df['Polarity'] > 0]
    negative_comments = comments_df[comments_df['Polarity'] < 0]
    step_size = 10
    array_of_negative_parts = []
    array_of_positive_parts = []
    update.text("📚 Doing sentiment analysis...")
    # Loop through the Negative Comments DataFrame in steps of 10
    for i in range(0, len(negative_comments), step_size):
    
        df_part = negative_comments.iloc[i:i + step_size]
        concatenated_string = '.'.join(df_part['Comment'])
        son = concatenated_string.replace('\n', '')
        son_str  = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', son)
        # Append the result to the list
        array_of_negative_parts.append(son_str)
    update.text("👍 Done")
    # Loop through the Positive Comments DataFrame in steps of 10
    for i in range(0, len(positive_comments), step_size):
    
        df_part = positive_comments.iloc[i:i + step_size]
        concatenated_string = '.'.join(df_part['Comment'])
        son = concatenated_string.replace('\n', '')
        son_str  = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', son)
        # Append the result to the list
        array_of_positive_parts.append(son_str)
    
    negative_entities = []
    positive_entities = []

    update.text("🔗 Getting entity links...")
    for i in range(0, len(array_of_negative_parts)): 
        negative_entities.append(link_entities(array_of_negative_parts[i]))
    for i in range(0, len(array_of_positive_parts)): 
        positive_entities.append(link_entities(array_of_positive_parts[i]))
    st.session_state.positive_entities = positive_entities
    st.session_state.negative_entities = negative_entities 
    update.text("👍 Done")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(comments_df['Comment'])
    silhouette_scores = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    # Choosing the ideal number of clusters
    ideal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 because we started from k=2
    # Performing clustering with the ideal number of clusters
    kmeans = KMeans(n_clusters=ideal_num_clusters, random_state=42)
    comments_df['Cluster'] = kmeans.fit_predict(X)
    update.text("📊 Creating plots... ")
    for cluster_num in range(ideal_num_clusters):
        cluster_comments = comments_df[comments_df['Cluster'] == cluster_num]['Comment']
        
        # Combine all comments in the cluster into a single string
        cluster_text = ' '.join(cluster_comments)
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
        # Plotting the word cloud
        image_path = f'wordcloud_cluster_{cluster_num}.png'
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster {cluster_num}')
        plt.savefig(image_path)
        plt.close()
        wordcloud_images.append(image_path)
    st.session_state.wordcloud_image = wordcloud_images
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(['Positive', 'Negative'], [len(positive_comments), len(negative_comments)])
    ax1.set_ylabel('Number of Comments')
    ax1.set_title('Distribution of Positive and Negative Comments')
    fig.delaxes(ax2)
    plt.tight_layout()
    plt.savefig('sentiment.png')
    st.session_state.sentiment = 'sentiment.png'
    negative_entities_for_word_cloud = " ".join(array_of_negative_parts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_entities_for_word_cloud)
    # WordCloud 
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Entities')
    
    positive_entities_for_word_cloud = " ".join(array_of_positive_parts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_entities_for_word_cloud)
    # WordCloud 
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Entities')
    update.empty()
    st.success("Analysis completed! :ok_hand:")
    
with st.spinner("Analyzing... :mag:"):
    if button_pressed:
        analyze()

    
        

if st.session_state.sentiment =='sentiment.png':
    st.subheader("📊 Distrubition of comment sentiment")
    st.image(st.session_state.sentiment , caption='Distrubition of comments')

if st.session_state.wordcloud_image and len(st.session_state.wordcloud_image)>0:
    st.subheader("Word Clouds")
    selected_image_path = st.selectbox("Select an Image", st.session_state.wordcloud_image)
    print(st.session_state.wordcloud_image)
    # Streamlit UI
    st.subheader("KMeans Clustering Analysis")

    image_container = st.empty()


    image_container.image(selected_image_path, caption='Selected Image')

if len(st.session_state.positive_entities) > 0 and len(st.session_state.negative_entities) > 0:
    neg_col, pos_col = st.columns(2)
    neg_col.subheader("Entities in negative comments.")
    pos_col.subheader("Entities in positive comments.")
    neg_set = set()
    pos_set = set()
    for result in st.session_state.positive_entities:
        for link in result:
            title = link
            link = link.replace(" ", "_")
            pos_set.add(link)
    
    for result in st.session_state.negative_entities:
        for link in result:
            title = link
            link = link.replace(" ", "_")
            neg_set.add(link)
    for text in neg_set:
        neg_col.markdown(f"[Wikipedia](https://en.wikipedia.org/wiki/{text}) {text}")
    for text in pos_set:
        pos_col.markdown(f"[Wikipedia](https://en.wikipedia.org/wiki/{text}) {text}")



    


