import streamlit as st
import requests
import googleapiclient.discovery
import json
import pandas as pd
from textblob import TextBlob 
from langdetect import detect  
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
import mpld3
import streamlit.components.v1 as components




load_dotenv()

youtube_api_key = os.getenv("YOUTUBE_API_KEY")

tagme_api_key = os.getenv("TAGME_API_KEY")

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)




def search_youtube(query, max_results):
    try:
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=max_results,
            order="relevance",
            regionCode="US",
            relevanceLanguage="en"
        ).execute()

        videos = search_response.get("items", [])

        # Exclude live videos
        completed_videos = [video for video in videos if 'liveBroadcastContent' in video['snippet'] and video['snippet']['liveBroadcastContent'] == 'none']

        st.session_state.videos = completed_videos

    except googleapiclient.errors.HttpError as e:
        print(f"YouTube API error occurred: {e}")
        return []


def get_video_comments(video_id, max_results):
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
        # Extract entities from the TagMe response
        entities = []
        entity_pos = []
        if "annotations" in result:
            for annotation in result["annotations"]:
                # Check for the existence of 'title' in the annotation
                if "title" in annotation and annotation["link_probability"]>0.3 and annotation["rho"]>0.2:
                    entities.append(annotation["title"])
                    entity_pos.append(annotation["start"])

        return entities , entity_pos

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


def generate_network_graph(G , pos):

    fig, ax = plt.subplots(figsize=(20,5)) 
    nx.draw(G,pos, with_labels=True,node_color='white',  font_size=10, bbox=dict(boxstyle="round,pad=0.3", alpha=0.5))
    # Use mpld3 to save the figure as HTML
    html_content = mpld3.fig_to_html(fig)
    #mpld3.show(fig=fig, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None)  
    return html_content

def sna(G):
    degree_centrality = nx.degree_centrality(G  )
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=300)

    centralities_dict = {
    'Node': list(G.nodes()),
    'Degree Centrality': list(degree_centrality.values()),
    'Closeness Centrality': list(closeness_centrality.values()),
    'Betweenness Centrality': list(betweenness_centrality.values()),
    'Eigenvector Centrality': list(eigenvector_centrality.values())}
    return  pd.DataFrame(centralities_dict)






     

st.title("Text Analysis App")
query = st.text_input("Enter YouTube search query:")
max_video_num = st.number_input("Enter the maximum number of video:", min_value=1, max_value=100, step=1)

search = st.button("Search videos!")

if 'wordcloud_image' not in st.session_state:
        # If not, initialize it
        st.session_state.wordcloud_image = []
if 'negative_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.negative_entities = []
if 'positive_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.positive_entities = []
if 'neutural_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.neutural_entities = []
if 'sentiment' not in st.session_state:
        # If not, initialize it
        st.session_state.sentiment = ''
if 'graph_pos' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_pos = None
if 'graph_neg' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_neg = None
if 'graph_neu' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_neu = None
if 'pos' not in st.session_state:
        # If not, create an empty placeholder
        st.session_state.pos_pos, st.session_state.neg_pos   ,  st.session_state.neu_pos = None, None, None
if 'videos' not in st.session_state:    
    st.session_state.videos = []
if 'id' not in st.session_state:
    st.session_state.id = []





with st.spinner("Search... :mag:"):
    if search:
        if len(st.session_state.id)>0:
            st.session_state.id = []
        search_youtube(query , max_video_num)



if len(st.session_state.videos) > 0:
    with st.expander("Video results!"):
        for video in st.session_state.videos:
            video_id = video['id']['videoId']
            video_title = video['snippet']['title']
            video_description = video['snippet']['description']
            video_thumbnail = video['snippet']['thumbnails']['default']['url']
            channel_name = video['snippet']['channelTitle']

            # Create two columns layout
            col1, col2 = st.columns(2)

            # Column 1: Display checkbox
            checkbox_key = f"checkbox_{video_id}"
            is_checked = col2.checkbox(f"**{video_title}**\n\n{video_description}\n\nChannel: {channel_name}", key=checkbox_key)

            # Check if the checkbox was previously checked but is now unchecked
            was_checked = video_id in st.session_state.id
            if was_checked and not is_checked:
                # Remove video ID from the array
                st.session_state.id.remove(video_id)
            elif is_checked and not was_checked:
                # If checkbox is checked, add video ID to the array
                st.session_state.id.append(video_id)

            # Column 2: Display thumbnail and link
            col1.image(video_thumbnail, width=240)
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            col1.markdown(f"[Watch Video]({video_link})")

    # Display selected video IDs
    st.write('Selected Video IDs:', st.session_state.id)














def analyze():
    if len(st.session_state.wordcloud_image)>0:
        for path in st.session_state.wordcloud_image:
            os.remove(path)
    video_ids = st.session_state.id
    wordcloud_images = []
    all_comments_data = {'Comment': [], 'Polarity': [], 'Size': []}
    update = st.empty()
    update.text("💭 Getting comments from videos...") 
    # Get comments for each of the first four videos
    for video_id in video_ids:
        comments = get_video_comments(video_id , max_commen_result)
        for comment in comments:
            polarity = analyze_sentiment(comment)
            # Detect the language of the comment
            language = detect_language(comment)
            # If the polarity is not zero and the language is English, add the comment and polarity to the DataFrame
            if language == 'en':
                all_comments_data['Comment'].append(comment)
                all_comments_data['Polarity'].append(polarity)
                all_comments_data['Size'].append(len(comment))
           
    update.text("👍 Done")
    df = pd.DataFrame(all_comments_data)

# Specify the maximum length (400 in this case)
    max_length = 500

# Filter rows based on string length
    comments_df = df[df['Comment'].str.len() <= max_length]
    
    step_size = 5 

    # Count the number of positive and negative comments
    positive_comments = comments_df[comments_df['Polarity'] > 0]
    negative_comments = comments_df[comments_df['Polarity'] < 0]
    neutural_comments = comments_df[comments_df['Polarity'] == 0]
    step_size = 10
    array_of_negative_parts = []
    array_of_positive_parts = []
    array_of_neutural_parts = []

    update.text("📚 Doing sentiment analysis...")
# Loop through the Negative Comments DataFrame in steps of 10
# Loop through the Negative Comments DataFrame in steps of 10
    for i in range(0, len(negative_comments), step_size):
    
        df_part = negative_comments.iloc[i:i + step_size]

        concatenated_string = '.'.join(df_part['Comment'])

        #son = concatenated_string.replace('\n', '')
        #son_str  = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', son)
        # Append the result to the list

        array_of_negative_parts.append(concatenated_string)
    update.text("👍 Done")
    # Loop through the Positive Comments DataFrame in steps of 10
    for i in range(0, len(positive_comments), step_size):
    
        df_part = positive_comments.iloc[i:i + step_size]

        concatenated_string = '.'.join(df_part['Comment'])

        #son = concatenated_string.replace('\n', '')
        #son_str  = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', son)

        # Append the result to the list
        array_of_positive_parts.append(concatenated_string)
    
    
    for i in range(0, len(neutural_comments), step_size):
    
        df_part = neutural_comments.iloc[i:i + step_size]

        concatenated_string = '.'.join(df_part['Comment'])

        #son = concatenated_string.replace('\n', '')
        #son_str  = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', son)

        # Append the result to the list
        array_of_neutural_parts.append(concatenated_string)


    
    negative_entities = []
    negative_pos = []
    positive_entities = []
    positive_pos = []
    neutural_entities = []
    neutural_pos = []

    update.text("🔗 Getting entity links...")
    for i in range(0, len(array_of_negative_parts)): 
        ent,  pos = tagme_entity_linking(array_of_negative_parts[i])
        negative_entities.append(ent)
        negative_pos.append(pos)

    for i in range(0, len(array_of_positive_parts)): 
        ent,  pos = tagme_entity_linking(array_of_positive_parts[i])
        positive_entities.append(ent)
        positive_pos.append(pos)
    
    for i in range(0, len(array_of_neutural_parts)): 
        ent,  pos = tagme_entity_linking(array_of_neutural_parts[i])
        neutural_entities.append(ent)
        neutural_pos.append(pos)
    
    G = nx.Graph()
    df_comment_index = 0
    last_comment = 0
    df_size = len(negative_comments)
    for i in range(len(negative_pos)):
        for j in range(len(negative_pos[i])):
            if len(negative_pos[i])==0:
                continue
            indexxx = 0
            for x in range(df_comment_index , df_comment_index+5):
                if x==df_size:
                    break
                
                if indexxx < negative_pos[i][j] < indexxx+int(negative_comments.iloc[x, negative_comments.columns.get_loc("Size")]):
                    print(negative_entities[i][j]) #entitiy
                    #print(negative_comments.iloc[x, negative_comments.columns.get_loc("Comment")]) #comment
                    print(x) # comment loc
                    if last_comment ==x and last_comment!=0:
                        G.add_edge(negative_entities[i][j-1],negative_entities[i][j]) 
                    else:
                        G.add_node(negative_entities[i][j])


                    last_comment = x 
                if not G.has_node(negative_entities[i][j]):
                    G.add_node(negative_entities[i][j]) 
                indexxx += int(negative_comments.iloc[x, negative_comments.columns.get_loc("Size")])
                #print(indexxx)

            
        df_comment_index+=5 #ikinci array finishleyende +5 olmaidi
        if df_comment_index> df_size:
                break
    for result in negative_entities:
        for link in result:
            print(link)
            if not G.has_node(link):
                G.add_node(link)
                
    pos = nx.spring_layout(G)

    #nx.draw(G,pos, with_labels=True,node_color='white',  font_size=10, bbox=dict(boxstyle="round,pad=0.3", alpha=0.5))
    st.session_state.graph_neg, st.session_state.pos_neg =G,pos

    #plt.savefig("graph_neg.png", format="png", dpi=300)
    #st.session_state.graph_neg = "graph_neg.png"






    G = nx.Graph()
    df_comment_index = 0
    last_comment = 0
    df_size = len(positive_comments)
    for i in range(len(positive_pos)):
        for j in range(len(positive_pos[i])):
            if len(positive_pos[i])==0:
                continue
            indexxx = 0
            for x in range(df_comment_index , df_comment_index+5):
                if x==df_size:
                    break
                
                if indexxx < positive_pos[i][j] < indexxx+int(positive_comments.iloc[x, positive_comments.columns.get_loc("Size")]):
                    print(positive_entities[i][j]) #entitiy
                    #print(negative_comments.iloc[x, negative_comments.columns.get_loc("Comment")]) #comment
                    print(x) # comment loc
                    if last_comment ==x and last_comment!=0:
                        G.add_edge(positive_entities[i][j-1],positive_entities[i][j]) 
                    else:
                        G.add_node(positive_entities[i][j])


                    last_comment = x 
                if not G.has_node(positive_entities[i][j]):
                    G.add_node(positive_entities[i][j])    
                indexxx += int(positive_comments.iloc[x, positive_comments.columns.get_loc("Size")])
                #print(indexxx)

            
        df_comment_index+=5 #ikinci array finishleyende +5 olmaidi
        if df_comment_index> df_size:
                break
    for result in positive_entities:
        for link in result:
            print(link)
            if not G.has_node(link):
                G.add_node(link)
             
    pos = nx.spring_layout(G)
    st.session_state.graph_pos, st.session_state.pos_pos =G,pos
    #nx.draw(G,pos, with_labels=True,node_color='white',  font_size=10, bbox=dict(boxstyle="round,pad=0.3", alpha=0.5))
    #plt.savefig("graph_pos.png", format="png", dpi=300)
    #st.session_state.graph_pos = "graph_pos.png"

    G = nx.Graph()
    df_comment_index = 0
    last_comment = 0
    df_size = len(neutural_comments)
    for i in range(len(neutural_pos)):
        for j in range(len(neutural_pos[i])):
            if len(neutural_pos[i])==0:
                continue
            indexxx = 0
            for x in range(df_comment_index , df_comment_index+5):
                if x==df_size:
                    break
                
                if indexxx < neutural_pos[i][j] < indexxx+int(neutural_comments.iloc[x, neutural_comments.columns.get_loc("Size")]):
                    #print(neutural_entities[i][j]) #entitiy
                    #print(negative_comments.iloc[x, negative_comments.columns.get_loc("Comment")]) #comment
                    #print(x) # comment loc
                    if last_comment ==x and last_comment!=0:
                        G.add_edge(neutural_entities[i][j-1],neutural_entities[i][j]) 
                    else:
                        G.add_node(neutural_entities[i][j])


                    last_comment = x 
                 
                indexxx += int(neutural_comments.iloc[x, neutural_comments.columns.get_loc("Size")])
                #print(indexxx)

            
        df_comment_index+=5 #ikinci array finishleyende +5 olmaidi
        if df_comment_index> df_size:
                break
    for result in neutural_entities:
        for link in result:
            print(link)
            if not G.has_node(link):
                G.add_node(link)
              
    pos = nx.spring_layout(G)
    st.session_state.graph_neu, st.session_state.neu_pos =G,pos

    st.session_state.positive_entities = positive_entities
    st.session_state.negative_entities = negative_entities 
    st.session_state.neutural_entities  = neutural_entities

   


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







if  len(st.session_state.videos)>0:
    max_commen_result = st.number_input("Enter the maximum number of comments for each video:", min_value=50, max_value=500, step=1)

    button_pressed = st.button("Start Analysis")
    with st.spinner("Analyzing... :mag:"):
        if button_pressed:
            analyze()






    
        

if st.session_state.sentiment =='sentiment.png':
    st.subheader("📊 Distrubition of comment sentiment")
    st.image(st.session_state.sentiment , caption='Distrubition of comments')

if st.session_state.graph_neg:
    st.subheader("📌 Negative Entity Relations")
    graph_html = generate_network_graph(st.session_state.graph_neg , st.session_state.neg_pos)
    st.components.v1.html(graph_html, width=800, height=600)
    df = sna(st.session_state.graph_neg)

    with st.expander("SNA"):
        st.dataframe(df)




if st.session_state.graph_pos:
    st.subheader("📌 Positive Entity Relations")
    graph_html = generate_network_graph(st.session_state.graph_pos , st.session_state.pos_pos)
    st.components.v1.html(graph_html, width=800, height=520)
    df = sna(st.session_state.graph_pos)
    
    with st.expander("SNA"):
        st.dataframe(df)


if st.session_state.graph_neu:
    st.subheader("📌 Neutural Entity Relations")
    graph_html = generate_network_graph(st.session_state.graph_neu , st.session_state.neu_pos)
    st.components.v1.html(graph_html, width=800, height=520)
    df = sna(st.session_state.graph_neu)
    
    with st.expander("SNA"):
        st.dataframe(df)



if st.session_state.wordcloud_image and len(st.session_state.wordcloud_image)>0:
    st.subheader("Word Clouds")
    selected_image_path = st.selectbox("Select an Image", st.session_state.wordcloud_image)

    st.subheader("KMeans Clustering Analysis")

    image_container = st.empty()


    image_container.image(selected_image_path, caption='Selected Image')

def make_clickable(entity):
    return f'<a href="https://en.wikipedia.org/wiki/{entity}" target="_blank">{entity}</a>'


if len(st.session_state.positive_entities) > 0 :
    pos_list  = []
    for result in st.session_state.positive_entities:
        for link in result:
            title = link
            link = link.replace(" ", "_")

            pos_list.append(link)

    df = pd.DataFrame(pos_list, columns=['Entity'])
    df['Frequency'] = df.groupby('Entity')['Entity'].transform('count')
    df = df.drop_duplicates(subset='Entity')
 
    df = df.sort_values(by='Frequency', ascending=False)
    df = df.reset_index(drop=True)
    df['Entity'] = df['Entity'].apply(make_clickable)

    

if len(st.session_state.negative_entities) > 0:
    neg_list  = []
    for result in st.session_state.negative_entities:
        for link in result:
            title = link
            link = link.replace(" ", "_")
            neg_list.append(link)
    neg_df = pd.DataFrame(neg_list, columns=['Entity'])
    neg_df['Frequency'] = neg_df.groupby('Entity')['Entity'].transform('count')
    neg_df = neg_df.drop_duplicates(subset='Entity')

    neg_df = neg_df.sort_values(by='Frequency', ascending=False)
    neg_df = neg_df.reset_index(drop=True)
    neg_df['Entity'] = neg_df['Entity'].apply(make_clickable)


if len(st.session_state.neutural_entities) > 0 :

    

    neu_list  = []

    for result in st.session_state.neutural_entities:
        for link in result:
            title = link
            link = link.replace(" ", "_")

            neu_list.append(link)

    neu_df = pd.DataFrame(neu_list, columns=['Entity'])
    neu_df['Frequency'] = neu_df.groupby('Entity')['Entity'].transform('count')
    neu_df = neu_df.drop_duplicates(subset='Entity')
 
    neu_df = neu_df.sort_values(by='Frequency', ascending=False)
    neu_df = neu_df.reset_index(drop=True)
    neu_df['Entity'] = neu_df['Entity'].apply(make_clickable)
# Display the DataFrame with clickable Wikipedia links


if len(st.session_state.neutural_entities)>0 or len( st.session_state.negative_entities)>0  or len(st.session_state.positive_entities)>0:
        
    with st.expander("Positive entites"):
        st.write(df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("Negative entites"):
        st.write(neg_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("Neutral entites"):
        st.write(neu_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("All entites"):
        all_list = []
        all_ent = st.session_state.neutural_entities + st.session_state.negative_entities + st.session_state.positive_entities
        for result in all_ent:
            for link in result:
                title = link
                link = link.replace(" ", "_")

                all_list.append(link)
        all_df = pd.DataFrame(all_list, columns=['Entity'])
        all_df['Frequency'] = all_df.groupby('Entity')['Entity'].transform('count')
        all_df = all_df.drop_duplicates(subset='Entity')
    
        all_df = all_df.sort_values(by='Frequency', ascending=False)
        all_df = all_df.reset_index(drop=True)
        all_df['Entity'] = all_df['Entity'].apply(make_clickable)

        st.write(all_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)





    


