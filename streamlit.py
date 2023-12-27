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
from itertools import combinations
import nltk
nltk.download('punkt')  # Download the punkt tokenizer data
import wikipediaapi
import graphviz
from nltk.tokenize import sent_tokenize




load_dotenv()

youtube_api_key = os.getenv("YOUTUBE_API_KEY")

tagme_api_key = os.getenv("TAGME_API_KEY")

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)




if 'wordcloud_image' not in st.session_state:
        # If not, initialize it
        st.session_state.wordcloud_image = []
if 'negative_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.negative_entities = []
if 'positive_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.positive_entities = []
if 'neutral_entities' not in st.session_state:
        # If not, initialize it
        st.session_state.neutral_entities = []
if 'sentiment' not in st.session_state:
        # If not, initialize it
        st.session_state.sentiment = ''
if 'graph_pos' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_pos =  nx.Graph()
if 'graph_neg' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_neg = nx.Graph()
if 'graph_neu' not in st.session_state:
        # If not, initialize it
        st.session_state.graph_neu =  nx.Graph()
if 'pos' not in st.session_state:
        # If not, create an empty placeholder
        st.session_state.pos_pos, st.session_state.neg_pos   ,  st.session_state.neu_pos = None, None, None
if 'videos' not in st.session_state:    
    st.session_state.videos = []
if 'id' not in st.session_state:
    st.session_state.id = []
if 'comment_number' not in st.session_state:
    st.session_state.comment_number = 0 
if 'sentence_number' not in st.session_state:
    st.session_state.sentence_number = 0 
if 'bar_positive' not in st.session_state:
    st.session_state.bar_positive = 0 
if 'bar_negative' not in st.session_state:
    st.session_state.bar_negative = 0 
if 'bar_neu' not in st.session_state:
    st.session_state.bar_neu = 0 

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
        #Exclude short comments
        comments[:] = [comment_data for comment_data in comments if len(comment_data) >= 15]
        print(comments)
            
        st.session_state.comment_number +=len(comments)
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
                if "title" in annotation and annotation["link_probability"]>0.15 and annotation["rho"]>0.3:
                    entities.append(annotation["title"])
                    entity_pos.append(annotation["start"])

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






wiki_wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.WIKI, headers={'User-Agent': 'Your-User-Agent-Name/1.0 (senkaam2@gmail.com)'})

def accepted_entites(titles):
    
    accepted_list = []
    if len(titles)>0:
        for index, title in enumerate(titles):
            try:
                page = wiki_wiki.page(title)
                
                if page.exists():
                    categories = page.categories.keys() 
                    if categories:
                        found = any(("book" in element) or ("movie" in element) or ("music" in element) or ("album" in element) or ("film" in element) or ("song" in element) or ("television series" in element) for element in categories)
                        if not found:
                            accepted_list.append(title)
                        else:
                            print(title)
                            print(categories)
            except :
                accepted_list.append(title)
                print(title)
    return accepted_list





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
    

def link_entities_and_create_graph(comment, sentiment):
    sentences = sent_tokenize(comment)
    st.session_state.sentence_number +=len(sentences)
    for sentence in sentences:
            entities = tagme_entity_linking(sentence)
            final_entities = accepted_entites(entities)
            add_edges_nodes(final_entities, sentiment)
    
def add_edges_nodes(nodes , sentiment):
    
    if sentiment==1 :
        st.session_state.positive_entities+=nodes
        st.session_state.graph_pos.add_nodes_from(nodes)
        if len(nodes) > 1:
            for edge_pair in combinations(nodes, 2):
                st.session_state.graph_pos.add_edge(*edge_pair)
    elif  sentiment == -1 :
        st.session_state.negative_entities += nodes
        st.session_state.graph_neg.add_nodes_from(nodes)
        print(nodes)
        print(len(nodes))
        if len(nodes)>1:
            for edge_pair in combinations(nodes, 2):
                st.session_state.graph_neg.add_edge(*edge_pair)
    elif sentiment==0:
        st.session_state.neutral_entities += nodes
        st.session_state.graph_neu.add_nodes_from(nodes)
      
        if len(nodes)>1:
            for edge_pair in combinations(nodes, 2):
                st.session_state.graph_neu.add_edge(*edge_pair)
        


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



def generate_network_graph(G):
    # Convert the NetworkX graph to a PyDot graph
    pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
    # Specify the layout algorithm (e.g., 'neato') in PyDot
    pydot_graph.set("layout", "circo")
   

    # Convert the PyDot graph to a Graphviz graph
    graphviz_graph = graphviz.Source(pydot_graph.to_string())

    # Get the SVG content as a string
    svg_content = graphviz_graph.pipe(format='svg').decode('utf-8')

    return svg_content


     

st.title("Text Analysis App")
with st.expander("Acknowledgements"):
    
    st.write("We will collect comments that have more than 15 characters and are written in English. Additionally, during the analysis, entities such as book names, song names, television series and film names will be excluded.")


query = st.text_input("Enter YouTube search query:")
max_video_num = st.number_input("Enter the maximum number of video:", min_value=1, max_value=100, step=1)

search = st.button("Search videos!")
with st.expander("Enter custom video you want!"):
    
    user_video = st.text_input("Enter YouTube video link:")
    add_custom_video = st.button("Add this video!")

    if add_custom_video:
            index = user_video.find('v=')

            if index != -1:
                # Extract the 11 characters after 'v='
                video_id = user_video[index + 2:index + 13]
                st.session_state.id.append(video_id)








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
    all_comments_data = {'Comment': [], 'Polarity': []}
    update = st.empty()
    update.text("💭 Getting comments from videos...") 
    # Get comments for each of the first four videos
    for video_id in video_ids:
        comments = get_video_comments(video_id , max_commen_result)
        for comment in comments:
            language = detect_language(comment)
            polarity = analyze_sentiment(comment)

            # If the polarity is not zero and the language is English, add the comment and polarity to the DataFrame
            if language == 'en':
                all_comments_data['Comment'].append(comment)
                all_comments_data['Polarity'].append(polarity)
                
           
                
           
    update.text("🔗 Getting entity links!..")
    comments_df = pd.DataFrame(all_comments_data)

    # Count the number of positive and negative comments
    positive_comments = comments_df[comments_df['Polarity'] > 0.2]
    negative_comments = comments_df[comments_df['Polarity'] < -0.2]
    neutral_comments = comments_df[(comments_df['Polarity'] >= -0.2) & (comments_df['Polarity'] <= 0.2)]
    st.session_state.bar_positive = len(positive_comments)
    st.session_state.bar_negative =len(negative_comments)
    st.session_state.bar_neu = len(neutral_comments)

    for index, row in positive_comments.iterrows():
        comment_text = row['Comment']
        link_entities_and_create_graph(comment_text,1)

    for index, row in negative_comments.iterrows():
        comment_text = row['Comment']
        link_entities_and_create_graph(comment_text,-1)
        
    for index, row in neutral_comments.iterrows():
        comment_text = row['Comment']
        link_entities_and_create_graph(comment_text,0)
    update.text("💢 Clustering comments... ")
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
    
 
  
    
 
    update.empty()
   




if  len(st.session_state.id)>0:
    max_commen_result = st.number_input("Enter the maximum number of comments for each video:", min_value=50, max_value=500, step=1)

    button_pressed = st.button("Start Analysis")
    with st.spinner("Analyzing... :mag:"):
        if button_pressed:
            analyze()



if len(st.session_state.neutral_entities)>0 or len( st.session_state.negative_entities)>0  or len(st.session_state.positive_entities)>0:
    st.success("Analysis completed! :white_check_mark:")
    st.write(f" {st.session_state.comment_number} number of comments found eligible for this analysis, and {st.session_state.sentence_number} number of sentences analyzed.")

    
    st.title('Sentiment Analysis')

    # Define the chart data
    chart_data = {'Positive': st.session_state.bar_positive, 'Negative': st.session_state.bar_negative}

    # Create a Streamlit sidebar to adjust the bar width
    # Omitted for brevity

    # Create a bar chart using matplotlib
    fig, ax = plt.subplots()

    # Set colors based on positive/negative values
    colors = ['green', 'red']  # Positive is green, Negative is red

    # Specify the width of the bars
    bar_width = 0.16

    # Create bars with specified colors and width
    bars = ax.bar(chart_data.keys(), chart_data.values(), color=colors, width=bar_width)

    # Adjust the size of the figure
    fig.set_size_inches(4, 3)  # Set the width and height of the figure in inches

    # Display the bar chart using Streamlit
    st.pyplot(fig)



    
        




if st.session_state.graph_pos:
    st.subheader("📌 Entities and Relations in  Positive Sentences")
    svg_content = generate_network_graph(st.session_state.graph_pos)
    st.image(svg_content)
    posi_df = sna(st.session_state.graph_pos) 
    with st.expander("SNA"):
        st.dataframe(posi_df)



if st.session_state.graph_neg:
    st.subheader("📌 Entity Relations in Negative Sentences")
    svg_content = generate_network_graph(st.session_state.graph_neg)
    st.image(svg_content)
    nega_df = sna(st.session_state.graph_neg)
    with st.expander("SNA"):
        st.dataframe(nega_df)


if st.session_state.graph_neu:
    st.subheader("📌 Entities and Relations in Neutural Sentences")
    svg_content = generate_network_graph(st.session_state.graph_neu )
    st.image(svg_content) 
    neut_df = sna(st.session_state.graph_neu)
    with st.expander("SNA"):
        st.dataframe(neut_df)
    st.subheader("📌 Entities and Relations in All Sentences")
    merged_graph = nx.compose(st.session_state.graph_pos , st.session_state.graph_neg)
    merged_graph = nx.compose(merged_graph,st.session_state.graph_neu )

    all_svg_content = generate_network_graph(merged_graph)
    st.image(all_svg_content)
    all_df_c = sna(merged_graph)
    with st.expander("SNA"):
        st.dataframe(all_df_c)
    




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
    print(st.session_state.positive_entities)
    for link in st.session_state.positive_entities:
            if len(link)==0:
                continue
        
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
    for link in st.session_state.negative_entities:
        if len(link)>0:
            title = link
            link = link.replace(" ", "_")
            neg_list.append(link)
    neg_df = pd.DataFrame(neg_list, columns=['Entity'])
    neg_df['Frequency'] = neg_df.groupby('Entity')['Entity'].transform('count')
    neg_df = neg_df.drop_duplicates(subset='Entity')

    neg_df = neg_df.sort_values(by='Frequency', ascending=False)
    neg_df = neg_df.reset_index(drop=True)
    neg_df['Entity'] = neg_df['Entity'].apply(make_clickable)


if len(st.session_state.neutral_entities) > 0 :

    

    neu_list  = []
    print(st.session_state.neutral_entities)
    for link in st.session_state.neutral_entities:
        
        if len(link)>0:
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


if len(st.session_state.neutral_entities)>0 or len( st.session_state.negative_entities)>0  or len(st.session_state.positive_entities)>0:
        
    with st.expander("Entities  in  Positive Sentences"):
        if len( st.session_state.positive_entities)>0:
            st.write(df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("Entities in Negative Sentences"):
        if len( st.session_state.negative_entities)>0:
            st.write(neg_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("Entities in Neutural Sentences"):
        if len( st.session_state.neutral_entities)>0:
            st.write(neu_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)
    with st.expander("All entites"):
        all_list = []
        all_list = st.session_state.neutral_entities + st.session_state.negative_entities + st.session_state.positive_entities

        all_df = pd.DataFrame(all_list, columns=['Entity'])
        all_df['Frequency'] = all_df.groupby('Entity')['Entity'].transform('count')
        all_df = all_df.drop_duplicates(subset='Entity')
    
        all_df = all_df.sort_values(by='Frequency', ascending=False)
        all_df = all_df.reset_index(drop=True)
        all_df['Entity'] = all_df['Entity'].apply(make_clickable)

        st.write(all_df.to_html(escape=False, render_links=True), unsafe_allow_html=True)






