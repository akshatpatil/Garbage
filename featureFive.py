import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import requests
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="OpenAlex Keyword Explorer",
    page_icon="📚",
    layout="wide"
)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

# Define helper functions for API calls
def search_openalex(query, page=1, per_page=10, filter_string=""):
    """Search OpenAlex API for works matching the query"""
    base_url = "https://api.openalex.org/works"
    
    # Email for polite pool - replace with your email
    email = "your.email@example.com"
    
    # Build query parameters
    params = {
        "search": query,
        "page": page,
        "per_page": per_page,
        "filter": filter_string,
        "mailto": email
    }
    
    # Make request
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None

def get_work_details(work_id):
    """Get detailed information about a specific work"""
    # Email for polite pool - replace with your email
    email = "your.email@example.com"
    
    url = f"https://api.openalex.org/{work_id}?mailto={email}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None

def format_openalex_works(works_data):
    """Format OpenAlex API response into a pandas DataFrame"""
    if not works_data or 'results' not in works_data:
        return pd.DataFrame()
    
    formatted_works = []
    
    for work in works_data['results']:
        # Extract basic information
        work_info = {
            "id": work.get("id", ""),
            "title": work.get("title", "No title"),
            "abstract": work.get("abstract", "No abstract available"),
            "publication_year": work.get("publication_year", None),
            "citation_count": work.get("cited_by_count", 0),
            "type": work.get("type", "unknown")
        }
        
        # Extract authors (first 3)
        authors = work.get("authorships", [])
        author_names = []
        for author in authors[:3]:
            if author.get("author", {}).get("display_name"):
                author_names.append(author["author"]["display_name"])
        work_info["authors"] = ", ".join(author_names) + ("..." if len(authors) > 3 else "")
        
        # Extract journal/venue name
        if work.get("primary_location") and work["primary_location"].get("source"):
            work_info["venue"] = work["primary_location"]["source"].get("display_name", "Unknown venue")
        else:
            work_info["venue"] = "Unknown venue"
        
        # Extract concepts/keywords (top 5 by score)
        concepts = work.get("concepts", [])
        concepts.sort(key=lambda x: x.get("score", 0), reverse=True)
        keywords = [concept.get("display_name", "") for concept in concepts[:5] if concept.get("score", 0) > 0.3]
        work_info["keywords"] = keywords
        
        # Extract DOI
        work_info["doi"] = work.get("doi", "")
        
        formatted_works.append(work_info)
    
    return pd.DataFrame(formatted_works)

def preprocess_text(text):
    """Clean and tokenize text"""
    if not text or text == "No abstract available":
        return []
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

def extract_keywords(text, n=20):
    """Extract top n keywords from text using TF-IDF"""
    if not text or text == "No abstract available":
        return {}
    
    # Create a single-document corpus for TF-IDF
    corpus = [text]
    
    # Instantiate TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)  # Include both unigrams and bigrams
    )
    
    # Fit and transform the corpus
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get feature names and TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Sort keywords by TF-IDF score
        sorted_idx = np.argsort(tfidf_scores)[::-1]
        
        # Create a dictionary of keyword:score
        keywords = {feature_names[idx]: float(tfidf_scores[idx]) for idx in sorted_idx[:n]}
        
        return keywords
    except:
        st.warning("Could not extract keywords from this text.")
        return {}

def generate_wordcloud(keywords, width=800, height=400):
    """Generate a word cloud image from keywords"""
    if not keywords:
        # Create a placeholder image with text
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No keywords available", 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=18)
        ax.axis('off')
        
        # Convert plot to image
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return buf
    
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='viridis',
        max_words=50,
        max_font_size=100
    )
    
    # Generate word cloud
    wc.generate_from_frequencies(keywords)
    
    # Create a figure and plot the word cloud
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    # Convert plot to image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def rank_works_by_keyword(works_df, selected_keyword):
    """Rank works based on relevance to a selected keyword"""
    if works_df.empty:
        return works_df
    
    # Combine title and abstract for similarity calculation
    works_df['combined_text'] = works_df['title'] + " " + works_df['abstract'].fillna("")
    
    # Calculate similarity between each work and the selected keyword
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(list(works_df['combined_text']) + [selected_keyword])
        
        # Get the last row (the keyword) and calculate similarity with all works
        keyword_vector = tfidf_matrix[-1]
        work_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(work_vectors, keyword_vector)
        
        # Add similarity scores to the dataframe
        works_df['relevance_score'] = similarities
        
        # Sort by relevance score
        ranked_works = works_df.sort_values(by='relevance_score', ascending=False)
        
        return ranked_works
    except:
        works_df['relevance_score'] = 0.0
        return works_df

# Main app
def main():
    st.title("📚 OpenAlex Keyword & Concept Explorer")
    st.markdown("Explore academic papers and extract key concepts using the [OpenAlex API](https://docs.openalex.org)")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Search OpenAlex", "Extract Keywords", "Paper Explorer"])
    
    # Initialize session state variables if they don't exist
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = None
    if 'selected_paper' not in st.session_state:
        st.session_state['selected_paper'] = None
    if 'keywords' not in st.session_state:
        st.session_state['keywords'] = None
    if 'selected_keyword' not in st.session_state:
        st.session_state['selected_keyword'] = None
    
    with tab1:
        st.header("Search Academic Papers")
        
        # Search form
        with st.form(key='search_form'):
            search_query = st.text_input("Search Query:", value="machine learning natural language processing")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year_filter = st.slider("Publication Year:", 2000, 2025, (2018, 2025))
            
            with col2:
                is_open_access = st.selectbox("Open Access:", ["Any", "Only Open Access", "No Open Access"])
            
            with col3:
                sort_by = st.selectbox("Sort By:", ["Relevance", "Publication Date", "Citation Count"])
            
            submitted = st.form_submit_button("Search OpenAlex")
        
        if submitted:
            # Construct filter string
            filters = []
            
            # Add year filter
            filters.append(f"publication_year:{year_filter[0]}-{year_filter[1]}")
            
            # Add open access filter
            if is_open_access == "Only Open Access":
                filters.append("is_oa:true")
            elif is_open_access == "No Open Access":
                filters.append("is_oa:false")
            
            # Combine filters
            filter_string = ",".join(filters)
            
            # Show loading spinner
            with st.spinner("Searching OpenAlex..."):
                # Call OpenAlex API
                results = search_openalex(search_query, filter_string=filter_string)
                
                if results and 'results' in results:
                    # Format results into DataFrame
                    works_df = format_openalex_works(results)
                    
                    if not works_df.empty:
                        # Store in session state
                        st.session_state['search_results'] = works_df
                        
                        # Display result count
                        meta = results.get('meta', {})
                        total_count = meta.get('count', 0)
                        st.success(f"Found {total_count} papers. Displaying first {len(works_df)} results.")
                        
                        # Allow sorting
                        if sort_by == "Publication Date":
                            works_df = works_df.sort_values(by='publication_year', ascending=False)
                        elif sort_by == "Citation Count":
                            works_df = works_df.sort_values(by='citation_count', ascending=False)
                        
                        # Display results with clickable titles
                        for i, (_, work) in enumerate(works_df.iterrows()):
                            paper_title = work['title']
                            authors = work['authors']
                            venue = work['venue']
                            year = work['publication_year']
                            citations = work['citation_count']
                            
                            # Create a compact paper display
                            with st.container():
                                st.markdown(f"### {i+1}. {paper_title}")
                                cols = st.columns([3, 1])
                                with cols[0]:
                                    st.write(f"**Authors:** {authors}")
                                    st.write(f"**Published in:** {venue} ({year})")
                                with cols[1]:
                                    st.write(f"**Citations:** {citations}")
                                
                                # Buttons for paper selection
                                if st.button(f"View Paper Details #{i}"):
                                    st.session_state['selected_paper'] = works_df.iloc[i]
                                    # Switch to next tab
                                    st.experimental_set_query_params(tab="extract_keywords")
                                
        
                                st.divider()
                    else:
                        st.warning("No results found. Try a different search query.")
                else:
                    st.error("Error retrieving results from OpenAlex API.")
        
        # Display existing search results if available
        elif st.session_state['search_results'] is not None:
            works_df = st.session_state['search_results']
            
            st.success(f"Displaying {len(works_df)} results from previous search.")
            
            for i, (_, work) in enumerate(works_df.iterrows()):
                paper_title = work['title']
                authors = work['authors']
                venue = work['venue']
                year = work['publication_year']
                citations = work['citation_count']
                
                with st.container():
                    st.markdown(f"### {i+1}. {paper_title}")
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(f"**Authors:** {authors}")
                        st.write(f"**Published in:** {venue} ({year})")
                    with cols[1]:
                        st.write(f"**Citations:** {citations}")
                    
                    if st.button(f"View Paper Details #{i}"):
                        st.session_state['selected_paper'] = works_df.iloc[i]
                        # Switch to next tab
                        st.experimental_set_query_params(tab="extract_keywords")
                    
                    st.divider()
    
    with tab2:
        st.header("Extract Keywords from Paper")
        
        if st.session_state['selected_paper'] is not None:
            paper = st.session_state['selected_paper']
            
            # Display paper details
            st.subheader(paper['title'])
            st.write(f"**Authors:** {paper['authors']}")
            st.write(f"**Published in:** {paper['venue']} ({paper['publication_year']})")
            
            # Display DOI link if available
            if paper['doi']:
                st.write(f"**DOI:** [Link to paper]({paper['doi']})")
            
            # Show abstract
            st.subheader("Abstract")
            st.write(paper['abstract'])
            
            # Show OpenAlex concepts
            st.subheader("OpenAlex Concepts")
            if paper['keywords']:
                for keyword in paper['keywords']:
                    st.markdown(f"- {keyword}")
            else:
                st.write("No concepts available from OpenAlex.")
            
            # Extract keywords
            col1, col2 = st.columns([1, 3])
            
            with col1:
                num_keywords = st.slider("Number of keywords to extract", 5, 50, 20)
                extract_button = st.button("Extract Custom Keywords")
            
            if extract_button or 'keywords' not in st.session_state or st.session_state['keywords'] is None:
                # Extract keywords
                with st.spinner("Extracting keywords..."):
                    keywords = extract_keywords(paper['abstract'], n=num_keywords)
                    
                    # Store in session state
                    st.session_state['keywords'] = keywords
                    
                    with col2:
                        st.subheader("Extracted Keywords")
                        
                        if keywords:
                            # Display keyword table
                            keyword_df = pd.DataFrame({
                                'Keyword': list(keywords.keys()),
                                'Score': list(keywords.values())
                            })
                            st.dataframe(keyword_df, hide_index=True)
                        else:
                            st.warning("Could not extract keywords from this abstract.")
            else:
                with col2:
                    st.subheader("Extracted Keywords")
                    
                    if st.session_state['keywords']:
                        # Display keyword table
                        keyword_df = pd.DataFrame({
                            'Keyword': list(st.session_state['keywords'].keys()),
                            'Score': list(st.session_state['keywords'].values())
                        })
                        st.dataframe(keyword_df, hide_index=True)
                    else:
                        st.warning("No keywords available.")
            
            # Generate and display word cloud
            if st.session_state['keywords']:
                st.subheader("Keyword Cloud")
                wordcloud_img = generate_wordcloud(st.session_state['keywords'])
                st.image(wordcloud_img, width=800)
                
                # Help text
                st.info("👆 Click on the 'Paper Explorer' tab to find papers related to these keywords!")
        else:
            st.info("Select a paper from the 'Search OpenAlex' tab to extract keywords.")
    
    with tab3:
        st.header("Paper Explorer")
        
        # Check if keywords are available
        if 'keywords' not in st.session_state or st.session_state['keywords'] is None:
            st.info("Extract keywords first in the 'Extract Keywords' tab.")
        else:
            # Display word cloud
            st.subheader("Select a keyword to find related papers")
            
            # Re-generate word cloud
            wordcloud_img = generate_wordcloud(st.session_state['keywords'])
            st.image(wordcloud_img, width=800)
            
            # Create buttons for each keyword
            st.subheader("Select a keyword:")
            
            # Create multiple columns for keyword buttons
            keyword_cols = st.columns(4)
            for i, keyword in enumerate(st.session_state['keywords'].keys()):
                col_idx = i % 4
                with keyword_cols[col_idx]:
                    if st.button(keyword):
                        st.session_state['selected_keyword'] = keyword
            
            # Display selected keyword and ranked papers
            if 'selected_keyword' in st.session_state and st.session_state['selected_keyword']:
                selected_keyword = st.session_state['selected_keyword']
                st.subheader(f"Searching OpenAlex for papers related to '{selected_keyword}'")
                
                # Search OpenAlex for the selected keyword
                with st.spinner(f"Searching for papers related to '{selected_keyword}'..."):
                    results = search_openalex(selected_keyword, per_page=25)
                    
                    if results and 'results' in results:
                        # Format results into DataFrame
                        related_works = format_openalex_works(results)
                        
                        if not related_works.empty:
                            # Rank by relevance to the keyword
                            ranked_works = rank_works_by_keyword(related_works, selected_keyword)
                            
                            # Display ranked papers
                            meta = results.get('meta', {})
                            total_count = meta.get('count', 0)
                            st.success(f"Found {total_count} papers related to '{selected_keyword}'. Displaying top {len(ranked_works)} results.")
                            
                            for i, (_, work) in enumerate(ranked_works.iterrows()):
                                relevance = work.get('relevance_score', [0])[0] if isinstance(work.get('relevance_score'), np.ndarray) else 0
                                with st.expander(f"{i+1}. {work['title']} (Relevance: {relevance:.2f})", expanded=i==0):
                                    st.write(f"**Authors:** {work['authors']}")
                                    st.write(f"**Published in:** {work['venue']} ({work['publication_year']})")
                                    st.write(f"**Citations:** {work['citation_count']}")
                                    if work['doi']:
                                        st.write(f"**DOI:** [Link to paper]({work['doi']})")
                                    st.write(f"**Abstract:** {work['abstract']}")
                                    
                                    # Display keywords
                                    if work['keywords']:
                                        st.write("**Keywords:** " + ", ".join(work['keywords']))
                        else:
                            st.warning(f"No papers found related to '{selected_keyword}'.")
                    else:
                        st.error("Error retrieving results from OpenAlex API.")

if __name__ == "__main__":
    main()