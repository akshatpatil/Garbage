import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import time

# Set page config
st.set_page_config(
    page_title="Intelligent Field Detection",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 Intelligent Field Detection")
st.markdown("""
This app automatically detects the research fields from a paper's title and abstract.
Enter text or search for papers on OpenAlex to see their field classification.
""")

# OpenAlex API functions
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

def get_concept_details(concept_id):
    """Get detailed information about a specific concept"""
    # Email for polite pool - replace with your email
    email = "your.email@example.com"
    
    url = f"https://api.openalex.org/{concept_id}?mailto={email}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error when fetching concept details: {e}")
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
        
        # Extract concepts/keywords (all concepts)
        concepts = work.get("concepts", [])
        concepts.sort(key=lambda x: x.get("score", 0), reverse=True)
        work_info["concepts"] = concepts
        
        # Extract DOI
        work_info["doi"] = work.get("doi", "")
        
        formatted_works.append(work_info)
    
    return pd.DataFrame(formatted_works)

def extract_concepts_from_text(title, abstract, top_n=5):
    """
    Use OpenAlex API to find papers similar to the input text,
    then extract concepts from those papers.
    """
    combined_text = f"{title} {abstract}" if abstract else title
    
    # Search for similar papers
    search_results = search_openalex(combined_text, per_page=5)
    
    if not search_results or 'results' not in search_results:
        return []
    
    # Collect all concepts from the search results
    all_concepts = []
    for paper in search_results['results']:
        if 'concepts' in paper:
            all_concepts.extend(paper['concepts'])
    
    # Deduplicate concepts based on ID
    unique_concepts = {}
    for concept in all_concepts:
        concept_id = concept.get('id')
        if concept_id not in unique_concepts:
            unique_concepts[concept_id] = concept
        else:
            # If we've seen this concept before, keep the higher score
            if concept.get('score', 0) > unique_concepts[concept_id].get('score', 0):
                unique_concepts[concept_id] = concept
    
    # Sort by score and get top N
    sorted_concepts = sorted(unique_concepts.values(), key=lambda x: x.get('score', 0), reverse=True)
    
    return sorted_concepts[:top_n]

# Create tabs for the app
tab1, tab2 = st.tabs(["Text Input", "Search Papers"])

# Tab 1: Text Input
with tab1:
    st.header("Detect Fields from Text")
    
    title = st.text_input("Paper Title:", key="title_input", 
                       placeholder="Enter the title of your paper here...")
    
    abstract = st.text_area("Abstract:", height=200, key="abstract_input",
                        placeholder="Enter the abstract of your paper here...")
    
    analyze_button = st.button("Detect Fields")
    
    if analyze_button and (title or abstract):
        with st.spinner("Analyzing text and detecting fields..."):
            # Extract concepts from text
            extracted_concepts = extract_concepts_from_text(title, abstract)
            
            if extracted_concepts:
                # Store concepts in session state
                st.session_state['extracted_concepts'] = extracted_concepts
                
                # Display detected concepts
                st.subheader("Detected Fields")
                
                # Create a table of concepts
                concept_data = []
                for i, concept in enumerate(extracted_concepts):
                    concept_data.append({
                        'Rank': i+1,
                        'Field': concept.get('display_name', 'Unknown'),
                        'Score': concept.get('score', 0),
                        'Level': concept.get('level', 0),
                        'ID': concept.get('id', '')
                    })
                
                concept_df = pd.DataFrame(concept_data)
                st.dataframe(concept_df[['Rank', 'Field', 'Score', 'Level']], hide_index=True)
            else:
                st.warning("Could not extract any fields from the provided text. Please try with different text or use the search function.")
    elif analyze_button:
        st.warning("Please enter a title or abstract to analyze.")

# Tab 2: Search Papers
with tab2:
    st.header("Search and Detect Fields from Papers")
    
    # Search form
    with st.form(key='search_form'):
        search_query = st.text_input("Search Query:", value="machine learning natural language processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_filter = st.slider("Publication Year:", 2000, 2025, (2020, 2025))
        
        with col2:
            per_page = st.slider("Results Per Page:", 5, 25, 10)
        
        submitted = st.form_submit_button("Search Papers")
    
    if submitted:
        # Construct filter string
        filter_string = f"publication_year:{year_filter[0]}-{year_filter[1]}"
        
        # Show loading spinner
        with st.spinner("Searching OpenAlex..."):
            # Call OpenAlex API
            results = search_openalex(search_query, filter_string=filter_string, per_page=per_page)
            
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
                    
                    # Display results with buttons to select
                    for i, (_, work) in enumerate(works_df.iterrows()):
                        paper_title = work['title']
                        authors = work['authors']
                        venue = work['venue']
                        year = work['publication_year']
                        
                        with st.container():
                            st.markdown(f"### {i+1}. {paper_title}")
                            st.write(f"**Authors:** {authors}")
                            st.write(f"**Published in:** {venue} ({year})")
                            
                            # Button to analyze this paper
                            if st.button(f"Detect Fields for Paper #{i+1}"):
                                # Extract concepts directly from the paper
                                concepts = work['concepts']
                                
                                if concepts:
                                    # Store concepts in session state
                                    st.session_state['paper_concepts'] = concepts
                                    st.session_state['selected_paper'] = work
                                    
                                    # Display paper details
                                    st.subheader("Paper Details")
                                    st.write(f"**Title:** {work['title']}")
                                    st.write(f"**Abstract:** {work['abstract']}")
                                    
                                    # Display detected concepts
                                    st.subheader("Detected Fields")
                                    
                                    # Create a table of concepts
                                    concept_data = []
                                    for j, concept in enumerate(concepts):
                                        concept_data.append({
                                            'Rank': j+1,
                                            'Field': concept.get('display_name', 'Unknown'),
                                            'Score': concept.get('score', 0),
                                            'Level': concept.get('level', 0),
                                            'ID': concept.get('id', '')
                                        })
                                    
                                    concept_df = pd.DataFrame(concept_data)
                                    st.dataframe(concept_df[['Rank', 'Field', 'Score', 'Level']], hide_index=True)
                                else:
                                    st.warning("No field information available for this paper.")
                            
                            st.divider()
                else:
                    st.warning("No results found. Try a different search query.")
            else:
                st.error("Error retrieving results from OpenAlex API.")
    
    # Display previously selected paper if available
    elif 'selected_paper' in st.session_state and 'paper_concepts' in st.session_state:
        work = st.session_state['selected_paper']
        concepts = st.session_state['paper_concepts']
        
        st.subheader("Previously Selected Paper")
        st.write(f"**Title:** {work['title']}")
        st.write(f"**Abstract:** {work['abstract']}")
        
        # Display detected concepts
        st.subheader("Detected Fields")
        
        # Create a table of concepts
        concept_data = []
        for j, concept in enumerate(concepts):
            concept_data.append({
                'Rank': j+1,
                'Field': concept.get('display_name', 'Unknown'),
                'Score': concept.get('score', 0),
                'Level': concept.get('level', 0),
                'ID': concept.get('id', '')
            })
        
        concept_df = pd.DataFrame(concept_data)
        st.dataframe(concept_df[['Rank', 'Field', 'Score', 'Level']], hide_index=True)

# Add some information about how the app works
with st.expander("How It Works"):
    st.markdown("""
    ### How the Field Detection Works

    This application uses OpenAlex's powerful API to detect research fields from text:

    1. **Text Analysis**: When you enter a paper title and abstract, the app searches for similar papers on OpenAlex.
    
    2. **Concept Extraction**: From these similar papers, we extract the most relevant research concepts.
    
    The result is a comprehensive understanding of where your research fits within the broader academic landscape.
    """)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAlex API | Field detection leverages OpenAlex's concept hierarchies")