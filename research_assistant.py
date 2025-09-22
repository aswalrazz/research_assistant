import streamlit as st
import os
import google.generativeai as genai
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import sqlite3
from utilities import GEMINI_MODELS, init_session_state, encode_pdf
from gemini_interface import analyze_pdf_content, process_query_stream

# Load environment variables
load_dotenv()

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
OPENALEX_API_URL = "https://api.openalex.org/works?search="
CROSSREF_API_URL = "https://api.crossref.org/works?query="
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

if not API_KEY:
    st.error("❌ API key not found. Set GEMINI_API_KEY in a .env file or as an environment variable.")
    st.stop()

# Configure Gemini API (Not used here)
genai.configure(api_key=API_KEY)

# -------------------- APP CONFIGURATION --------------------
st.set_page_config(page_title="Bibliometric Analysis - Saudi Arabia & Global", layout="wide", page_icon="📊")

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
        h1 {text-align: center; font-weight: bold; color: #FF9933; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);}
        .stButton button {background-color: #138808; color: white; font-weight: bold; border-radius: 10px; transition: 0.3s;}
        .stButton button:hover {background-color: #FF9933; transform: scale(1.05);}
        .stDataFrame {border: 2px solid #000080; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
        .logo {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LANGUAGE SELECTOR --------------------
language = st.sidebar.radio("Select Language / اختر اللغة", ["English", "العربية"])

def translate(text):
    translations = {
        "English": {
            "title": "Research Assistant",
            "search": "Enter Keywords:",
            "region_filter": "Select Region:",
            "fetch_data": "Fetch Research Data",
            "results": "Research Papers",
            "visualizations": "Visualizations",
            "archive": "Archived Data",
            "upload_pdf": "Upload Research Paper (PDF)",
            "analyze_pdf": "Analyze Paper",
            "ask_question": "Ask a question about the paper",
            "link_extraction": "Link Extraction"
        },
        "العربية": {
            "title": "مساعد البحث",
            "search": "أدخل الكلمات المفتاحية:",
            "region_filter": "اختر المنطقة:",
            "fetch_data": "جلب بيانات البحث",
            "results": "أوراق البحث",
            "visualizations": "التصورات البيانية",
            "archive": "البيانات المؤرشفة",
            "upload_pdf": "تحميل ورقة بحثية (PDF)",
            "analyze_pdf": "تحليل الورقة",
            "ask_question": "اطرح سؤالًا عن الورقة",
            "link_extraction": "استخراج الروابط"
        }
    }
    return translations[language].get(text, text)

# -------------------- DATABASE SETUP FOR ARCHIVE --------------------
conn = sqlite3.connect("archive.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS research_archive (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        authors TEXT,
        institution TEXT,
        year TEXT,
        type TEXT,
        citations INTEGER,
        doi TEXT
    )
""")
conn.commit()

# -------------------- APP MAIN FUNCTION --------------------
def main():
    is_arabic = language == "العربية"
    st.title("📚 Research Assistant ✍" if not is_arabic else "📚 مساعد البحث ✍")
    init_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration" if not is_arabic else "⚙️ الإعدادات")
        sidebar_tabs = st.tabs([translate("search"), translate("upload_pdf")])

        with sidebar_tabs[0]:
            # Model Selection Logic
            selected_model_name = list(GEMINI_MODELS.keys())[0]
            model_id = GEMINI_MODELS[selected_model_name]
            st.session_state.selected_model = model_id

            if st.session_state.selected_model:
                st.session_state.model = genai.GenerativeModel(st.session_state.selected_model)

            # API Request Handling
            search_query = st.text_input(translate("search"), key="search_input")
            regions = ["All Saudi Arabia", "Riyadh", "Jeddah", "Makkah", "Madinah", "Jazan", "Eastern Province"]
            region_selected = st.selectbox(translate("region_filter"), regions)
            global_view = st.checkbox("View Global Research")
            fetch_button = st.button(translate("fetch_data"))

        with sidebar_tabs[1]:
            # File Upload Section
            st.subheader(translate("upload_pdf"))
            uploaded_file = st.file_uploader("", type=["pdf"])  # Removed label from file_uploader
            if uploaded_file is not None:
                pdf_content = encode_pdf(uploaded_file.read())
                st.session_state.pdf_content = pdf_content
                if st.button(translate("analyze_pdf")):
                    st.session_state.messages = []
                    st.session_state.start_analysis = True

    # --- DATA FETCHING AND DISPLAY ---
    if fetch_button:
        with st.spinner("Fetching data..."):
            openalex_url = f"https://api.openalex.org/works?search={search_query}" if global_view else f"https://api.openalex.org/works?search={search_query},countries.sa"
            openalex_data = fetch_data(openalex_url)
            df_open_alexs = pd.DataFrame(process_open_alexs(openalex_data))

            crossref_url = f"{CROSSREF_API_URL}{search_query}"
            crossref_data = fetch_data(crossref_url)
            df_crossrefs = pd.DataFrame(process_crossrefs(crossref_data))

            semantic_scholar_params = {'query': search_query, 'fields': 'title,authors,year,venue,citationCount,url,openAccessPdf'}
            semantic_scholar_response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=semantic_scholar_params).json()
            semantic_df = pd.DataFrame(process_semantic_scholars(semantic_scholar_response))

            # Check if 'Citations' column exists in semantic_df before converting
            if 'Citations' not in semantic_df.columns:
                semantic_df['Citations'] = 0  # Initialize the 'Citations' column with 0

            # Convert citation columns to numeric, errors='coerce' will turn non-numeric values into NaN
            df_open_alexs['Citations_OpenAlex'] = pd.to_numeric(df_open_alexs['Citations_OpenAlex'], errors='coerce').fillna(0)
            df_crossrefs['Citations'] = pd.to_numeric(df_crossrefs['Citations'], errors='coerce').fillna(0)
            semantic_df['Citations'] = pd.to_numeric(semantic_df['Citations'], errors='coerce').fillna(0)

            # Concatenate all dataframes
            combined_df = pd.concat([df_open_alexs, df_crossrefs, semantic_df], ignore_index=True)

            # Calculate an overall citation score, prioritizing OpenAlex citations if available
            combined_df['Overall_Citations'] = combined_df['Citations_OpenAlex'].fillna(combined_df['Citations'])

            # Sort by overall citations in descending order and take top 10
            top_10_combined = combined_df.sort_values('Overall_Citations', ascending=False).head(10)

        tab1, tab2, tab3, tab4 = st.tabs([translate('results'), translate('visualizations'), translate('archive'), 'Consolidated Top 10'])

        with tab1:
            research_paper_tabs = st.tabs(['OpenAlex Results', 'CrossRef Results', 'Semantic Scholar Results'])

            with research_paper_tabs[0]:
                if not df_open_alexs.empty:
                    df_open_alexs['DOI'] = df_open_alexs['DOI'].apply(lambda x: f'<a href="{x}" target="_blank">DOI</a>' if x != 'N/A' else 'N/A')
                    st.markdown(df_open_alexs.sort_values('Citations_OpenAlex', ascending=False).to_html(escape=False, render_links=True), unsafe_allow_html=True)
                else:
                    st.info("No papers found in OpenAlex.")

            with research_paper_tabs[1]:
                if not df_crossrefs.empty:
                    df_crossrefs['DOI'] = df_crossrefs['DOI'].apply(lambda x: f'<a href="{x}" target="_blank">DOI</a>' if x != 'N/A' else 'N/A')
                    st.markdown(df_crossrefs.sort_values('Citations', ascending=False).to_html(escape=False, render_links=True), unsafe_allow_html=True)
                else:
                    st.info("No papers found in CrossRef.")

            with research_paper_tabs[2]:
                if not semantic_df.empty:
                    semantic_df['url'] = semantic_df['url'].apply(lambda x: f'<a href="{x}" target="_blank">URL</a>' if x != 'N/A' else 'N/A')
                    semantic_df['download_pdf'] = semantic_df['download_pdf'].apply(lambda x: f'<a href="{x}" target="_blank">PDF</a>' if x != 'N/A' else 'N/A')
                    st.markdown(semantic_df.sort_values('Citations', ascending=False).to_html(escape=False, render_links=True), unsafe_allow_html=True)
                else:
                    st.info("No papers found in Semantic Scholar.")

        with tab2:
            st.subheader("📊 Advanced Visualizations")
            visualization_tabs = st.tabs(['OpenAlex', 'CrossRef', 'Semantic Scholar'])

            with visualization_tabs[0]:
                st.subheader("OpenAlex Visualizations")
                if not df_open_alexs.empty:
                    create_visualizations_openalex(df_open_alexs)
                else:
                    st.info("No data available for OpenAlex visualizations.")

            with visualization_tabs[1]:
                st.subheader("CrossRef Visualizations")
                if not df_crossrefs.empty:
                    create_visualizations_crossref(df_crossrefs)
                else:
                    st.info("No data available for CrossRef visualizations.")

            with visualization_tabs[2]:
                st.subheader("Semantic Scholar Visualizations")
                if not semantic_df.empty:
                    create_visualizations_semantic_scholar(semantic_df)
                else:
                    st.info("No data available for Semantic Scholar visualizations.")

        with tab3:
            st.subheader("🗄️ Archived Data")
            if st.button("Archive Current Results"):
                if not df_open_alexs.empty:
                    for _, row in df_open_alexs.iterrows():
                        c.execute("""
                            INSERT INTO research_archive (title, authors, institution, year, type, citations, doi)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (row['Title'], row['Authors'], row['Institution'], row['Year'], row['Type'], row['Citations_OpenAlex'], row['DOI']))
                    conn.commit()
                    st.success("Data archived successfully!")
                else:
                    st.info("No data available to archive.")

            c.execute("PRAGMA table_info(research_archive)")
            columns_info = c.fetchall()
            column_names = [col[1] for col in columns_info]

            c.execute("SELECT * FROM research_archive")
            archived_data = c.fetchall()
            archived_df = pd.DataFrame(archived_data, columns=column_names)
            st.dataframe(archived_df)

        with tab4:
            st.subheader("Top 10 Combined Results")
            if not top_10_combined.empty:
                st.markdown(top_10_combined.to_html(escape=False, render_links=True), unsafe_allow_html=True)
            else:
                st.info("No papers found across all sources.")

    if st.session_state.start_analysis:
        if st.session_state.model and st.session_state.pdf_content:
            with st.spinner("Analyzing research paper..." if not is_arabic else "جارٍ تحليل الورقة البحثية..."):
                message_placeholder = st.empty()
                full_response = []
                for chunk in analyze_pdf_content(st.session_state.model, st.session_state.pdf_content):
                    if hasattr(chunk, "text"):
                        full_response.append(chunk.text)
                        message_placeholder.markdown("".join(full_response))
                        notes = "".join(full_response)
            st.session_state.notes = notes
            st.session_state.messages.append({"role": "assistant", "content": "📑 Research Paper Analysis\n\n" + notes})
        st.session_state.start_analysis = False

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Footer Section
    st.markdown(
        f"""
        <div style="text-align: center; color: #C57307; margin-bottom: 10px;">
            ---
            <p> Powered by Viz_AI </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    prompt = st.chat_input(translate("ask_question"))

    if prompt:
        if not st.session_state.pdf_content or not st.session_state.notes:
            st.error("Please upload a PDF first and analyze it!" if not is_arabic else "يرجى تحميل ملف PDF أولاً وتحليله!")
            return
        if not st.session_state.model:
            st.error("Please configure your API key first!" if not is_arabic else "يرجى تكوين مفتاح API الخاص بك أولاً!")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = []

            # Debugging: Print arguments to verify
            print("Model:", st.session_state.model)
            print("Notes:", st.session_state.notes)
            print("Prompt:", prompt)

            # Ensure process_query_stream is called
            for chunk in process_query_stream(model=st.session_state.model, notes=st.session_state.notes, query=prompt, pdf_content=st.session_state.pdf_content):
                if hasattr(chunk, "text"):
                    full_response.append(chunk.text)
                    response_placeholder.markdown("".join(full_response))

            final_response = "".join(full_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

def fetch_data(url):
    try:
        response = requests.get(url).json()
        return response
    except Exception as e:
        print(f'Error fetching data from URL:{e}')
        return {}

def process_open_alexs(data):
    research_papers = []
    for entry in data.get('results', []):
        paper_doi_link = entry.get("doi", 'N/A') if entry.get("doi") else 'N/A'

        authors_list = []
        for authorships_entry in (entry.get("authorships") or []):
            if "author" in authorships_entry and "display_name" in authorships_entry["author"]:
                authors_list.append(authorships_entry["author"]["display_name"])
            else:
                authors_list.append("Unknown")
        authors = ', '.join(authors_list)

        institutions_list = []
        for authorships_entry in (entry.get("authorships") or []):
            if "institutions" in authorships_entry:
                for inst in authorships_entry["institutions"]:
                    institutions_list.append(inst.get("display_name", "Unknown"))
            else:
                institutions_list.append("Unknown")
        institution = ', '.join(institutions_list)

        paper = {
            'Title': entry.get('title', 'N/A'),
            'Authors': authors,
            'Institution': institution,
            'Year': entry.get('publication_year', "N/A"),
            'Type': entry.get("type", "journal-article"),
            'Citations_OpenAlex': entry.get("cited_by_count", None),
            'DOI': paper_doi_link
        }
        research_papers.append(paper)
    return research_papers

def process_crossrefs(data):
    research_papers = []
    if 'message' in data and 'items' in data['message']:
        for entry in data['message']['items']:
            authors = ', '.join(f"{author.get('family', '')}, {author.get('given', '')}" for author in entry.get('author', []))
            paper = {
                'Title': entry.get('title', ['N/A'])[0],
                'Authors': authors,
                'Institution': 'N/A',
                'Year': entry.get('published-print', {}).get('date-parts', [['N/A']])[0][0] if entry.get('published-print') else 'N/A',
                'Type': entry.get('type', 'N/A'),
                'Citations': entry.get('is-referenced-by-count', 0),
                'DOI': entry.get('DOI', 'N/A')
            }
            research_papers.append(paper)
    return research_papers

def process_semantic_scholars(data):
    papers = []
    for result in data.get('data', []):
        if result:
            authors = ', '.join([author['name'] for author in result.get('authors', [])])
            paper = {
                'Title': result.get('title', 'N/A'),
                'Authors': authors,
                'Institution': result.get('venue', 'N/A'),
                'Year': result.get('year', 'N/A'),
                'Citations': result.get('citationCount', 0),
                'url': result.get('url', 'N/A'),
                'download_pdf': result.get('openAccessPdf', {}).get('url', 'N/A') if result.get('openAccessPdf') else 'N/A'
            }
            papers.append(paper)
    return papers

def create_visualizations_openalex(df):
    # Group by year and count the number of papers published each year
    yearly_publications = df.groupby('Year').size().reset_index(name='Count')
    yearly_publications = yearly_publications[yearly_publications['Year'] != 'N/A']

    # Ensure 'Year' is numeric for sorting
    yearly_publications['Year'] = pd.to_numeric(yearly_publications['Year'], errors='coerce')
    yearly_publications = yearly_publications.dropna(subset=['Year'])

    yearly_publications = yearly_publications.sort_values('Year')

    # Create a bar chart of yearly publications
    fig_yearly_publications = px.bar(yearly_publications, x='Year', y='Count', title='Yearly Publications')
    st.plotly_chart(fig_yearly_publications)

    # Citations per year
    yearly_citations = df.groupby('Year')['Citations_OpenAlex'].sum().reset_index()
    yearly_citations = yearly_citations[yearly_citations['Year'] != 'N/A']

    # Ensure 'Year' is numeric
    yearly_citations['Year'] = pd.to_numeric(yearly_citations['Year'], errors='coerce')
    yearly_citations = yearly_citations.dropna(subset=['Year'])
    yearly_citations = yearly_citations.sort_values('Year')

    # Create a line chart of citations per year
    fig_yearly_citations = px.line(yearly_citations, x='Year', y='Citations_OpenAlex', title='Citations Per Year')
    st.plotly_chart(fig_yearly_citations)

def create_visualizations_crossref(df):
    # Group by year and count the number of papers published each year
    yearly_publications = df.groupby('Year').size().reset_index(name='Count')
    yearly_publications = yearly_publications[yearly_publications['Year'] != 'N/A']

    # Ensure 'Year' is numeric for sorting
    yearly_publications['Year'] = pd.to_numeric(yearly_publications['Year'], errors='coerce')
    yearly_publications = yearly_publications.dropna(subset=['Year'])
    yearly_publications = yearly_publications.sort_values('Year')

    # Create a bar chart of yearly publications
    fig_yearly_publications = px.bar(yearly_publications, x='Year', y='Count', title='Yearly Publications')
    st.plotly_chart(fig_yearly_publications)

    # Citations per year
    yearly_citations = df.groupby('Year')['Citations'].sum().reset_index()
    yearly_citations = yearly_citations[yearly_citations['Year'] != 'N/A']

    # Ensure 'Year' is numeric
    yearly_citations['Year'] = pd.to_numeric(yearly_citations['Year'], errors='coerce')
    yearly_citations = yearly_citations.dropna(subset=['Year'])
    yearly_citations = yearly_citations.sort_values('Year')

    # Create a line chart of citations per year
    fig_yearly_citations = px.line(yearly_citations, x='Year', y='Citations', title='Citations Per Year')
    st.plotly_chart(fig_yearly_citations)

def create_visualizations_semantic_scholar(df):
    # Group by year and count the number of papers published each year
    yearly_publications = df.groupby('Year').size().reset_index(name='Count')
    yearly_publications = yearly_publications[yearly_publications['Year'] != 'N/A']

    # Ensure 'Year' is numeric for sorting
    yearly_publications['Year'] = pd.to_numeric(yearly_publications['Year'], errors='coerce')
    yearly_publications = yearly_publications.dropna(subset=['Year'])
    yearly_publications = yearly_publications.sort_values('Year')

    # Create a bar chart of yearly publications
    fig_yearly_publications = px.bar(yearly_publications, x='Year', y='Count', title='Yearly Publications')
    st.plotly_chart(fig_yearly_publications)

    # Citations per year
    yearly_citations = df.groupby('Year')['Citations'].sum().reset_index()
    yearly_citations = yearly_citations[yearly_citations['Year'] != 'N/A']

    # Ensure 'Year' is numeric
    yearly_citations['Year'] = pd.to_numeric(yearly_citations['Year'], errors='coerce')
    yearly_citations = yearly_citations.dropna(subset=['Year'])
    yearly_citations = yearly_citations.sort_values('Year')

    # Create a line chart of citations per year
    fig_yearly_citations = px.line(yearly_citations, x='Year', y='Citations', title='Citations Per Year')
    st.plotly_chart(fig_yearly_citations)

if __name__ == "__main__":
    main()
