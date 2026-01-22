import streamlit as st
import os
import pickle
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral
from typing import List
import re
from dotenv import load_dotenv
load_dotenv()

class RAGDataCleaningAgent:
    """
    A RAG (Retrieval-Augmented Generation) agent that answers questions about 
    best practices for data cleaning using information from the PDF.
    """
    
    def __init__(self, agent_id: str, pdf_path: str, api_key=None):
        """
        Initialize the RAG agent with the Mistral API key and PDF document.
        
        Args:
            agent_id (str): The ID of your agent in Mistral Studio
            pdf_path (str): Path to the PDF document with data cleaning best practices
            api_key (str): Mistral API key. If None, will try to get from environment variable.
        """
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
        
        if not api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=api_key)
        self.agent_id = agent_id
        self.pdf_path = pdf_path
        
        # Create cache directory
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate cache file name based on PDF content hash
        self.cache_file = os.path.join(self.cache_dir, f"rag_cache_{self._get_pdf_hash()}.pkl")
        
        # Load or create the vector database
        self.chunks, self.vectorizer, self.chunk_vectors = self._load_or_create_vector_db()
    
    def _get_pdf_hash(self):
        """
        Generate a hash of the PDF content to use for caching.
        Since we're simulating the PDF content, we'll use a constant hash.
        """
        # In a real implementation, you would read the actual PDF file:
        # with open(self.pdf_path, 'rb') as f:
        #     content = f.read()
        # return hashlib.md5(content).hexdigest()
        
        # For simulation purposes, we'll use a constant
        return "simulated_pdf_content_hash"
    
    def _load_or_create_vector_db(self):
        """
        Load the vector database from cache or create it if it doesn't exist.
        """
        if os.path.exists(self.cache_file):
            # Load from cache
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            st.info("Loaded RAG database from cache")
            return data['chunks'], data['vectorizer'], data['chunk_vectors']
        else:
            # Create new vector database
            chunks = self._extract_and_chunk_pdf_content()
            vectorizer = TfidfVectorizer()
            chunk_vectors = vectorizer.fit_transform(chunks)
            
            # Save to cache
            cache_data = {
                'chunks': chunks,
                'vectorizer': vectorizer,
                'chunk_vectors': chunk_vectors
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            st.info("Created and cached RAG database")
            
            return chunks, vectorizer, chunk_vectors
    
    def _extract_and_chunk_pdf_content(self) -> List[str]:
        """
        Extract text content from the PDF file and split into chunks.
        
        Returns:
            List[str]: List of text chunks from the PDF
        """
        try:
            # Simulated content from the PDF
            simulated_content = """
            Data Cleaning Best Practices

            1. Handling Missing Values:
               - Identify missing values using statistical methods
               - Decide whether to remove, impute, or flag missing values
               - Common imputation methods: mean, median, mode, or predictive models

            2. Removing Duplicates:
               - Identify duplicate records based on key fields
               - Remove duplicates while preserving data integrity
               - Document the removal process for audit purposes

            3. Outlier Detection:
               - Use statistical methods like Z-score or IQR to identify outliers
               - Visualize data using box plots or scatter plots
               - Determine if outliers are errors or valid extreme values

            4. Data Type Validation:
               - Ensure each column has the correct data type
               - Convert data types as needed for analysis
               - Validate dates, numbers, and categorical values

            5. Standardization and Normalization:
               - Normalize numerical data to a common scale
               - Standardize features to have zero mean and unit variance
               - Apply appropriate scaling based on the algorithm to be used

            6. Handling Inconsistent Formats:
               - Standardize date formats across the dataset
               - Normalize text data (uppercase/lowercase, remove extra spaces)
               - Ensure consistent categorical values

            7. Data Validation:
               - Check for business rule compliance
               - Validate relationships between related fields
               - Perform range checks for numerical values

            8. Documentation:
               - Document all cleaning steps taken
               - Track original vs. cleaned data statistics
               - Maintain reproducible cleaning processes
            """
            
            # Simple chunking by paragraphs
            paragraphs = [p.strip() for p in simulated_content.split('\n\n') if p.strip()]
            
            # Further chunk large paragraphs if needed
            chunks = []
            for para in paragraphs:
                if len(para) > 500:  # If paragraph is too long, split by sentences
                    sentences = re.split(r'[.!?]+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(current_chunk + sentence) < 500:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                else:
                    chunks.append(para)
            
            # Filter out very small chunks
            chunks = [chunk for chunk in chunks if len(chunk) > 20]
            
            return chunks
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ["Data cleaning best practices information."]
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant chunks based on the query using TF-IDF similarity.
        
        Args:
            query (str): The user's question
            top_k (int): Number of top chunks to retrieve
            
        Returns:
            List[str]: Top-k most relevant chunks
        """
        # Transform the query using the fitted vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get indices of top-k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return the corresponding chunks
        return [self.chunks[i] for i in top_indices if similarities[i] > 0.1]  # Threshold to avoid irrelevant chunks
    
    def ask_question(self, question: str):
        """
        Ask a question to the RAG data cleaning agent.
        
        Args:
            question (str): The question about data cleaning best practices
            
        Returns:
            str: The agent's response
        """
        # Retrieve relevant chunks from the PDF
        relevant_chunks = self._retrieve_relevant_chunks(question)
        
        if not relevant_chunks:
            context = "The provided document does not contain information about this topic."
        else:
            context = "Based on the data cleaning best practices document:\n\n" + "\n\n".join(relevant_chunks)
        
        # For the conversations API, we need to format the input differently
        # Combine context and question into a single input
        full_query = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer based only on the provided context."

        inputs = [
            {"role": "user", "content": full_query}
        ]
        
        try:
            response = self.client.beta.conversations.start(
                agent_id=self.agent_id,
                inputs=inputs,
            )
            return response
        except Exception as e:
            print(f"Error communicating with agent: {e}")
            return None


class ExcelToDashboardAgent:
    """
    An agent that takes Excel files as input and generates dashboards based on the data.
    """
    
    def __init__(self, agent_id: str, api_key=None):
        """
        Initialize the Excel-to-Dashboard agent.
        
        Args:
            agent_id (str): The ID of your Excel-to-Dashboard agent in Mistral Studio
            api_key (str): Mistral API key. If None, will try to get from environment variable.
        """
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
        
        if not api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=api_key)
        self.agent_id = agent_id
    
    def analyze_excel(self, file_path: str):
        """
        Analyze an Excel file and generate insights.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            dict: Analysis results and recommendations
        """
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Basic information about the dataset
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'summary_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Prepare input for the agent - ask for specific visualization recommendations
        analysis_request = f"""
        Analyze this Excel file with the following characteristics:
        - Shape: {info['shape']}
        - Columns: {info['columns']}
        - Numerical columns: {info['numerical_columns']}
        - Categorical columns: {info['categorical_columns']}
        - Date columns: {info['date_columns']}
        
        Based on this data structure, provide specific recommendations for:
        1. Which columns would be most interesting to visualize
        2. What types of relationships exist between columns
        3. What insights might be revealed by visualizing this data
        4. Which visualization types would be most appropriate for this dataset
        
        Focus on practical insights that would help a business user understand the data.
        """
        
        inputs = [
            {"role": "user", "content": analysis_request}
        ]
        
        try:
            response = self.client.beta.conversations.start(
                agent_id=self.agent_id,
                inputs=inputs,
            )
            return response, df, info
        except Exception as e:
            print(f"Error analyzing Excel file: {e}")
            return None, df, info
    
    def generate_rich_dashboard(self, df: pd.DataFrame):
        """
        Generate a rich dashboard with multiple visualization types based on the data.
        
        Args:
            df (pd.DataFrame): The data to visualize
            
        Returns:
            plotly.Figure: Generated rich dashboard
        """
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Create a dashboard with multiple subplots
        n_charts = min(6, 2 + len(numerical_cols) + len(categorical_cols) + len(date_cols))
        if n_charts < 2:
            n_charts = 2  # Ensure at least 2 charts
        
        # Determine subplot layout
        cols = 3
        rows = (n_charts + 2) // 3  # At least 2 rows to accommodate multiple charts
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[
                "Data Overview", 
                "Distribution", 
                "Relationships" if len(numerical_cols) >= 2 else "Top Categories",
                "Trends" if date_cols else "Correlation" if len(numerical_cols) >= 2 else "Value Counts",
                "Patterns", 
                "Summary"
            ][:n_charts],
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        chart_idx = 0
        row = 1
        col = 1
        
        # Chart 1: Data Overview (always present)
        if chart_idx < n_charts:
            # Create a simple overview showing data shape and types
            overview_text = f"Dataset Overview:<br>Rows: {df.shape[0]}<br>Columns: {df.shape[1]}<br>Numeric: {len(numerical_cols)}<br>Categorical: {len(categorical_cols)}"
            fig.add_annotation(
                text=overview_text,
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=row, col=col
            )
            col += 1
            if col > cols:
                col = 1
                row += 1
            chart_idx += 1
        
        # Chart 2: Distribution of first numerical column (if available)
        if numerical_cols and chart_idx < n_charts:
            fig.add_trace(
                go.Histogram(x=df[numerical_cols[0]], name=f"Distribution of {numerical_cols[0]}", showlegend=False, marker_color='lightblue'),
                row=row, col=col
            )
            fig.update_xaxes(title_text=numerical_cols[0], row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
            col += 1
            if col > cols:
                col = 1
                row += 1
            chart_idx += 1
        
        # Chart 3: Relationships or Top Categories
        if chart_idx < n_charts:
            if len(numerical_cols) >= 2:
                # Scatter plot for relationship between first two numerical columns
                fig.add_trace(
                    go.Scatter(
                        x=df[numerical_cols[0]], 
                        y=df[numerical_cols[1]], 
                        mode='markers',
                        name=f"{numerical_cols[0]} vs {numerical_cols[1]}",
                        showlegend=False,
                        marker=dict(size=8, opacity=0.6)
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text=numerical_cols[0], row=row, col=col)
                fig.update_yaxes(title_text=numerical_cols[1], row=row, col=col)
            elif categorical_cols:
                # Bar chart for top categories in first categorical column
                top_categories = df[categorical_cols[0]].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=top_categories.index.astype(str), 
                        y=top_categories.values, 
                        name=f"Top {categorical_cols[0]}",
                        showlegend=False,
                        marker_color='coral'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text=categorical_cols[0], row=row, col=col)
                fig.update_yaxes(title_text="Count", row=row, col=col)
            else:
                # Fallback: histogram of first numerical column
                if numerical_cols:
                    fig.add_trace(
                        go.Histogram(x=df[numerical_cols[0]], name=f"Distribution of {numerical_cols[0]}", showlegend=False),
                        row=row, col=col
                    )
                    fig.update_xaxes(title_text=numerical_cols[0], row=row, col=col)
                    fig.update_yaxes(title_text="Count", row=row, col=col)
            col += 1
            if col > cols:
                col = 1
                row += 1
            chart_idx += 1
        
        # Chart 4: Trends or Correlation
        if chart_idx < n_charts:
            if date_cols and numerical_cols:
                # Time series plot
                if len(date_cols) > 0 and len(numerical_cols) > 0:
                    # Group by date and aggregate the first numerical column
                    agg_data = df.groupby(df[date_cols[0]].dt.date)[numerical_cols[0]].mean().reset_index()
                    fig.add_trace(
                        go.Scatter(
                            x=agg_data[date_cols[0]], 
                            y=agg_data[numerical_cols[0]], 
                            mode='lines+markers', 
                            name=f"Trend of {numerical_cols[0]}",
                            showlegend=False,
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
                    fig.update_xaxes(title_text=date_cols[0], row=row, col=col)
                    fig.update_yaxes(title_text=numerical_cols[0], row=row, col=col)
            elif len(numerical_cols) >= 2:
                # Correlation heatmap
                corr_data = df[numerical_cols[:6]].corr()  # Limit to first 6 numerical columns
                fig.add_trace(
                    go.Heatmap(
                        z=corr_data.values,
                        x=corr_data.columns,
                        y=corr_data.columns,
                        colorscale='RdBu',
                        zmid=0,
                        name="Correlation",
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                # Fallback: pie chart of categorical distribution
                if categorical_cols:
                    top_categories = df[categorical_cols[0]].value_counts().head(8)
                    fig.add_trace(
                        go.Pie(
                            labels=top_categories.index.astype(str),
                            values=top_categories.values,
                            name="Category Distribution",
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            col += 1
            if col > cols:
                col = 1
                row += 1
            chart_idx += 1
        
        # Additional charts based on available data
        while chart_idx < n_charts and (row <= rows):
            if numerical_cols and len(numerical_cols) > 1 and chart_idx < n_charts:
                # Box plot for distribution comparison
                box_col = numerical_cols[min(chart_idx-1, len(numerical_cols)-1)]
                fig.add_trace(
                    go.Box(y=df[box_col], name=f"Box plot of {box_col}", showlegend=False),
                    row=row, col=col
                )
                fig.update_yaxes(title_text=box_col, row=row, col=col)
            elif categorical_cols and chart_idx < n_charts:
                # Another categorical visualization
                cat_col = categorical_cols[min(chart_idx % len(categorical_cols), len(categorical_cols)-1)]
                value_counts = df[cat_col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=value_counts.values,
                        y=value_counts.index.astype(str),
                        orientation='h',
                        name=f"Horizontal bar of {cat_col}",
                        showlegend=False,
                        marker_color='lightgreen'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Count", row=row, col=col)
                fig.update_yaxes(title_text=cat_col, row=row, col=col)
            
            col += 1
            if col > cols:
                col = 1
                row += 1
            chart_idx += 1
        
        fig.update_layout(
            height=300*rows, 
            showlegend=True, 
            title_text="AI-Analyzed Dashboard",
            title_x=0.5
        )
        return fig


def main():
    st.set_page_config(
        page_title="Multi-Agent Dashboard System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Multi-Agent Dashboard System")
    st.markdown("""
    This system includes two specialized agents:
    1. **Data Cleaning Assistant** - Answers questions about data cleaning best practices
    2. **Excel-to-Dashboard Generator** - Creates dashboards from Excel files
    """)
    
    # Initialize session state
    if 'cleaning_agent' not in st.session_state:
        st.session_state.cleaning_agent = None
    if 'dashboard_agent' not in st.session_state:
        st.session_state.dashboard_agent = None
    if 'cleaning_chat_history' not in st.session_state:
        st.session_state.cleaning_chat_history = []
    if 'excel_file' not in st.session_state:
        st.session_state.excel_file = None
    if 'dashboard_fig' not in st.session_state:
        st.session_state.dashboard_fig = None
    
    # Create tabs for the two agents
    tab1, tab2 = st.tabs(["🧹 Data Cleaning Assistant", "📈 Excel-to-Dashboard Generator"])
    
    with tab1:
        st.header("Data Cleaning Best Practices Assistant")
        st.markdown("""
        This assistant uses a Retrieval-Augmented Generation (RAG) system to answer questions 
        about data cleaning best practices based on the provided document.
        """)
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # API Key input
            api_key = st.text_input("Mistral API Key", type="password", 
                                   value=os.getenv("MISTRAL_API_KEY", ""))
            
            # Agent ID input
            cleaning_agent_id = st.text_input("Data Cleaning Agent ID", 
                                            value="ag_019be5c4d85c7705a2d5079f0cc40d72")
            
            # Initialize agent button
            if st.button("Initialize Data Cleaning Agent") and api_key and cleaning_agent_id:
                try:
                    st.session_state.cleaning_agent = RAGDataCleaningAgent(
                        agent_id=cleaning_agent_id, 
                        pdf_path="Data-cleaning.pdf",
                        api_key=api_key
                    )
                    st.success("Data Cleaning Agent initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")
            
            st.divider()
            
            # Suggested questions
            st.header("💡 Suggested Questions")
            suggested_questions = [
                "What are the most important steps in data cleaning?",
                "How should I handle missing values in my dataset?",
                "What are the best practices for removing duplicates?",
                "How do I detect and handle outliers in my data?",
                "How should I normalize or standardize my data?",
                "What are common data quality issues I should look for?",
                "How do I handle inconsistent data formats?"
            ]
            
            for i, question in enumerate(suggested_questions):
                if st.button(f"Ask: {question[:50]}...", key=f"suggest_cleaning_{i}"):
                    if "current_cleaning_question" not in st.session_state:
                        st.session_state.current_cleaning_question = question
                    else:
                        st.session_state.current_cleaning_question = question
                    st.rerun()
        
        # Main chat interface for cleaning agent
        if st.session_state.cleaning_agent:
            # Display chat history
            for message in st.session_state.cleaning_chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input for new question - always show the input
            user_input = st.chat_input("Ask a question about data cleaning best practices...", key="cleaning_input")
            
            # Check if there's a suggested question to ask
            if "current_cleaning_question" in st.session_state and st.session_state.current_cleaning_question:
                user_input = st.session_state.current_cleaning_question
                st.session_state.current_cleaning_question = None  # Reset after using
            
            if user_input:
                # Add user message to chat history
                st.session_state.cleaning_chat_history.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant information from the document..."):
                        # Show which chunks were retrieved
                        relevant_chunks = st.session_state.cleaning_agent._retrieve_relevant_chunks(user_input)
                        
                        if relevant_chunks:
                            with st.expander("📖 Retrieved from document", expanded=False):
                                for i, chunk in enumerate(relevant_chunks, 1):
                                    st.markdown(f"**Section {i}:** {chunk[:200]}...")
                        
                        # Get response from agent
                        response = st.session_state.cleaning_agent.ask_question(user_input)
                    
                    if response and hasattr(response, 'outputs') and response.outputs:
                        # Extract the content from the response
                        assistant_response = response.outputs[0].content
                        st.markdown(assistant_response)
                        
                        # Add assistant response to chat history
                        st.session_state.cleaning_chat_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                    else:
                        st.error("Failed to get response from agent. Please try again.")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", key="clear_cleaning_chat"):
                st.session_state.cleaning_chat_history = []
                st.rerun()
        
        else:
            st.info("👆 Please enter your API key and Agent ID in the sidebar and click 'Initialize Data Cleaning Agent'")
    
    with tab2:
        st.header("Excel-to-Dashboard Generator")
        st.markdown("""
        Upload an Excel file to automatically generate a rich dashboard with multiple visualizations.
        The AI agent analyzes your data and creates appropriate visualizations.
        """)
        
        # Dashboard agent configuration
        with st.sidebar:
            st.header("🔧 Dashboard Agent Config")
            
            # API Key input (reuse from cleaning agent)
            if 'api_key' not in locals():
                api_key = st.text_input("Mistral API Key", type="password", 
                                       value=os.getenv("MISTRAL_API_KEY", ""), key="dashboard_api_key")
            
            # Dashboard Agent ID input
            dashboard_agent_id = st.text_input("Dashboard Agent ID", 
                                             value="ag_019be5fc5a8571f2894c56acc1962e3b")
            
            # Initialize dashboard agent button
            if st.button("Initialize Dashboard Agent", key="init_dashboard") and api_key and dashboard_agent_id:
                try:
                    st.session_state.dashboard_agent = ExcelToDashboardAgent(
                        agent_id=dashboard_agent_id,
                        api_key=api_key
                    )
                    st.success("Dashboard Agent initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing dashboard agent: {e}")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an Excel file", 
            type=["xlsx", "xls"],
            key="excel_uploader"
        )
        
        if uploaded_file is not None:
            st.session_state.excel_file = uploaded_file
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Save the uploaded file temporarily
            with open("temp_uploaded_file.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.session_state.dashboard_agent:
                with st.spinner("Analyzing Excel file and generating rich dashboard..."):
                    try:
                        response, df, info = st.session_state.dashboard_agent.analyze_excel("temp_uploaded_file.xlsx")
                        
                        # Display basic info about the file
                        st.subheader("File Information")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", info['shape'][0])
                        with col2:
                            st.metric("Columns", info['shape'][1])
                        with col3:
                            st.metric("Numeric Columns", len(info['numerical_columns']))
                        
                        # Show sample of the data
                        st.subheader("Sample of Data")
                        st.dataframe(df.head())
                        
                        # Generate and display rich dashboard
                        st.subheader("AI-Generated Rich Dashboard")
                        dashboard_fig = st.session_state.dashboard_agent.generate_rich_dashboard(df)
                        st.plotly_chart(dashboard_fig, use_container_width=True)
                        
                        # Show agent analysis if available
                        if response and hasattr(response, 'outputs') and response.outputs:
                            st.subheader("AI Insights")
                            st.markdown(response.outputs[0].content)
                        
                    except Exception as e:
                        st.error(f"Error processing Excel file: {e}")
            else:
                st.warning("Please initialize the Dashboard Agent first")
        
        else:
            st.info("👆 Please upload an Excel file to generate a dashboard")


if __name__ == "__main__":
    main()