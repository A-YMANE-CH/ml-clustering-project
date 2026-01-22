"""
Improved ML Problem Classification Agent - Enhanced K-Means and KNN
Better implementations with advanced preprocessing and parameter optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, Literal
import operator
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Define the state structure for our agent
class AgentState(TypedDict):
    problem_description: str
    dataset: pd.DataFrame
    problem_type: str
    reasoning: str
    model_results: dict
    figure: object
    messages: Annotated[list, operator.add]


class MLProblemAgent:
    """Agentic flow for ML problem classification and solution with improved algorithms"""

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_problem", self.analyze_problem)
        workflow.add_node("apply_kmeans", self.apply_kmeans)
        workflow.add_node("apply_knn", self.apply_knn)
        workflow.add_node("finalize", self.finalize_results)

        workflow.set_entry_point("analyze_problem")

        workflow.add_conditional_edges(
            "analyze_problem",
            self.route_to_algorithm,
            {
                "segmentation": "apply_kmeans",
                "classification": "apply_knn"
            }
        )

        workflow.add_edge("apply_kmeans", "finalize")
        workflow.add_edge("apply_knn", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def analyze_problem(self, state: AgentState) -> AgentState:
        dataset_info = f"""
        Dataset shape: {state['dataset'].shape}
        Columns: {list(state['dataset'].columns)}
        First few rows:
        {state['dataset'].head().to_string()}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ML engineer. Analyze the problem description and dataset.

Your task is to determine if this is a:
1. CLASSIFICATION problem (predicting categories/labels from features, requires labeled target variable)
2. SEGMENTATION problem (grouping similar data points, unsupervised clustering)

IMPORTANT RULES:
- If the dataset has a clear target/label column for prediction → CLASSIFICATION
- If the goal is to find patterns/groups without predefined labels → SEGMENTATION
- If the problem mentions "predict", "classify", "categorize" with known labels → CLASSIFICATION
- If the problem mentions "segment", "cluster", "group", "discover patterns" → SEGMENTATION

Respond in EXACTLY this format:
PROBLEM_TYPE: [classification or segmentation]
REASONING: [brief explanation]"""),
            ("user", f"""Problem Description: {state['problem_description']}

Dataset Information:
{dataset_info}

Analyze and classify this problem.""")
        ])

        response = self.llm.invoke(prompt.format_messages())
        response_text = response.content

        lines = response_text.strip().split('\n')
        problem_type = None
        reasoning = ""

        for line in lines:
            if line.startswith("PROBLEM_TYPE:"):
                problem_type = line.split(":", 1)[1].strip().lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        if problem_type not in ['classification', 'segmentation']:
            problem_type = 'segmentation'

        state['problem_type'] = problem_type
        state['reasoning'] = reasoning
        if f"Problem analyzed as: {problem_type}" not in state['messages']:
            state['messages'].append(f"Problem analyzed as: {problem_type}")
        if f"Reasoning: {reasoning}" not in state['messages']:
            state['messages'].append(f"Reasoning: {reasoning}")

        return state

    def route_to_algorithm(self, state: AgentState) -> Literal["classification", "segmentation"]:
        return state['problem_type']

    def apply_kmeans(self, state: AgentState) -> AgentState:
        if "Applying Improved K-Means clustering with advanced optimization..." not in state['messages']:
            state['messages'].append("Applying Improved K-Means clustering with advanced optimization...")

        try:
            df = state['dataset'].copy()

            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) == 0:
                if "ERROR: No numeric columns found in dataset" not in state['messages']:
                    state['messages'].append("ERROR: No numeric columns found in dataset")
                state['model_results'] = {'error': 'No numeric columns found'}
                return state

            X = df[numeric_cols].copy()
            if f"Using {len(numeric_cols)} numeric features: {numeric_cols}" not in state['messages']:
                state['messages'].append(f"Using {len(numeric_cols)} numeric features: {numeric_cols}")

            # Handle missing values with median imputation
            X = X.fillna(X.median())

            # Check if we have valid data
            if X.shape[0] < 2:
                if "ERROR: Need at least 2 samples for clustering" not in state['messages']:
                    state['messages'].append("ERROR: Need at least 2 samples for clustering")
                state['model_results'] = {'error': 'Insufficient samples'}
                return state

            # Advanced scaling with outlier-resistant RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            if "Data scaled using RobustScaler (resistant to outliers)" not in state['messages']:
                state['messages'].append("Data scaled using RobustScaler (resistant to outliers)")

            # Determine optimal number of clusters using multiple methods
            max_clusters = min(10, max(2, len(X) // 5))  # More conservative
            if max_clusters < 2:
                max_clusters = 2

            if f"Testing {max_clusters - 1} different cluster configurations..." not in state['messages']:
                state['messages'].append(f"Testing {max_clusters - 1} different cluster configurations...")

            # Test different numbers of clusters with K-Means
            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []
            all_labels = []

            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500,
                                init='k-means++')  # Better initialization
                clusters = kmeans.fit_predict(X_scaled)

                inertias.append(kmeans.inertia_)
                silhouette = silhouette_score(X_scaled, clusters)
                davies_bouldin = davies_bouldin_score(X_scaled, clusters)

                silhouette_scores.append(silhouette)
                davies_bouldin_scores.append(davies_bouldin)
                all_labels.append(clusters)

            # Choose optimal k using silhouette score (higher is better)
            optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
            optimal_silhouette = max(silhouette_scores)

            # Choose optimal k using Davies-Bouldin (lower is better)
            optimal_k_db = davies_bouldin_scores.index(min(davies_bouldin_scores)) + 2

            # Use silhouette as primary metric but validate with DB
            optimal_k = optimal_k_silhouette

            # Final clustering with optimal k
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50, max_iter=500,
                                  init='k-means++')
            final_clusters = final_kmeans.fit_predict(X_scaled)

            # Recalculate metrics for final clustering
            final_silhouette = silhouette_score(X_scaled, final_clusters)
            final_davies_bouldin = davies_bouldin_score(X_scaled, final_clusters)

            # Calculate cluster statistics
            cluster_sizes = pd.Series(final_clusters).value_counts().sort_index()

            state['model_results'] = {
                'algorithm': 'K-Means',
                'n_clusters': optimal_k,
                'silhouette_score': float(final_silhouette),
                'davies_bouldin_score': float(final_davies_bouldin),
                'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.to_dict().items()},
                'inertia': float(final_kmeans.inertia_),
                'all_silhouette_scores': {k+2: float(s) for k, s in enumerate(silhouette_scores)},
                'all_davies_bouldin_scores': {k+2: float(s) for k, s in enumerate(davies_bouldin_scores)},
                'features_used': numeric_cols
            }

            # Create enhanced visualization
            state['figure'] = self._visualize_improved_kmeans(X_scaled, final_clusters, optimal_k,
                                                            inertias, silhouette_scores, davies_bouldin_scores)

            if f"✓ Improved K-Means completed: {optimal_k} clusters, Silhouette={final_silhouette:.3f}" not in state['messages']:
                state['messages'].append(f"✓ Improved K-Means completed: {optimal_k} clusters, Silhouette={final_silhouette:.3f}")

        except Exception as e:
            if f"ERROR in Improved K-Means: {str(e)}" not in state['messages']:
                state['messages'].append(f"ERROR in Improved K-Means: {str(e)}")
            state['model_results'] = {'error': str(e)}
            import traceback
            tb_str = traceback.format_exc()
            if tb_str not in state['messages']:
                state['messages'].append(tb_str)

        return state

    def apply_knn(self, state: AgentState) -> AgentState:
        if "Applying K-Nearest Neighbors classifier with optimization..." not in state['messages']:
            state['messages'].append("Applying K-Nearest Neighbors classifier with optimization...")

        try:
            df = state['dataset'].copy()

            # Find target column
            target_candidates = ['target', 'label', 'class', 'y', 'output']
            target_col = None

            for col in df.columns:
                if col.lower() in target_candidates:
                    target_col = col
                    break

            if target_col is None:
                target_col = df.columns[-1]

            if f"Using '{target_col}' as target column" not in state['messages']:
                state['messages'].append(f"Using '{target_col}' as target column")

            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            if f"Target classes: {sorted(y.unique().tolist())}" not in state['messages']:
                state['messages'].append(f"Target classes: {sorted(y.unique().tolist())}")
            if f"Class distribution: {y.value_counts().to_dict()}" not in state['messages']:
                state['messages'].append(f"Class distribution: {y.value_counts().to_dict()}")

            # Get numeric columns only
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) == 0:
                if "ERROR: No numeric features found" not in state['messages']:
                    state['messages'].append("ERROR: No numeric features found")
                state['model_results'] = {'error': 'No numeric features found'}
                return state

            if f"Using {len(numeric_cols)} features: {numeric_cols}" not in state['messages']:
                state['messages'].append(f"Using {len(numeric_cols)} features: {numeric_cols}")

            X = X[numeric_cols].copy()
            X = X.fillna(X.median())  # Use median for robust imputation

            # Check for valid data
            if len(X) < 5:
                if "ERROR: Need at least 5 samples for classification" not in state['messages']:
                    state['messages'].append("ERROR: Need at least 5 samples for classification")
                state['model_results'] = {'error': 'Insufficient samples'}
                return state

            # Split data with better test size
            test_size = min(0.3, max(0.2, 50 / len(X)))  # Adaptive test size
            stratify_param = y if len(y.unique()) > 1 and len(y) > 10 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_param
            )

            if f"Train size: {len(X_train)}, Test size: {len(X_test)}" not in state['messages']:
                state['messages'].append(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Scale features (critical for KNN)
            scaler = RobustScaler()  # More robust to outliers
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Hyperparameter tuning for k
            k_range = range(1, min(31, len(X_train) // 2), 2)  # Test odd numbers
            k_scores = []

            if f"Testing k values: {list(k_range)[:10]}..." not in state['messages']:
                state['messages'].append(f"Testing k values: {list(k_range)[:10]}...")

            from sklearn.model_selection import cross_val_score

            best_k = None
            best_score = 0

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
                # Use cross-validation for more reliable evaluation
                scores = cross_val_score(knn, X_train_scaled, y_train, cv=min(5, len(X_train)//2), scoring='accuracy')
                mean_score = scores.mean()
                k_scores.append(mean_score)

                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k

            optimal_k = best_k if best_k else 5
            if f"Optimal k selected: {optimal_k} (CV accuracy: {best_score:.3f})" not in state['messages']:
                state['messages'].append(f"Optimal k selected: {optimal_k} (CV accuracy: {best_score:.3f})")

            # Train final model with optimal k
            knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance', metric='euclidean')
            knn.fit(X_train_scaled, y_train)

            # Predict
            y_pred = knn.predict(X_test_scaled)
            y_pred_proba = knn.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = knn.score(X_test_scaled, y_test)
            train_accuracy = knn.score(X_train_scaled, y_train)

            # Get classification report
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            except:
                report = {}

            # Calculate per-class accuracies
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)

            state['model_results'] = {
                'algorithm': 'KNN',
                'n_neighbors': optimal_k,
                'accuracy': float(accuracy),
                'train_accuracy': float(train_accuracy),
                'cv_accuracy': float(best_score),
                'classification_report': report,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'features_used': numeric_cols,
                'k_scores': {int(k): float(s) for k, s in zip(k_range, k_scores)},
                'confusion_matrix': cm.tolist()
            }

            # Create visualization
            state['figure'] = self._visualize_knn(
                X_train_scaled, X_test_scaled, y_train, y_test, y_pred,
                k_range, k_scores, optimal_k
            )

            if f"✓ KNN completed: k={optimal_k}, Test accuracy={accuracy:.3f}, Train accuracy={train_accuracy:.3f}" not in state['messages']:
                state['messages'].append(f"✓ KNN completed: k={optimal_k}, Test accuracy={accuracy:.3f}, Train accuracy={train_accuracy:.3f}")

            if train_accuracy - accuracy > 0.15 and "⚠ Warning: Possible overfitting detected (train accuracy >> test accuracy)" not in state['messages']:
                state['messages'].append("⚠ Warning: Possible overfitting detected (train accuracy >> test accuracy)")

        except Exception as e:
            if f"ERROR in KNN: {str(e)}" not in state['messages']:
                state['messages'].append(f"ERROR in KNN: {str(e)}")
            state['model_results'] = {'error': str(e)}
            import traceback
            tb_str = traceback.format_exc()
            if tb_str not in state['messages']:
                state['messages'].append(tb_str)

        return state

    def _visualize_improved_kmeans(self, X_scaled, clusters, n_clusters, inertias, silhouette_scores, davies_bouldin_scores):
        """Create enhanced K-Means visualization with PCA projection"""
        # Close any existing figures
        plt.close('all')

        fig = plt.figure(figsize=(20, 5))
        gs = fig.add_gridspec(1, 4, wspace=0.3)

        try:
            # Plot 1: PCA-based cluster visualization (most informative 2D projection)
            ax1 = fig.add_subplot(gs[0])
            
            # Apply PCA to get the most informative 2D projection
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=clusters, cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidths=0.5)
            ax1.set_title(f'K-Means Clustering (k={n_clusters})\nPCA Projection', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=10)
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=10)
            plt.colorbar(scatter, ax=ax1, label='Cluster')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Elbow curve
            ax2 = fig.add_subplot(gs[1])
            k_values = list(range(2, len(inertias) + 2))
            ax2.plot(k_values, inertias, marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=6)
            ax2.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
            ax2.set_title('Elbow Method', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=10)
            ax2.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Silhouette score
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(k_values, silhouette_scores, marker='s', linestyle='-', color='green', linewidth=2, markersize=6)
            ax3.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
            ax3.set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Clusters (k)', fontsize=10)
            ax3.set_ylabel('Silhouette Score', fontsize=10)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Davies-Bouldin score
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(k_values, davies_bouldin_scores, marker='^', linestyle='-', color='orange', linewidth=2, markersize=6)
            ax4.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected k={n_clusters}')
            ax4.set_title('Davies-Bouldin Score Analysis', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Number of Clusters (k)', fontsize=10)
            ax4.set_ylabel('Davies-Bouldin Score', fontsize=10)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

        except Exception as e:
            print(f"Error creating visualization: {e}")
            ax1.text(0.5, 0.5, f'Visualization error:\n{str(e)}',
                    ha='center', va='center', fontsize=10)

        return fig

    def _visualize_knn(self, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, k_range, k_scores, optimal_k):
        """Create KNN visualization and return the figure"""
        # Close any existing figures
        plt.close('all')

        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, wspace=0.3)

        try:
            # Plot 1: True labels
            ax1 = fig.add_subplot(gs[0])
            if X_test_scaled.shape[1] >= 2 and len(X_test_scaled) > 0:
                # Apply PCA for visualization if more than 2 dimensions
                if X_test_scaled.shape[1] > 2:
                    pca = PCA(n_components=2, random_state=42)
                    X_test_2d = pca.fit_transform(X_test_scaled)
                else:
                    X_test_2d = X_test_scaled
                
                # Convert to numeric if needed
                y_test_numeric = pd.factorize(y_test)[0]
                y_pred_numeric = pd.factorize(y_pred)[0]

                scatter1 = ax1.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                      c=y_test_numeric, cmap='viridis', alpha=0.7,
                                      edgecolors='black', s=60, linewidths=0.5)
                ax1.set_title('True Labels (Test Set)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Feature 1 (scaled)', fontsize=10)
                ax1.set_ylabel('Feature 2 (scaled)', fontsize=10)
                plt.colorbar(scatter1, ax=ax1, label='Class')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Need at least 2 features\nfor 2D visualization',
                       ha='center', va='center', fontsize=12)
                ax1.set_title('Insufficient features')

            # Plot 2: Predicted labels
            ax2 = fig.add_subplot(gs[1])
            if X_test_scaled.shape[1] >= 2 and len(X_test_scaled) > 0:
                # Apply PCA for visualization if more than 2 dimensions
                if X_test_scaled.shape[1] > 2:
                    pca = PCA(n_components=2, random_state=42)
                    X_test_2d = pca.fit_transform(X_test_scaled)
                else:
                    X_test_2d = X_test_scaled
                
                # Mark incorrect predictions with X
                correct = y_test_numeric == y_pred_numeric

                scatter2 = ax2.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                      c=y_pred_numeric, cmap='viridis', alpha=0.7,
                                      edgecolors='black', s=60, linewidths=0.5)

                # Mark incorrect predictions
                if not all(correct):
                    ax2.scatter(X_test_2d[~correct, 0], X_test_2d[~correct, 1],
                               marker='x', s=200, c='red', linewidths=3, label='Misclassified')

                ax2.set_title('Predicted Labels (Test Set)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Feature 1 (scaled)', fontsize=10)
                ax2.set_ylabel('Feature 2 (scaled)', fontsize=10)
                plt.colorbar(scatter2, ax=ax2, label='Class')
                if not all(correct):
                    ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Need at least 2 features\nfor 2D visualization',
                       ha='center', va='center', fontsize=12)
                ax2.set_title('Insufficient features')

            # Plot 3: K value optimization
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(list(k_range), k_scores, marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=6)
            ax3.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
            ax3.set_title('K-Value Optimization (Cross-Validation)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Neighbors (k)', fontsize=10)
            ax3.set_ylabel('Cross-Validation Accuracy', fontsize=10)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

        except Exception as e:
            print(f"Error creating visualization: {e}")
            ax1.text(0.5, 0.5, f'Visualization error:\n{str(e)}',
                    ha='center', va='center', fontsize=10)

        return fig

    def finalize_results(self, state: AgentState) -> AgentState:
        state['messages'].append("Analysis complete!")
        return state

    def run(self, problem_description: str, dataset: pd.DataFrame) -> dict:
        initial_state = {
            'problem_description': problem_description,
            'dataset': dataset,
            'problem_type': '',
            'reasoning': '',
            'model_results': {},
            'figure': None,
            'messages': []
        }

        final_state = self.graph.invoke(initial_state)

        return {
            'problem_type': final_state['problem_type'],
            'reasoning': final_state['reasoning'],
            'results': final_state['model_results'],
            'figure': final_state.get('figure'),
            'messages': final_state['messages']
        }


# Streamlit App
def main():
    st.set_page_config(page_title="Improved ML Problem Agent", page_icon="🤖", layout="wide")

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    st.title("🤖 Improved ML Problem Classification Agent")
    st.markdown("**Enhanced K-Means and KNN with Better Results**")
    st.markdown("This agent analyzes your ML problem and automatically applies improved K-Means (segmentation) or KNN (classification)")

    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your free API key from https://makersuite.google.com/app/apikey"
        )

        if not api_key:
            st.warning("Please enter your Google API key to continue")
            st.info("💡 Get a free key at [Google AI Studio](https://makersuite.google.com/app/apikey)")

        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. Enter your Google API key
        2. Choose to upload data or use examples
        3. Describe your ML problem
        4. Click 'Analyze Problem'
        5. View results and visualizations
        """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📊 Dataset Input")

        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV", "Use Example - Customer Segmentation", "Use Example - Iris Classification"]
        )

        df = None

        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error loading CSV: {str(e)}")

        elif data_source == "Use Example - Customer Segmentation":
            np.random.seed(42)
            # Create a more complex dataset with clearer clusters
            cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0.2], [0.2, 0.5]], 60)
            cluster2 = np.random.multivariate_normal([6, 6], [[0.8, -0.3], [-0.3, 0.8]], 60)
            cluster3 = np.random.multivariate_normal([2, 6], [[0.6, 0.1], [0.1, 0.6]], 60)
            
            data_points = np.vstack([cluster1, cluster2, cluster3])
            df = pd.DataFrame(data_points, columns=['feature1', 'feature2'])
            df['age'] = np.random.randint(18, 70, len(df))
            df['income'] = np.random.randint(20000, 150000, len(df))
            st.success("✅ Loaded example customer segmentation dataset with clear clusters")

        elif data_source == "Use Example - Iris Classification":
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            st.success("✅ Loaded example Iris classification dataset")

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Dataset Info")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows", df.shape[0])
            col_b.metric("Columns", df.shape[1])
            col_c.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

    with col2:
        st.header("📝 Problem Description")

        if data_source == "Use Example - Customer Segmentation":
            default_desc = "what's the type of this problemwhat's the type of this problem."
        elif data_source == "Use Example - Iris Classification":
            default_desc = "what's the type of this problem."
        else:
            default_desc = ""

        problem_description = st.text_area(
            "Describe your machine learning problem:",
            value=default_desc,
            height=150,
            placeholder="e.g., I want to predict customer churn based on usage patterns..."
        )

        st.markdown("---")

        analyze_button = st.button("🚀 Analyze Problem", type="primary", use_container_width=True)

    # Analysis section
    if analyze_button:
        if not api_key:
            st.error("❌ Please enter your Google API key in the sidebar")
            st.session_state.analyzed = False
        elif df is None:
            st.error("❌ Please upload a dataset or select an example")
            st.session_state.analyzed = False
        elif not problem_description.strip():
            st.error("❌ Please describe your machine learning problem")
            st.session_state.analyzed = False
        else:
            with st.spinner("🔍 Analyzing your problem with AI..."):
                try:
                    agent = MLProblemAgent(api_key=api_key)
                    st.session_state.results = agent.run(problem_description, df)
                    st.session_state.analyzed = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
                    st.session_state.analyzed = False

    # Display results if available
    if st.session_state.analyzed and st.session_state.results is not None:
        results = st.session_state.results

        st.success("✅ Analysis Complete!")

        # Results section
        st.markdown("---")
        st.header("📊 Results")

        # Check for errors
        if 'error' in results.get('results', {}):
            st.error(f"Analysis error: {results['results']['error']}")
            with st.expander("📝 View Error Log"):
                for msg in results.get('messages', []):
                    st.text(msg)
            return

        # Problem type and reasoning
        col1, col2 = st.columns([1, 2])
        with col1:
            problem_type = results['problem_type'].upper()
            if problem_type == "CLASSIFICATION":
                st.success(f"### 🎯 {problem_type}")
            else:
                st.info(f"### 🔍 {problem_type}")

        with col2:
            st.markdown("**Reasoning:**")
            st.write(results['reasoning'])

        st.markdown("---")

        # Model results
        st.subheader("📈 Model Results")

        model_results = results['results']

        if model_results.get('algorithm') == 'K-Means':
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Number of Clusters", model_results['n_clusters'])
            col2.metric("Silhouette Score", f"{model_results['silhouette_score']:.4f}",
                       help="Higher is better (range: -1 to 1). Values > 0.5 indicate good clustering.")
            col3.metric("Davies-Bouldin Score", f"{model_results['davies_bouldin_score']:.4f}",
                       help="Lower is better. Values < 1.0 indicate good clustering.")
            col4.metric("Inertia", f"{model_results['inertia']:.2f}",
                       help="Within-cluster sum of squares. Lower indicates tighter clusters.")

            st.markdown("**Cluster Sizes:**")
            cluster_df = pd.DataFrame(
                list(model_results['cluster_sizes'].items()),
                columns=['Cluster', 'Size']
            ).sort_values('Cluster')
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)

            # Quality interpretation
            sil_score = model_results['silhouette_score']
            if sil_score > 0.7:
                st.success("✓ Excellent clustering quality - clusters are well-separated and distinct")
            elif sil_score > 0.5:
                st.info("✓ Good clustering quality - reasonable separation between clusters")
            elif sil_score > 0.3:
                st.warning("⚠ Fair clustering quality - some overlap between clusters")
            else:
                st.error("⚠ Poor clustering quality - consider fewer/more clusters or different features")

        elif model_results.get('algorithm') == 'KNN':
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("K-Neighbors", model_results['n_neighbors'])
            col2.metric("Test Accuracy", f"{model_results['accuracy']:.4f}")
            col3.metric("Train Accuracy", f"{model_results.get('train_accuracy', 0):.4f}")
            col4.metric("CV Accuracy", f"{model_results.get('cv_accuracy', 0):.4f}",
                       help="Cross-validation accuracy during k optimization")

            # Model quality interpretation
            test_acc = model_results['accuracy']
            train_acc = model_results.get('train_accuracy', 0)

            if test_acc > 0.9:
                st.success("✓ Excellent classification performance")
            elif test_acc > 0.75:
                st.info("✓ Good classification performance")
            elif test_acc > 0.6:
                st.warning("⚠ Fair classification performance - consider feature engineering")
            else:
                st.error("⚠ Poor classification performance - may need more/better features")

            if train_acc - test_acc > 0.15:
                st.warning("⚠ Possible overfitting detected (training accuracy much higher than test)")

            if model_results.get('classification_report'):
                st.markdown("**Classification Report:**")
                report = model_results['classification_report']

                # Convert classification report to DataFrame
                try:
                    report_df = pd.DataFrame(report).transpose()
                    report_df = report_df.round(4)
                    st.dataframe(report_df, use_container_width=True)
                except:
                    st.json(report)

        # Visualization
        st.markdown("---")
        st.subheader("📊 Visualization")

        if results.get('figure') is not None:
            try:
                st.pyplot(results['figure'])
                plt.close(results['figure'])
            except Exception as e:
                st.warning(f"Could not display visualization: {str(e)}")
        else:
            st.info("No visualization available")

        # Messages log
        with st.expander("📝 View Process Log"):
            for msg in results.get('messages', []):
                st.text(msg)


if __name__ == "__main__":
    main()