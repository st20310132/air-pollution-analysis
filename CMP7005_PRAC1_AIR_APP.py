import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="CMP7005 Data Analysis Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling for pages
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #1565C0;
        margin-top: 0.7rem;
        margin-bottom: 0.7rem;
    }
    .info-box {        
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Heading
st.markdown('<h1 class="main-header">Data Analysis Application</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Upload Data", "Data Overview", "Exploratory Data Analysis", "Modeling and Prediction"])

# Function to download data
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    st.pyplot(fig)

# Function to plot distribution of a column
def plot_distribution(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[column].dtype in ['int64', 'float64']:
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
    else:
        value_counts = df[column].value_counts()
        if len(value_counts) > 15:
            top_values = value_counts.head(15)
            sns.barplot(x=top_values.index, y=top_values.values, ax=ax)
            ax.set_title(f'Top 15 values of {column}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            sns.countplot(data=df, x=column, ax=ax)
            ax.set_title(f'Count of {column}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    st.pyplot(fig)

# Function to plot scatter plot
def plot_scatter(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'{x_col} vs {y_col}')
    st.pyplot(fig)

# Function to plot box plot
def plot_boxplot(df, x_col, y_col=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if y_col:
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f'Box Plot of {y_col} by {x_col}')
    else:
        sns.boxplot(data=df, x=x_col, ax=ax)
        ax.set_title(f'Box Plot of {x_col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

# Page: Upload Data
if page == "Upload Data":
    st.markdown('<h2 class="section-header">Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    with st.expander("Data Upload Instructions", expanded=True):
        st.markdown("""
        <div class="info-box">
        <p>Upload your dataset in CSV or Excel format. The application will automatically detect the file type and load it.</p>
        <p>You can also use one of the sample datasets to explore the application's features.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            # Show a sample of the data
            st.markdown('<h3 class="subsection-header">Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(data.head())
            
        except Exception as e:
            st.error(f"Error loading the dataset: {e}")
    
    # If data is loaded, show the next step guidance
    if st.session_state.data is not None:
        st.markdown('<h3 class="subsection-header">Next Steps</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p>Your data is now loaded! Use the navigation sidebar to:</p>
        <ul>
            <li><strong>Data Overview</strong>: View basic statistics and information about your dataset.</li>
            <li><strong>Exploratory Data Analysis</strong>: Create visualizations and analyze relationships.</li>
            <li><strong>Modeling and Prediction</strong>: Build machine learning models and make predictions.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Page: Data Overview
elif page == "Data Overview":
    if st.session_state.data is not None:
        data = st.session_state.data
        st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="subsection-header">Dataset Information</h3>', unsafe_allow_html=True)
            st.write(f"**Rows**: {data.shape[0]}")
            st.write(f"**Columns**: {data.shape[1]}")
            st.write(f"**Duplicated rows**: {data.duplicated().sum()}")
            memory_usage = data.memory_usage(deep=True).sum()
            st.write(f"**Memory usage**: {memory_usage / 1e6:.2f} MB")
        
        with col2:
            st.markdown('<h3 class="subsection-header">Data Types</h3>', unsafe_allow_html=True)
            dtype_counts = data.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            st.table(dtype_counts)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistics", "Missing Values", "Column Details"])
        
        with tab1:
            st.markdown('<h3 class="subsection-header">Data Preview</h3>', unsafe_allow_html=True)
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(data.head(num_rows))
            
            # Download option
            st.markdown(get_download_link(data, "dataset.csv", "Download data as CSV"), unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h3 class="subsection-header">Descriptive Statistics</h3>', unsafe_allow_html=True)
            
            # Numeric statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("**Numeric Columns Statistics:**")
                st.dataframe(data[numeric_cols].describe())
            
            # Categorical statistics
            cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                st.write("**Categorical Columns Statistics:**")
                cat_stats = pd.DataFrame({
                    'Column': cat_cols,
                    'Unique Values': [data[col].nunique() for col in cat_cols],
                    'Mode': [data[col].mode()[0] if not data[col].mode().empty else None for col in cat_cols],
                    'Mode Frequency': [data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0 for col in cat_cols],
                    'Mode Percentage': [data[col].value_counts().iloc[0] / len(data) * 100 if not data[col].value_counts().empty else 0 for col in cat_cols]
                })
                st.dataframe(cat_stats)
        
        with tab3:
            st.markdown('<h3 class="subsection-header">Missing Values Analysis</h3>', unsafe_allow_html=True)
            
            # Calculate missing values
            missing_values = data.isnull().sum().reset_index()
            missing_values.columns = ['Column', 'Missing Values']
            missing_values['Percentage'] = (missing_values['Missing Values'] / len(data) * 100).round(2)
            missing_values = missing_values.sort_values('Missing Values', ascending=False)
            
            if missing_values['Missing Values'].sum() > 0:
                # Show missing values table
                st.dataframe(missing_values)
                
                # Visualize missing values
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(missing_values['Column'], missing_values['Percentage'])
                ax.set_title('Percentage of Missing Values by Column')
                ax.set_xlabel('Columns')
                ax.set_ylabel('Percentage Missing (%)')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                
                # Handling missing values
                st.markdown('<h4 class="subsection-header">Handle Missing Values</h4>', unsafe_allow_html=True)
                columns_with_missing = missing_values[missing_values['Missing Values'] > 0]['Column'].tolist()
                
                if columns_with_missing:
                    selected_col = st.selectbox("Select column to handle missing values", columns_with_missing)
                    handling_method = st.selectbox(
                        f"Choose method to handle missing values in {selected_col}",
                        ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with a constant value"]
                    )
                    
                    if st.button("Apply"):
                        if handling_method == "Drop rows":
                            data = data.dropna(subset=[selected_col])
                            st.success(f"Dropped rows with missing values in {selected_col}")
                        
                        elif handling_method == "Fill with mean" and data[selected_col].dtype in ['int64', 'float64']:
                            data[selected_col] = data[selected_col].fillna(data[selected_col].mean())
                            st.success(f"Filled missing values in {selected_col} with mean")
                        
                        elif handling_method == "Fill with median" and data[selected_col].dtype in ['int64', 'float64']:
                            data[selected_col] = data[selected_col].fillna(data[selected_col].median())
                            st.success(f"Filled missing values in {selected_col} with median")
                        
                        elif handling_method == "Fill with mode":
                            data[selected_col] = data[selected_col].fillna(data[selected_col].mode()[0])
                            st.success(f"Filled missing values in {selected_col} with mode")
                        
                        elif handling_method == "Fill with a constant value":
                            if data[selected_col].dtype in ['int64', 'float64']:
                                const_value = st.number_input("Enter constant value", value=0.0)
                            else:
                                const_value = st.text_input("Enter constant value", value="Unknown")
                            
                            data[selected_col] = data[selected_col].fillna(const_value)
                            st.success(f"Filled missing values in {selected_col} with {const_value}")
                        
                        # Update session state
                        st.session_state.data = data
                        st.experimental_rerun()
            else:
                st.success("No missing values found in the dataset.")
        
        with tab4:
            st.markdown('<h3 class="subsection-header">Column Details</h3>', unsafe_allow_html=True)
            
            selected_column = st.selectbox("Select a column to view details", data.columns)
            
            st.write(f"**Column:** {selected_column}")
            st.write(f"**Type:** {data[selected_column].dtype}")
            st.write(f"**Number of unique values:** {data[selected_column].nunique()}")
            st.write(f"**Number of missing values:** {data[selected_column].isnull().sum()}")
            
            # Show distribution
            st.markdown('<h4 class="subsection-header">Value Distribution</h4>', unsafe_allow_html=True)
            
            if data[selected_column].dtype in ['int64', 'float64']:
                # Numeric column
                st.write(f"**Min:** {data[selected_column].min()}")
                st.write(f"**Max:** {data[selected_column].max()}")
                st.write(f"**Mean:** {data[selected_column].mean()}")
                st.write(f"**Median:** {data[selected_column].median()}")
                st.write(f"**Standard Deviation:** {data[selected_column].std()}")
                
                # Histogram
                plot_distribution(data, selected_column)
                plot_boxplot(data, selected_column)
            else:
                # Categorical column
                value_counts = data[selected_column].value_counts()
                st.write("**Value Counts:**")
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(data) * 100).round(2)
                }))
                
                # Bar chart
                plot_distribution(data, selected_column)
            
    else:
        st.warning("Please upload a dataset first in the 'Upload Data' page.")

# Page: Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    if st.session_state.data is not None:
        data = st.session_state.data
        st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Correlation Analysis", "Distribution Analysis", "Relationship Analysis", "Custom Plot"])
        
        with tab1:
            st.markdown('<h3 class="subsection-header">Correlation Analysis</h3>', unsafe_allow_html=True)
            
            # Correlation matrix
            plot_correlation_heatmap(data)
            
            # Select top correlations
            numeric_df = data.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                st.markdown('<h4 class="subsection-header">Top Correlations</h4>', unsafe_allow_html=True)
                corr_matrix = numeric_df.corr().abs().unstack()
                corr_matrix = corr_matrix[corr_matrix < 1]  # Remove self-correlations
                top_correlations = corr_matrix.sort_values(ascending=False).head(10)
                
                top_corr_df = pd.DataFrame(top_correlations).reset_index()
                top_corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
                st.table(top_corr_df)
        
        with tab2:
            st.markdown('<h3 class="subsection-header">Distribution Analysis</h3>', unsafe_allow_html=True)
            
            # Select a column to visualize its distribution
            dist_col = st.selectbox("Select a column to visualize its distribution", data.columns)
            
            # Plot the distribution
            plot_distribution(data, dist_col)
            
            # If numeric, show additional stats
            if data[dist_col].dtype in ['int64', 'float64']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Skewness", f"{data[dist_col].skew():.3f}")
                with col2:
                    st.metric("Kurtosis", f"{data[dist_col].kurtosis():.3f}")
                with col3:
                    # Check normality using Shapiro-Wilk test if not too many samples
                    if len(data) <= 5000:
                        from scipy.stats import shapiro
                        sample = data[dist_col].dropna()
                        if len(sample) > 3:  # Shapiro-Wilk test requires at least 3 samples
                            try:
                                stat, p_value = shapiro(sample)
                                is_normal = "Yes" if p_value > 0.05 else "No"
                                st.metric("Normal Distribution", is_normal)
                            except:
                                st.metric("Normal Distribution", "Could not determine")
                        else:
                            st.metric("Normal Distribution", "Not enough data")
                    else:
                        st.metric("Normal Distribution", "Dataset too large to test")
        
        with tab3:
            st.markdown('<h3 class="subsection-header">Relationship Analysis</h3>', unsafe_allow_html=True)
            
            # Get numeric and categorical columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Select plot type
            plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Box Plot", "Pair Plot", "Joint Plot"])
            
            if plot_type == "Scatter Plot":
                # Select columns for scatter plot
                x_col = st.selectbox("Select X-axis column", numeric_cols)
                y_col = st.selectbox("Select Y-axis column", numeric_cols)
                
                # Optional: Add color by category
                if cat_cols:
                    use_color = st.checkbox("Color by category")
                    if use_color:
                        color_col = st.selectbox("Select category for color", cat_cols)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=data, x=x_col, y=y_col, hue=color_col, ax=ax)
                        ax.set_title(f'{x_col} vs {y_col} (colored by {color_col})')
                        st.pyplot(fig)
                    else:
                        plot_scatter(data, x_col, y_col)
                else:
                    plot_scatter(data, x_col, y_col)
                
                # Calculate correlation
                correlation = data[[x_col, y_col]].corr().iloc[0, 1]
                st.write(f"**Correlation coefficient:** {correlation:.3f}")
            
            elif plot_type == "Box Plot":
                # Select columns for box plot
                if cat_cols:
                    x_col = st.selectbox("Select category column (X-axis)", cat_cols)
                    y_col = st.selectbox("Select numeric column (Y-axis)", numeric_cols)
                    
                    # Check if too many categories
                    if data[x_col].nunique() > 15:
                        st.warning(f"Warning: {x_col} has more than 15 categories. Only top 15 will be displayed.")
                        # Get top 15 categories by frequency
                        top_cats = data[x_col].value_counts().head(15).index
                        filtered_data = data[data[x_col].isin(top_cats)]
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.boxplot(data=filtered_data, x=x_col, y=y_col, ax=ax)
                        ax.set_title(f'Box Plot of {y_col} by {x_col} (top 15 categories)')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        st.pyplot(fig)
                    else:
                        plot_boxplot(data, x_col, y_col)
                    
                    # Add ANOVA test if there are enough samples
                    if data[x_col].nunique() >= 2:
                        from scipy.stats import f_oneway
                        
                        # Perform ANOVA
                        groups = []
                        categories = []
                        for category, group in data.groupby(x_col)[y_col]:
                            if len(group) > 0:
                                groups.append(group)
                                categories.append(category)
                        
                        if len(groups) >= 2:
                            try:
                                f_stat, p_value = f_oneway(*groups)
                                st.write(f"**ANOVA F-statistic:** {f_stat:.3f}")
                                st.write(f"**ANOVA p-value:** {p_value:.5f}")
                                
                                if p_value < 0.05:
                                    st.write("**Result:** There is a statistically significant difference between groups.")
                                else:
                                    st.write("**Result:** There is no statistically significant difference between groups.")
                            except:
                                st.write("Could not perform ANOVA test.")
                else:
                    st.warning("No categorical columns found in the dataset for box plot analysis.")
            
            elif plot_type == "Pair Plot":
                # Select columns for pair plot
                if len(numeric_cols) > 1:
                    num_cols_to_include = st.slider("Number of numeric columns to include", 2, min(6, len(numeric_cols)), 3)
                    selected_numeric_cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:num_cols_to_include])
                    
                    # Optional: Add color by category
                    hue_col = None
                    if cat_cols:
                        use_hue = st.checkbox("Color by category")
                        if use_hue:
                            hue_col = st.selectbox("Select category for color", cat_cols)
                    
                    if len(selected_numeric_cols) >= 2:
                        with st.spinner("Generating pair plot..."):
                            fig = sns.pairplot(data[selected_numeric_cols + ([hue_col] if hue_col else [])], 
                                               hue=hue_col, height=2.5, aspect=1.2)
                            fig.fig.suptitle("Pair Plot of Selected Features", y=1.02)
                            st.pyplot(fig.fig)
                    else:
                        st.error("Please select at least 2 numeric columns for pair plot.")
                else:
                    st.warning("Not enough numeric columns for pair plot analysis.")
            
            elif plot_type == "Joint Plot":
                # Select columns for joint plot
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column", numeric_cols)
                    y_col = st.selectbox("Select Y-axis column (different from X)", 
                                        [col for col in numeric_cols if col != x_col])
                    
                    kind = st.selectbox("Select joint plot type", ["scatter", "kde", "hex", "reg"])
                    
                    with st.spinner("Generating joint plot..."):
                        fig = sns.jointplot(data=data, x=x_col, y=y_col, kind=kind, height=8)
                        st.pyplot(fig.fig)
                        
                        if kind == "reg":
                            st.write(f"**Pearson correlation:** {data[[x_col, y_col]].corr().iloc[0, 1]:.3f}")
                else:
                    st.warning("Not enough numeric columns for joint plot analysis.")
        
        with tab4:
            st.markdown('<h3 class="subsection-header">Custom Plot</h3>', unsafe_allow_html=True)
            
            plot_options = ["Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Box Plot", "Violin Plot", "Count Plot"]
            custom_plot = st.selectbox("Select plot type", plot_options)
            
            if custom_plot == "Bar Chart":
                # Bar chart options
                x_col = st.selectbox("Select X-axis column", data.columns)
                
                # Check if categorical or too many unique values
                if data[x_col].dtype in ['int64', 'float64'] and data[x_col].nunique() > 15:
                    use_bins = st.checkbox("Use bins for numeric data")
                    if use_bins:
                        num_bins = st.slider("Number of bins", 2, 20, 10)
                        data['binned'] = pd.cut(data[x_col], num_bins)
                        x_col = 'binned'
                
                # Aggregation options
                if data.select_dtypes(include=[np.number]).columns.tolist():
                    use_agg = st.checkbox("Aggregate a numeric column")
                    if use_agg:
                        y_col = st.selectbox("Select column to aggregate", 
                                            data.select_dtypes(include=[np.number]).columns)
                        agg_func = st.selectbox("Select aggregation function", 
                                                ["mean", "sum", "count", "min", "max"])
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        df_agg = data.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        sns.barplot(data=df_agg, x=x_col, y=y_col, ax=ax)
                        ax.set_title(f'Bar Chart of {y_col} ({agg_func}) by {x_col}')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        st.pyplot(fig)
                    else:
                        # Count plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        value_counts = data[x_col].value_counts().reset_index()
                        value_counts.columns = [x_col, 'count']
                        sns.barplot(data=value_counts, x=x_col, y='count', ax=ax)
                        ax.set_title(f'Count of {x_col}')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        st.pyplot(fig)
                else:
                    # Count plot for categorical data
                    fig, ax = plt.subplots(figsize=(12, 6))
                    value_counts = data[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'count']
                    sns.barplot(data=value_counts, x=x_col, y='count', ax=ax)
                    ax.set_title(f'Count of {x_col}')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig)
            
            elif custom_plot == "Line Chart":
                # Line chart options
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column", numeric_cols)
                    y_col = st.selectbox("Select Y-axis column", [col for col in numeric_cols if col != x_col])
                    
                    # Sort data by x_col
                    sorted_data = data.sort_values(by=x_col)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.plot(sorted_data[x_col], sorted_data[y_col])
                    ax.set_title(f'Line Chart of {y_col} vs {x_col}')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric columns for a line chart.")
            
            elif custom_plot == "Pie Chart":
                # Pie chart options
                cat_col = st.selectbox("Select column for pie chart", data.columns)
                
                # Limit to top N categories if too many
                max_categories = st.slider("Maximum number of categories to display", 2, 15, 8)
                value_counts = data[cat_col].value_counts()
                
                if len(value_counts) > max_categories:
                    # Keep top categories and group others
                    top_counts = value_counts.head(max_categories-1)
                    others_count = value_counts[max_categories-1:].sum()
                    
                    # Create new series with Others category
                    pie_data = pd.concat([top_counts, pd.Series({'Others': others_count})])
                else:
                    pie_data = value_counts
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', 
                       startangle=90, shadow=True)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax.set_title(f'Distribution of {cat_col}')
                st.pyplot(fig)
            
            elif custom_plot == "Histogram":
                # Histogram options
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    hist_col = st.selectbox("Select column for histogram", numeric_cols)
                    num_bins = st.slider("Number of bins", 5, 100, 20)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.histplot(data[hist_col], bins=num_bins, kde=True, ax=ax)
                    ax.set_title(f'Histogram of {hist_col}')
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns found for histogram.")
            
            elif custom_plot == "Box Plot":
                # Box plot options
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    y_col = st.selectbox("Select numeric column for box plot", numeric_cols)
                    
                    # Optionally group by category
                    cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                    if cat_cols:
                        use_grouping = st.checkbox("Group by category")
                        if use_grouping:
                            x_col = st.selectbox("Select category for grouping", cat_cols)
                            plot_boxplot(data, x_col, y_col)
                        else:
                            plot_boxplot(data, y_col)
                    else:
                        plot_boxplot(data, y_col)
                else:
                    st.warning("No numeric columns found for box plot.")
            
            elif custom_plot == "Violin Plot":
                # Violin plot options
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                
                if numeric_cols and cat_cols:
                    y_col = st.selectbox("Select numeric column for violin plot", numeric_cols)
                    x_col = st.selectbox("Select category for grouping", cat_cols)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.violinplot(data=data, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f'Violin Plot of {y_col} by {x_col}')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.warning("Need both numeric and categorical columns for violin plot.")
            
            elif custom_plot == "Count Plot":
                # Count plot options
                cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                if cat_cols:
                    x_col = st.selectbox("Select category for count plot", cat_cols)
                    
                    # Optional hue
                    use_hue = st.checkbox("Add second category (hue)")
                    if use_hue and len(cat_cols) > 1:
                        hue_col = st.selectbox("Select second category", [col for col in cat_cols if col != x_col])
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.countplot(data=data, x=x_col, hue=hue_col, ax=ax)
                        ax.set_title(f'Count Plot of {x_col} by {hue_col}')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.countplot(data=data, x=x_col, ax=ax)
                        ax.set_title(f'Count Plot of {x_col}')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        st.pyplot(fig)
                else:
                    st.warning("No categorical columns found for count plot.")
    else:
        st.warning("Please upload a dataset first in the 'Upload Data' page.")

# Page: Modeling and Prediction
elif page == "Modeling and Prediction":
    if st.session_state.data is not None:
        data = st.session_state.data
        st.markdown('<h2 class="section-header">Modeling and Prediction</h2>', unsafe_allow_html=True)
        
        # Data preparation
        if st.session_state.target_column is None:
            st.markdown('<h3 class="subsection-header">Data Preparation</h3>', unsafe_allow_html=True)
            
            # Select target column
            target_column = st.selectbox("Select target column (what you want to predict)", data.columns)
            
            # Determine problem type
            if data[target_column].dtype in ['int64', 'float64']:
                if data[target_column].nunique() <= 10:
                    problem_type_options = ["Classification", "Regression"]
                    problem_type = st.selectbox("Select problem type", problem_type_options)
                else:
                    problem_type = "Regression"
            else:
                problem_type = "Classification"
            
            # Save selections
            if st.button("Confirm Target and Problem Type"):
                st.session_state.target_column = target_column
                st.session_state.problem_type = problem_type
                st.success(f"Target column set to '{target_column}' for {problem_type.lower()}.")
                st.experimental_rerun()
        
        # Feature selection and preprocessing
        else:
            target_column = st.session_state.target_column
            problem_type = st.session_state.problem_type
            
            tab1, tab2, tab3 = st.tabs(["Feature Selection", "Model Training", "Predictions"])
            
            with tab1:
                st.markdown('<h3 class="subsection-header">Feature Selection and Preprocessing</h3>', unsafe_allow_html=True)
                
                # Reset option
                if st.button("Reset Target and Problem Type"):
                    st.session_state.target_column = None
                    st.session_state.problem_type = None
                    st.session_state.model = None
                    st.session_state.predictions = None
                    st.success("Target and problem type have been reset.")
                    st.experimental_rerun()
                
                # Display current settings
                st.write(f"**Target column:** {target_column}")
                st.write(f"**Problem type:** {problem_type}")
                
                # Feature selection
                st.markdown('<h4 class="subsection-header">Select Features</h4>', unsafe_allow_html=True)
                
                # Get available features (exclude target)
                available_features = [col for col in data.columns if col != target_column]
                
                # Select features to include
                selected_features = st.multiselect(
                    "Select features to include in the model",
                    available_features,
                    default=available_features
                )
                
                if not selected_features:
                    st.warning("Please select at least one feature for modeling.")
                else:
                    # Show feature information
                    feature_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Type': [data[col].dtype for col in selected_features],
                        'Missing Values': [data[col].isnull().sum() for col in selected_features],
                        'Missing %': [(data[col].isnull().sum() / len(data) * 100).round(2) for col in selected_features]
                    })
                    st.dataframe(feature_df)
                    
                    # Handle missing values
                    missing_cols = [col for col in selected_features if data[col].isnull().sum() > 0]
                    if missing_cols:
                        st.markdown('<h4 class="subsection-header">Handle Missing Values</h4>', unsafe_allow_html=True)
                        missing_handling = st.selectbox(
                            "Choose method to handle missing values",
                            ["Drop rows with any missing values", 
                             "Fill numeric with mean, categorical with mode", 
                             "Advanced: Specify by column"]
                        )
                    
                    # Feature preprocessing
                    st.markdown('<h4 class="subsection-header">Feature Preprocessing</h4>', unsafe_allow_html=True)
                    
                    # Identify numeric and categorical features
                    numeric_features = [col for col in selected_features if data[col].dtype in ['int64', 'float64']]
                    categorical_features = [col for col in selected_features if col not in numeric_features]
                    
                    # Preprocessing options
                    st.write("**Numeric Features:**")
                    scale_numeric = st.checkbox("Scale numeric features (recommended)", value=True)
                    
                    if categorical_features:
                        st.write("**Categorical Features:**")
                        cat_encoding = st.selectbox(
                            "Categorical encoding method",
                            ["One-Hot Encoding", "Label Encoding"]
                        )
                    
                    # Create preprocessed dataset for modeling
                    if st.button("Prepare Data for Modeling"):
                        # Create a copy to avoid modifying original data
                        model_data = data.copy()
                        
                        # Handle missing values
                        if missing_cols:
                            if missing_handling == "Drop rows with any missing values":
                                model_data = model_data.dropna(subset=selected_features)
                                st.write(f"Dropped {len(data) - len(model_data)} rows with missing values.")
                            
                            elif missing_handling == "Fill numeric with mean, categorical with mode":
                                for col in selected_features:
                                    if model_data[col].dtype in ['int64', 'float64']:
                                        model_data[col] = model_data[col].fillna(model_data[col].mean())
                                    else:
                                        model_data[col] = model_data[col].fillna(model_data[col].mode()[0])
                                st.write("Filled missing values with mean/mode as appropriate.")
                        
                        # Create feature matrix and target vector
                        X = model_data[selected_features].copy()
                        y = model_data[target_column].copy()
                        
                        # Preprocess categorical features
                        if categorical_features:
                            if cat_encoding == "One-Hot Encoding":
                                X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
                                st.write(f"Applied one-hot encoding to {len(categorical_features)} categorical features.")
                            else:  # Label Encoding
                                label_encoders = {}
                                for col in categorical_features:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                                    label_encoders[col] = le
                                st.write(f"Applied label encoding to {len(categorical_features)} categorical features.")
                        
                        # Scale numeric features
                        if scale_numeric and numeric_features:
                            scaler = StandardScaler()
                            X[numeric_features] = scaler.fit_transform(X[numeric_features])
                            st.write(f"Scaled {len(numeric_features)} numeric features.")
                        
                        # Split data into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Save to session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = X.columns.tolist()
                        
                        st.success(f"Data prepared for modeling: {X.shape[0]} samples, {X.shape[1]} features")
                        st.write(f"Training set: {X_train.shape[0]} samples")
                        st.write(f"Test set: {X_test.shape[0]} samples")
            
            with tab2:
                st.markdown('<h3 class="subsection-header">Model Training</h3>', unsafe_allow_html=True)
                
                if 'X_train' not in st.session_state:
                    st.warning("Please prepare your data in the 'Feature Selection' tab first.")
                else:
                    # Model selection
                    if problem_type == "Classification":
                        model_options = [
                            "Random Forest",
                            "Logistic Regression",
                            "Support Vector Machine",
                            "K-Nearest Neighbors",
                            "Decision Tree"
                        ]
                    else:  # Regression
                        model_options = [
                            "Random Forest",
                            "Linear Regression",
                            "Support Vector Regression",
                            "K-Nearest Neighbors",
                            "Decision Tree"
                        ]
                    
                    selected_model = st.selectbox("Select a model", model_options)
                    
                    # Model hyperparameters
                    st.markdown('<h4 class="subsection-header">Model Hyperparameters</h4>', unsafe_allow_html=True)
                    
                    if selected_model == "Random Forest":
                        n_estimators = st.slider("Number of trees", 10, 500, 100)
                        max_depth = st.slider("Maximum depth of trees", 1, 50, 10)
                        min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
                        
                        if st.button("Train Model"):
                            with st.spinner("Training Random Forest model..."):
                                if problem_type == "Classification":
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        random_state=42
                                    )
                                else:
                                    model = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        random_state=42
                                    )
                                
                                model.fit(st.session_state.X_train, st.session_state.y_train)
                                st.session_state.model = model
                                st.success("Model trained successfully!")
                    
                    elif selected_model == "Logistic Regression" or selected_model == "Linear Regression":
                        from sklearn.linear_model import LogisticRegression, LinearRegression
                        
                        if selected_model == "Logistic Regression":
                            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                            max_iter = st.slider("Maximum iterations", 100, 2000, 1000)
                            
                            if st.button("Train Model"):
                                with st.spinner("Training Logistic Regression model..."):
                                    model = LogisticRegression(
                                        C=C,
                                        max_iter=max_iter,
                                        random_state=42
                                    )
                                    model.fit(st.session_state.X_train, st.session_state.y_train)
                                    st.session_state.model = model
                                    st.success("Model trained successfully!")
                        else:
                            if st.button("Train Model"):
                                with st.spinner("Training Linear Regression model..."):
                                    model = LinearRegression()
                                    model.fit(st.session_state.X_train, st.session_state.y_train)
                                    st.session_state.model = model
                                    st.success("Model trained successfully!")
                    
                    elif selected_model == "Support Vector Machine" or selected_model == "Support Vector Regression":
                        from sklearn.svm import SVC, SVR
                        
                        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                        
                        if st.button("Train Model"):
                            with st.spinner(f"Training Support Vector {'Classification' if problem_type == 'Classification' else 'Regression'} model..."):
                                if problem_type == "Classification":
                                    model = SVC(
                                        kernel=kernel,
                                        C=C,
                                        probability=True,
                                        random_state=42
                                    )
                                else:
                                    model = SVR(
                                        kernel=kernel,
                                        C=C
                                    )
                                
                                model.fit(st.session_state.X_train, st.session_state.y_train)
                                st.session_state.model = model
                                st.success("Model trained successfully!")
                    
                    elif selected_model == "K-Nearest Neighbors":
                        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                        
                        n_neighbors = st.slider("Number of neighbors", 1, 20, 5)
                        weights = st.selectbox("Weight function", ["uniform", "distance"])
                        
                        if st.button("Train Model"):
                            with st.spinner(f"Training K-Nearest Neighbors {'Classification' if problem_type == 'Classification' else 'Regression'} model..."):
                                if problem_type == "Classification":
                                    model = KNeighborsClassifier(
                                        n_neighbors=n_neighbors,
                                        weights=weights
                                    )
                                else:
                                    model = KNeighborsRegressor(
                                        n_neighbors=n_neighbors,
                                        weights=weights
                                    )
                                
                                model.fit(st.session_state.X_train, st.session_state.y_train)
                                st.session_state.model = model
                                st.success("Model trained successfully!")
                    
                    elif selected_model == "Decision Tree":
                        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                        
                        max_depth = st.slider("Maximum depth of tree", 1, 50, 10)
                        min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
                        min_samples_leaf = st.slider("Minimum samples in leaf", 1, 20, 1)
                        
                        if st.button("Train Model"):
                            with st.spinner(f"Training Decision Tree {'Classification' if problem_type == 'Classification' else 'Regression'} model..."):
                                if problem_type == "Classification":
                                    model = DecisionTreeClassifier(
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        random_state=42
                                    )
                                else:
                                    model = DecisionTreeRegressor(
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        random_state=42
                                    )
                                
                                model.fit(st.session_state.X_train, st.session_state.y_train)
                                st.session_state.model = model
                                st.success("Model trained successfully!")
                    
                    # Model evaluation
                    if st.session_state.model is not None:
                        st.markdown('<h4 class="subsection-header">Model Evaluation</h4>', unsafe_allow_html=True)
                        
                        model = st.session_state.model
                        X_train = st.session_state.X_train
                        X_test = st.session_state.X_test
                        y_train = st.session_state.y_train
                        y_test = st.session_state.y_test
                        
                        # Make predictions
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        # Save predictions
                        st.session_state.predictions = y_test_pred
                        
                        # Evaluate the model
                        if problem_type == "Classification":
                            # Classification metrics
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                            
                            # Training metrics
                            train_accuracy = accuracy_score(y_train, y_train_pred)
                            
                            # Test metrics
                            test_accuracy = accuracy_score(y_test, y_test_pred)
                            
                            # Calculate more detailed metrics if binary or multiclass
                            unique_classes = len(np.unique(y_test))
                            
                            if unique_classes == 2:  # Binary classification
                                test_precision = precision_score(y_test, y_test_pred, average='binary')
                                test_recall = recall_score(y_test, y_test_pred, average='binary')
                                test_f1 = f1_score(y_test, y_test_pred, average='binary')
                            else:  # Multi-class classification
                                test_precision = precision_score(y_test, y_test_pred, average='weighted')
                                test_recall = recall_score(y_test, y_test_pred, average='weighted')
                                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                            
                            # Display metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Accuracy", f"{train_accuracy:.4f}")
                            with col2:
                                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Precision", f"{test_precision:.4f}")
                            with col2:
                                st.metric("Recall", f"{test_recall:.4f}")
                            with col3:
                                st.metric("F1 Score", f"{test_f1:.4f}")
                            
                            # Confusion Matrix
                            st.write("**Confusion Matrix:**")
                            cm = confusion_matrix(y_test, y_test_pred)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                            
                            # Classification report
                            st.write("**Classification Report:**")
                            report = classification_report(y_test, y_test_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Feature importance if available
                            if hasattr(model, 'feature_importances_'):
                                st.write("**Feature Importance:**")
                                feature_imp = pd.DataFrame({
                                    'Feature': st.session_state.feature_names,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.barplot(data=feature_imp, x='Importance', y='Feature', ax=ax)
                                ax.set_title('Feature Importance')
                                st.pyplot(fig)
                            
                        else:  # Regression metrics
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            # Training metrics
                            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                            train_r2 = r2_score(y_train, y_train_pred)
                            
                            # Test metrics
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                            test_mae = mean_absolute_error(y_test, y_test_pred)
                            test_r2 = r2_score(y_test, y_test_pred)
                            
                            # Display metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training RMSE", f"{train_rmse:.4f}")
                                st.metric("Training R²", f"{train_r2:.4f}")
                            with col2:
                                st.metric("Test RMSE", f"{test_rmse:.4f}")
                                st.metric("Test MAE", f"{test_mae:.4f}")
                                st.metric("Test R²", f"{test_r2:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.scatter(y_test, y_test_pred, alpha=0.5)
                            
                            # Add perfect prediction line
                            min_val = min(y_test.min(), y_test_pred.min())
                            max_val = max(y_test.max(), y_test_pred.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                            
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted Values')
                            st.pyplot(fig)
                            
                            # Residual plot
                            residuals = y_test - y_test_pred
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.scatter(y_test_pred, residuals, alpha=0.5)
                            ax.axhline(y=0, color='r', linestyle='--')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Residuals')
                            ax.set_title('Residual Plot')
                            st.pyplot(fig)
                            
                            # Residual distribution
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.histplot(residuals, kde=True, ax=ax)
                            ax.set_xlabel('Residuals')
                            ax.set_title('Residual Distribution')
                            st.pyplot(fig)
                            
                            # Feature importance if available
                            if hasattr(model, 'feature_importances_'):
                                st.write("**Feature Importance:**")
                                feature_imp = pd.DataFrame({
                                    'Feature': st.session_state.feature_names,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.barplot(data=feature_imp, x='Importance', y='Feature', ax=ax)
                                ax.set_title('Feature Importance')
                                st.pyplot(fig)
            
            with tab3:
                st.markdown('<h3 class="subsection-header">Predictions</h3>', unsafe_allow_html=True)
                
                if st.session_state.model is None:
                    st.warning("Please train a model in the 'Model Training' tab first.")
                else:
                    # Options for prediction
                    prediction_option = st.selectbox(
                        "Prediction options",
                        ["Make predictions on test set", "Make predictions on new data", "Make predictions on custom input"]
                    )
                    
                    if prediction_option == "Make predictions on test set":
                        if st.session_state.predictions is not None:
                            # Display predictions on test set
                            X_test = st.session_state.X_test
                            y_test = st.session_state.y_test
                            predictions = st.session_state.predictions
                            
                            # Create a dataframe with actual and predicted values
                            results_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': predictions,
                                'Difference': np.abs(y_test - predictions)
                            }).reset_index(drop=True)
                            
                            st.write("**Predictions on Test Set:**")
                            st.dataframe(results_df)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Display sample predictions
                            st.write("**Sample Predictions Visualization:**")
                            
                            sample_size = min(100, len(predictions))
                            sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
                            
                            if problem_type == "Regression":
                                # For regression, show actual vs predicted plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(np.arange(len(sample_indices)), y_test.iloc[sample_indices], label='Actual', alpha=0.7)
                                ax.scatter(np.arange(len(sample_indices)), predictions[sample_indices], label='Predicted', alpha=0.7)
                                ax.set_xlabel('Sample')
                                ax.set_ylabel('Value')
                                ax.set_title('Actual vs Predicted Values (Sample)')
                                ax.legend()
                                st.pyplot(fig)
                            else:
                                # For classification, show prediction distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                prediction_counts = pd.Series(predictions).value_counts().sort_index()
                                actual_counts = pd.Series(y_test).value_counts().sort_index()
                                
                                # Combine into a dataframe
                                counts_df = pd.DataFrame({
                                    'Actual': actual_counts,
                                    'Predicted': prediction_counts
                                }).fillna(0)
                                
                                counts_df.plot(kind='bar', ax=ax)
                                ax.set_xlabel('Class')
                                ax.set_ylabel('Count')
                                ax.set_title('Distribution of Actual vs Predicted Classes')
                                ax.legend()
                                st.pyplot(fig)
                    
                    elif prediction_option == "Make predictions on new data":
                        st.write("**Upload new data for prediction:**")
                        
                        # File upload for new data
                        new_data_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
                        
                        if new_data_file is not None:
                            try:
                                # Load new data
                                if new_data_file.name.endswith('.csv'):
                                    new_data = pd.read_csv(new_data_file)
                                else:
                                    new_data = pd.read_excel(new_data_file)
                                
                                st.success(f"New data loaded successfully with {new_data.shape[0]} rows and {new_data.shape[1]} columns.")
                                
                                # Show a sample of the data
                                st.dataframe(new_data.head())
                                
                                # Check if all required features are present
                                missing_features = [feat for feat in st.session_state.feature_names if feat not in new_data.columns]
                                
                                if missing_features:
                                    st.error(f"The following required features are missing in the new data: {', '.join(missing_features)}")
                                else:
                                    # Preprocess the new data
                                    X_new = new_data[st.session_state.feature_names]
                                    
                                    # Make predictions
                                    if st.button("Generate Predictions"):
                                        with st.spinner("Generating predictions..."):
                                            new_predictions = st.session_state.model.predict(X_new)
                                            
                                            # Create a dataframe with predictions
                                            results_df = pd.DataFrame({
                                                'Prediction': new_predictions
                                            }).reset_index(drop=True)
                                            
                                            # Add original data columns if user wants
                                            include_original = st.checkbox("Include original data columns in results")
                                            if include_original:
                                                for col in new_data.columns:
                                                    results_df[col] = new_data[col].values
                                            
                                            st.write("**Predictions on New Data:**")
                                            st.dataframe(results_df)
                                            
                                            # Download option
                                            csv = results_df.to_csv(index=False)
                                            b64 = base64.b64encode(csv.encode()).decode()
                                            href = f'<a href="data:file/csv;base64,{b64}" download="new_predictions.csv">Download predictions as CSV</a>'
                                            st.markdown(href, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error processing the new data: {e}")
                    
                    elif prediction_option == "Make predictions on custom input":
                        st.write("**Enter custom values for prediction:**")
                        
                        # Create input fields for each feature
                        custom_input = {}
                        for feature in st.session_state.feature_names:
                            # Check the original data type from the training data
                            orig_col = feature
                            # For one-hot encoded features, extract the original column name
                            if '_' in feature and feature not in st.session_state.X_train.columns:
                                orig_col = feature.split('_')[0]
                            
                            if orig_col in st.session_state.X_train.columns and st.session_state.X_train[orig_col].dtype in ['int64', 'float64']:
                                # Numeric input
                                custom_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
                            else:
                                # Categorical input
                                custom_input[feature] = st.text_input(f"Enter value for {feature}", value="")
                        
                        if st.button("Generate Prediction"):
                            # Convert custom input to dataframe
                            input_df = pd.DataFrame([custom_input])
                            
                            # Make prediction
                            try:
                                prediction = st.session_state.model.predict(input_df)[0]
                                
                                # Display prediction
                                if problem_type == "Classification":
                                    st.success(f"Predicted class: {prediction}")
                                    
                                    # If model supports probability prediction
                                    if hasattr(st.session_state.model, 'predict_proba'):
                                        proba = st.session_state.model.predict_proba(input_df)[0]
                                        
                                        # Display probabilities
                                        proba_df = pd.DataFrame({
                                            'Class': st.session_state.model.classes_,
                                            'Probability': proba
                                        }).sort_values('Probability', ascending=False)
                                        
                                        st.write("**Class Probabilities:**")
                                        st.dataframe(proba_df)
                                        
                                        # Visualize probabilities
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(data=proba_df, x='Class', y='Probability', ax=ax)
                                        ax.set_title('Prediction Probabilities by Class')
                                        ax.set_ylim(0, 1)
                                        st.pyplot(fig)
                                else:
                                    st.success(f"Predicted value: {prediction:.4f}")
                            except Exception as e:
                                st.error(f"Error generating prediction: {e}")
    else:
        st.warning("Please upload a dataset first in the 'Upload Data' page.")
