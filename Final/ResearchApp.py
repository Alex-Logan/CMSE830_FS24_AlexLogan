import os
import streamlit as st
import toml
import pandas as pd  # Ensure pandas is imported for data manipulation
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px  # Import Plotly Express for visualization
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # needed for MICE
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error  # Import mean_squared_error for evaluation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from scipy.stats import ttest_ind, chi2_contingency
import plotly.graph_objects as go
from collections import Counter

# Set the title of the Streamlit app
st.title("Anti-Cancer Peptides: A Medical Hopeful?")
st.markdown("### A data research application by Alex Logan")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Introduction", "Data Loading", "Handling the Data", "Modeling", "Main Insights", "References"])
# Sidebar - About This Section
st.sidebar.header("About This:")
st.sidebar.markdown("""
This dashboard is a condensed version of a documented dataset research application that can be viewed in full, all steps included, at the following link: [GitHub application Link](https://github.com/Alex-Logan/CMSE830_FS24_AlexLogan/tree/main/Final)

Please see the "Introduction" section for more details.
""")
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config.toml file
config_path = os.path.join(current_dir, '.streamlit', 'config.toml')

# Load the theme configuration from the config.toml file
try:
    config = toml.load(config_path)
    current_theme = config.get('theme', {}).get('base', 'light')  # Default to 'light' if not found
except (FileNotFoundError, toml.TomlDecodeError) as e:
    st.warning("Could not load config.toml, defaulting to light theme.")
    current_theme = 'light'  # Default to light if there's an error

# Display Mode Section
st.sidebar.header("Display Mode")
mode_options = ["Light", "Dark", "Red", "Gold", "Blue", "Green", "Purple", "Orange", "Pink"]

# Use session state to store the theme and update it dynamically
if 'theme' not in st.session_state:
    st.session_state.theme = current_theme

selected_mode = st.sidebar.selectbox("Choose theme:", mode_options, index=0 if st.session_state.theme == 'light' else 1)

# Apply the selected theme instantly
if selected_mode == "Dark":
    st.session_state.theme = 'dark'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;  /* Dark background */
            color: gray;  /* Gray text */
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;  /* Dark sidebar */
            color: gray;  /* Gray sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Red":
    st.session_state.theme = 'red'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffcccc;  /* Light red background */
            color: #990000;  /* Dark red text */
        }
        .sidebar .sidebar-content {
            background-color: #ff9999;  /* Red sidebar */
            color: #990000;  /* Dark red sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Gold":
    st.session_state.theme = 'gold'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #fff8dc;  /* Light gold background */
            color: #b8860b;  /* Dark gold text */
        }
        .sidebar .sidebar-content {
            background-color: #ffe135;  /* Gold sidebar */
            color: #b8860b;  /* Dark gold sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Blue":
    st.session_state.theme = 'blue'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #add8e6;  /* Light blue background */
            color: #00008b;  /* Dark blue text */
        }
        .sidebar .sidebar-content {
            background-color: #87cefa;  /* Blue sidebar */
            color: #00008b;  /* Dark blue sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Green":
    st.session_state.theme = 'green'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #d9f9d9;  /* Light green background */
            color: #006400;  /* Dark green text */
        }
        .sidebar .sidebar-content {
            background-color: #98fb98;  /* Green sidebar */
            color: #006400;  /* Dark green sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Purple":
    st.session_state.theme = 'purple'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #e6e6fa;  /* Lavender background */
            color: #4b0082;  /* Dark purple text */
        }
        .sidebar .sidebar-content {
            background-color: #d8bfd8;  /* Thistle sidebar */
            color: #4b0082;  /* Dark purple sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Orange":
    st.session_state.theme = 'orange'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffebcc;  /* Light orange background */
            color: #ff4500;  /* Dark orange text */
        }
        .sidebar .sidebar-content {
            background-color: #ffcc99;  /* Orange sidebar */
            color: #ff4500;  /* Dark orange sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif selected_mode == "Pink":
    st.session_state.theme = 'pink'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffe4e1;  /* Light pink background */
            color: #ff1493;  /* Dark pink text */
        }
        .sidebar .sidebar-content {
            background-color: #ffb6c1;  /* Pink sidebar */
            color: #ff1493;  /* Dark pink sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:  # Default to Light mode
    st.session_state.theme = 'light'
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;  /* Light background */
            color: black;  /* Dark text */
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;  /* Light sidebar */
            color: black;  /* Dark sidebar text */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display a message to indicate the current theme
st.write(f"Current theme: {st.session_state.theme.capitalize()}")

# Load datasets for later use
try:
    breastcancer = pd.read_csv('breastcancer.csv')
    peptides_b = pd.read_csv('peptides_b.csv')
    peptides_l = pd.read_csv('peptides_l.csv')
    lungcancer = pd.read_csv('lungcancer.csv')
    lungcancer_improved = pd.read_csv('improved_lungcancer_dataset.csv')
    peptides_l_improved = pd.read_csv('improved_peptides_dataset.csv')
    lungdatamerged = pd.read_csv('cleaned_integrated_lungcancer_peptides_advanced.csv')
except FileNotFoundError:
    # If the first attempt fails, try loading from the GitHub path
    try:
        breastcancer = pd.read_csv('Final/Raw_Data/breastcancer.csv')
        peptides_b = pd.read_csv('Final/Raw_Data/peptides_b.csv')
        peptides_l = pd.read_csv('Final/Raw_Data/peptides_l.csv')
        lungcancer = pd.read_csv('Final/Raw_Data/lungcancer.csv')
        lungcancer_improved = pd.read_csv('Final/Raw_Data/improved_lungcancer_dataset.csv')
        peptides_l_improved = pd.read_csv('Final/Raw_Data/improved_peptides_dataset.csv')
        lungdatamerged = pd.read_csv('Final/Raw_Data/cleaned_integrated_lungcancer_peptides_advanced.csv')
    except FileNotFoundError:
        # If both attempts fail, display an error message
        st.error("Error loading datasets: Make sure the files are in the correct directory.")
        st.stop()


# Create a copy of the peptides dataset
peptides_copy = peptides_b.copy()

# Re-encode the peptides_copy DataFrame if the class columns are missing
expected_columns = ['class_inactive - virtual', 'class_mod. active', 'class_very active']
if not all(col in peptides_copy.columns for col in expected_columns):
    # One-hot encoding for the class column if it doesn't exist
    peptides_copy = pd.get_dummies(peptides_copy, columns=['class'], drop_first=True)
    
# Define global variables for active peptide classes
inactive_peptides = peptides_copy[peptides_copy['class_inactive - virtual'] == 1]
mod_active_peptides = peptides_copy[peptides_copy['class_mod. active'] == 1]
very_active_peptides = peptides_copy[peptides_copy['class_very active'] == 1]

# Introduction page
if page == "Introduction":
    st.header("Introduction")
    st.markdown("""
    In general terms, the purpose of this application is to determine how anti-cancer peptides may relate to fighting the common causes of cancer. However, there are many types of cancer, and so this application's scope is specifically focused on breast and lung cancer. My datasets come from three sources:

    1. **Breast Cancer Dataset**
       - **Title:** Breast Cancer Wisconsin (Diagnostic) Data Set
       - **Source:** UCI Machine Learning Repository
       - **Data Format:** CSV
       - **Description:** This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes measurements like radius, texture, and smoothness, which are used to classify tumors as benign or malignant.
       - **Link:** [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

    2. **Anticancer Peptides Dataset(s)**
       - **Title:** Anticancer Peptides Dataset
       - **Source:** UCI Machine Learning Repository
       - **Data Format:** CSV
       - **Description:** This dataset includes a collection of peptides and their anticancer activity against various types of cancer, providing valuable information for research in medicinal chemistry.
       - **Link:** [Anticancer Peptides Data](https://www.kaggle.com/datasets/uciml/anticancer-peptides-dataset)

    3. **Lung Cancer Dataset**
       - **Title:** Lung Cancer Prediction Data Set
       - **Source:** Kaggle
       - **Data Format:** CSV
       - **Description:** This dataset includes features relevant to lung cancer prediction, such as smoking, alcohol consumption, and age, along with the target label indicating the presence or absence of lung cancer. It aims to support research in lung cancer detection and prevention.
       - **Link:** [Lung Cancer Prediction Data](https://www.kaggle.com/datasets/rashadrmammadov/lung-cancer-prediction)

    The work involved in the application is two-fold: to examine the most common effects leading to breast and lung cancer and also to examine the efficacy of peptides, with the ultimate goal being to find the most effective anti-cancer solutions. More specifically, what might make a peptide effective in the first place, and what would a "good" anticancer peptide do to the cells in order to be effective? And are the any other non-immediately related conclusions we can draw after doing this research?

    Bringing these goals together into a cohesive sum requires a bit of thinking, and we'll get to that soon. For now, there is some work to do to make this data more usable.

    While the information presented here is primarily intended for usage by those in the medical community, I hope for the application to be understandable by all - I have tried to keep the language I use casual.

NOTE: If the slider to scroll a page won't appear, try using the arrow keys until it does.
    """)

# Data Loading and Preparation page
elif page == "Data Loading":
    st.header("Data Loading and Preparation")
    
        # Display heads of both datasets
    st.subheader("Preview of Datasets")
    st.write("### Breast Cancer Dataset")
    st.write(breastcancer.head())
    st.write("### Peptides Dataset (Breast)")
    st.write(peptides_b.head())
    st.write("### Lung Cancer Dataset")
    st.write(lungcancer.head())
    st.write("### Peptides Dataset (Lung)")
    st.write(peptides_l.head())

    # =========================================================
    # COMPREHENSIVE STATISTICAL ANALYSIS
    # =========================================================

    # Summary Statistics
    def summary_statistics(df, name):
        st.subheader(f"Summary Statistics for {name}")
        st.write(df.describe(include='all'))

    # Correlation Analysis
    def correlation_analysis(df, target_col, name):
        st.subheader(f"Correlation Analysis for {name}")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if target_col not in numeric_df.columns:
            st.warning(f"Target column '{target_col}' is not numeric or not found in the dataset.")
            return
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        st.write(correlations)

    
    def ttest_features(df, target_col):
        st.subheader(f"T-Tests for Significant Features in {target_col}")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if target_col not in numeric_df.columns:
            st.warning(f"Target column '{target_col}' is not numeric or not found in the dataset.")
            return []
        significant_features = []
        for col in numeric_df.columns:
            if col != target_col:
                group1 = df[df[target_col] == 0][col]
                group2 = df[df[target_col] == 1][col]
                t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
                if p_val < 0.05:
                    significant_features.append((col, p_val))
                    st.write(f"{col}: p-value = {p_val}")
        st.write("Significant Features:", [x[0] for x in significant_features])
        return significant_features

    
    # Chi-Square Test for Peptides Class Distributions
    def chi_square_test(df, col, name):
        st.subheader(f"Chi-Square Test for {name} ({col})")
        if col in df.columns:
            observed = df[col].value_counts()
            chi2, p, dof, expected = chi2_contingency(pd.DataFrame([observed, observed]).transpose())
            st.write(f"Chi-Square Test Results: chi2 = {chi2}, p = {p}")
        else:
            st.warning(f"Column '{col}' not found in the dataset.")

    # Perform Summary Statistics
    summary_statistics(lungcancer, "Lung Cancer Dataset")
    summary_statistics(peptides_l, "Peptides Lung Dataset")
    summary_statistics(breastcancer, "Breast Cancer Dataset")
    summary_statistics(peptides_b, "Peptides Breast Dataset")

    # Perform Correlation Analysis
    if 'Stage' in lungcancer.columns:
        stage_mapping = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
        lungcancer['Stage_Numeric'] = lungcancer['Stage'].map(stage_mapping)

    if breastcancer['diagnosis'].dtype == 'object':
        breastcancer['diagnosis'] = breastcancer['diagnosis'].map({'B': 0, 'M': 1})

    correlation_analysis(breastcancer, 'diagnosis', "Breast Cancer Dataset")
    if 'Stage_Numeric' in lungcancer.columns:
        correlation_analysis(lungcancer, 'Stage_Numeric', "Lung Cancer Dataset")

    # Perform T-Tests
    significant_breast = ttest_features(breastcancer, 'diagnosis')
    if 'Stage_Numeric' in lungcancer.columns:
        significant_lung = ttest_features(lungcancer, 'Stage_Numeric')

    # Perform Chi-Square Test
    chi_square_test(peptides_b, 'class', "Peptides Breast Dataset")


    st.markdown("""
    For the purposes of our analysis, one of the most important parts of the peptides dataset is the class column, which categorizes peptides into four classes based on their activity levels:

    - **Very Active:** EC/IC/LD/LC50 ≤ 5 μM
    - **Moderately Active:** EC/IC/LD/LC50 ≤ 50 μM
    - **Experimental Inactive**
    - **Virtual Inactive**

    For the inactive peptides, the only thing we care about at the moment is that they are **Inactive**.

    Active peptides are those that are either Very Active or Moderately Active, as they show significant anticancer activity. This will be important soon!
    
    The peptides datasets are structured like this for both the breast and lung cancer related peptide files.
    
    # Breast Cancer Data IDA:

    Originally, this next part of the dataset observed **all** the columns of the breast cancer dataset; however, **the results weren't particularly meaningful**, so for now it's only being shown for the **diagnosis** column.

    ### Checking for Missing Values

    For the breast cancer dataset: are there missing values? Here's a check:
    """)
    
    # Check for missing values in breast cancer dataset
    missing_breastcancer = breastcancer.isnull().sum()
    missing_breastcancer = missing_breastcancer[missing_breastcancer > 0]  # show a column only if there's missing data
    st.write("Columns with missing values in breast cancer dataset:")
    st.write(missing_breastcancer)

    # Check for missing values in peptides dataset
    missing_peptides_b = peptides_b.isnull().sum()
    missing_peptides_b = missing_peptides_b[missing_peptides_b > 0] 
    st.write("Columns with missing values in peptides_b dataset:")
    st.write(missing_peptides_b)

    st.markdown("""
    ### Checking for Duplicates

    Now, we'll check for duplicates in the datasets:
    """)
    
    # Check for duplicates
    duplicates_breastcancer = breastcancer.duplicated().sum()
    duplicates_peptides_b = peptides_b.duplicated().sum()

    # Create a DataFrame to display duplicates in a table format
    duplicates_data = {
        'Dataset': ['Breast Cancer Dataset', 'Peptides Dataset'],
        'Number of Duplicate Rows': [duplicates_breastcancer, duplicates_peptides_b]
    }
    duplicates_df = pd.DataFrame(duplicates_data)
    
    st.table(duplicates_df)

    st.markdown("""
    ### Outlier Detection

    Next, we'll identify any outliers in the datasets using z-scores. Here's how many outliers we found:
    """)
    
    def detect_outliers_zscore(df, threshold=3):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        z_scores = stats.zscore(df[numerical_cols])  # obtaining the z-scores
        abs_z_scores = np.abs(z_scores)  # absolute Z-scores
        outliers = (abs_z_scores > threshold)  # identify outliers where z-score > threshold
        return outliers.sum(axis=0)  # return amount of outliers in each column

    z_threshold = 3
    outliers_breastcancer = detect_outliers_zscore(breastcancer, threshold=z_threshold)
    st.write(f"Outliers in breast cancer dataset (Z-score > {z_threshold}):")
    st.write(outliers_breastcancer)

    outliers_peptides_b = detect_outliers_zscore(peptides_b, threshold=z_threshold)
    st.write(f"\nOutliers in peptides_b dataset (Z-score > {z_threshold}):")
    st.write(outliers_peptides_b)

    st.markdown("""
    ### Class Imbalance Check

    Let's check for class imbalances in the datasets.
    """)
    
    def check_class_imbalance(df, column):
        class_counts = df[column].value_counts()
        return class_counts

    breast_cancer_distribution = check_class_imbalance(breastcancer, 'diagnosis')
    st.write("Breast cancer dataset diagnosis distribution:")
    st.write(breast_cancer_distribution)

    peptides_distribution = check_class_imbalance(peptides_b, 'class')
    st.write("Peptides_b dataset class imbalance:")
    st.write(peptides_distribution)

    st.markdown("""
    As we can see, the datasets are pretty clean! There's a strange "Unnamed 32" column in the breast cancer dataset that we'll be getting rid of, and a few other tasks at hand. We can see there's some class imbalance in the peptides dataset, and we might still want to consider some imputation. It's also clear some encoding will be necessary, which we'll get to in a moment, along with the rest of the data handling that needs to be done.

    The distribution between benign and malignant tumors doesn't seem too bad to me; it feels realistic. The peptides dataset is another story, which unfolded quite a bit over the course of this analysis.

    There's also the matter of making sure that these datasets can be related to each other in a meaningful way. This is why I dropped the lung cancer peptides dataset. The two datasets we are working with now are about the same type of cancer, so that's a start.

    For the time being, the method in which the datasets are merged together may sound unsatisfying, but let me explain the logic. We will combine the datasets together based on the type of cancer being analyzed - it is reasonable to assume that examining peptide efficacy against breast cancer alongside likely causes of breast cancer could suggest that certain peptides may be capable of mitigating the effects of common causes (though it is important to remember this is a hypothesis).

     # Lung Cancer Data IDA:
    """)

    # Step 1: Display Numerical Columns and Handle Missing Values with KNNImputer
    st.subheader("Numerical Columns in Lung Cancer Dataset")
    numerical_columns = lungcancer.select_dtypes(include=['float64', 'int64']).columns
    st.write(numerical_columns)

    imputer = KNNImputer(n_neighbors=5)
    lungcancer[numerical_columns] = pd.DataFrame(
        imputer.fit_transform(lungcancer[numerical_columns]),
        columns=numerical_columns,
        index=lungcancer.index
    )

    st.subheader("Lung Cancer Dataset After KNN Imputation")
    st.write(lungcancer.head())

    # Step 2: Encode Categorical Variables
    st.subheader("Encoding Peptide Classes")
    label_encoder = LabelEncoder()
    peptides_l['class_encoded'] = label_encoder.fit_transform(peptides_l['class'])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_classes = pd.DataFrame(
        encoder.fit_transform(peptides_l[['class']]),
        columns=encoder.get_feature_names_out(['class']),
        index=peptides_l.index
    )
    peptides_l = pd.concat([peptides_l, encoded_classes], axis=1)

    st.write("Peptides Dataset After Encoding")
    st.write(peptides_l.head())

    lungcancer['cancer_type'] = 'lung'
    peptides_l['cancer_type'] = 'peptides_lung'

    # Align Columns Between DataFrames
    lungcancer_columns = set(lungcancer.columns)
    peptides_columns = set(peptides_l.columns)
    for col in lungcancer_columns - peptides_columns:
        peptides_l[col] = np.nan
    for col in peptides_columns - lungcancer_columns:
        lungcancer[col] = np.nan

    peptides_l = peptides_l[lungcancer.columns]

    lungcancer.reset_index(drop=True, inplace=True)
    peptides_l.reset_index(drop=True, inplace=True)
    integrated_data_advanced = pd.concat([lungcancer, peptides_l], axis=0, ignore_index=True)

    st.subheader("Integrated Dataset (Lung Cancer and Peptides)")
    st.write(integrated_data_advanced.head())

    # Step 4: Feature Selection Using SelectKBest
    target_column = 'Stage'
    if target_column in lungcancer.columns:
        selector = SelectKBest(score_func=f_classif, k=5)
        selected_features = selector.fit_transform(lungcancer[numerical_columns], lungcancer[target_column])
        selected_feature_names = numerical_columns[selector.get_support()]
        lungcancer_selected = lungcancer[selected_feature_names]
        st.subheader("Selected Features for Lung Cancer Dataset")
        st.write(selected_feature_names)
    else:
        st.warning(f"Target column '{target_column}' not found in lungcancer dataset.")
        lungcancer_selected = lungcancer[numerical_columns]

    # Step 5: Dimensionality Reduction Using PCA
    pca = PCA(n_components=2, random_state=42)
    lungcancer_pca = pca.fit_transform(lungcancer_selected)
    lungcancer['PCA1'], lungcancer['PCA2'] = lungcancer_pca[:, 0], lungcancer_pca[:, 1]

    st.subheader("Lung Cancer Dataset After PCA")
    st.write(lungcancer[['PCA1', 'PCA2']].head())

    # Step 7: Normalize Integrated Data
    scaler = StandardScaler()
    numerical_columns_integrated = integrated_data_advanced.select_dtypes(include=['float64', 'int64']).columns
    integrated_data_advanced[numerical_columns_integrated] = scaler.fit_transform(
        integrated_data_advanced[numerical_columns_integrated]
    )

    st.subheader("Integrated Dataset After Normalization")
    st.write(integrated_data_advanced[numerical_columns_integrated].head())

    # Save Cleaned Data
    integrated_data_advanced.to_csv('cleaned_integrated_lungcancer_peptides_advanced.csv', index=False)
    st.success("Lung cancer and peptides datasets have been preprocessed, merged, and saved.")

    # Markdown explanation
    st.markdown("""


To prepare the lung cancer dataset for analysis, we started by addressing missing values in its numerical columns using **KNN imputation**. This ensured the data was complete and ready for further processing. Next, we turned to the peptides dataset, where we focused on encoding its `class` column. Using **label encoding**, we transformed the categorical `class` labels into numerical values, and then applied **one-hot encoding** to create separate columns for each peptide class. This setup enabled more detailed analysis later.

With both datasets cleaned, we aligned them to prepare for integration. Because the two datasets didn’t share any features, we **added placeholder columns (`NaN`)** to ensure their structures matched. This critical step allowed us to combine the datasets seamlessly. To identify the most relevant features in the lung cancer dataset, we used **SelectKBest**, which ranked the top five numerical features most strongly associated with the `Stage` variable. For dimensionality reduction, we applied **Principal Component Analysis (PCA)** to the lung cancer dataset, distilling its features into just two components while preserving key information.

We then explored potential relationships between the datasets by calculating **Pearson correlations** for comparable features (when applicable). To avoid meaningless calculations, we skipped features without variability, ensuring only meaningful comparisons were included. Once these steps were complete, we added a `cancer_type` column to label rows by their dataset of origin (lung cancer or peptides) and merged the datasets into one integrated table. As a final step, we applied **normalization** to all numerical columns in the integrated dataset, scaling their values to a consistent range. This ensured no single column dominated the analysis due to differences in units or magnitudes.

The final result is a (mostly!) clean, unified dataset that combines lung cancer patient data with peptide activity data, after having performed eight rigorous preprocessing and integration steps.
    """)

# Data Handling page
elif page == "Handling the Data":
    st.header("Handling the Data")
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.experimental import enable_iterative_imputer  # needed for MICE
    from sklearn.impute import IterativeImputer
    from scipy import stats
    
    # Encode 'diagnosis' column ('M' -> 1, 'B' -> 0)
    if 'diagnosis' in breastcancer.columns:
        breastcancer['diagnosis'] = breastcancer['diagnosis'].map({'M': 1, 'B': 0})
        st.write("Encoded 'diagnosis' column in breast cancer dataset.")
    
    # Encode 'class' column
    if 'class' in peptides_b.columns:
        # One-hot encoding for multi-class
        peptides_b = pd.get_dummies(peptides_b, columns=['class'], drop_first=True)
        # Convert boolean values to integers (0s and 1s)
        peptides_b = peptides_b.astype({col: 'int' for col in peptides_b.columns if 'class_' in col})
        st.write("One-hot encoded 'class' column in peptides_b dataset.")

    st.markdown("""

    # Breast Cancer Data:

    We start with some encoding- in the interest of demonstrating something a bit more advanced, the peptides dataset is encoded with an if/else system that observes the data and picks one of the following based upon what is found:
    
    **Label Encoding:** If the class column had only two unique classes, a simple label encoding was applied, converting the classes into binary values (0 or 1).
    
    **One-Hot Encoding:** If there were more than two unique classes, one-hot encoding was applied to create separate binary columns for each class. This allows the model to interpret each class independently without imposing an ordinal relationship.
    
    The print statement near the top of this section indicates that One-Hot encoding was applied- the occasional statements of the type you will see within this app are the outputs of invisible code chunks running behind the scenes to process the data in real-time. This result isn't a surprise- the earlier IDA indicated that the dataset was multi-class. 
    """)

    # Display the updated peptides dataset
    st.write("Updated Peptides Dataset:")
    st.write(peptides_b)

    st.markdown("""
    Further research suggests that the 'Unnamed: 32' column full of NANs occurs when datasets include unnecessary commas after all of the variables. In any event, that column has got to go! But don't worry, we aren't done talking about missingness yet, even though we have clean data.
    """)

    # Drop the 'Unnamed: 32' column if it exists
    if 'Unnamed: 32' in breastcancer.columns:
        breastcancer.drop(columns=['Unnamed: 32'], inplace=True)
        st.write("Dropped 'Unnamed: 32' column from breast cancer dataset.")

    # Drop duplicates
    breastcancer.drop_duplicates(inplace=True)
    peptides_b.drop_duplicates(inplace=True)

    st.markdown("""
    Ok, we've technically demonstrated one way to handle missingness now, but we'll talk on it a bit more in a moment. We already know there are no duplicates that need to be removed.
    
    Before we get to discussing missingness (and imputation) further, though, there's a couple more things to talk through. Let's address those outliers we found in our IDA earlier. Since we're examining data from human bodies, where one is never quite the same as the next, it seems risky to remove the outliers outright. Instead, let's apply standardization. Since we'll be doing linear regression later, this is a better choice than normalization. We'll exclude id and diagnosis from the standardization because it is not meaningful to apply to those columns, and would actively mess up the diagnosis column.
    """)

    # Standardize numerical columns
    numerical_cols = breastcancer.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('id')
    numerical_cols.remove('diagnosis')

    scaler = StandardScaler()
    breastcancer[numerical_cols] = scaler.fit_transform(breastcancer[numerical_cols])  # Fit and transform the numerical columns to standardize

    st.write("Standardized Breast Cancer Dataset (excluding ID and Diagnosis):")
    st.write(breastcancer.head())

    z_scores = np.abs(breastcancer[numerical_cols].mean(axis=0))  # An additional outlier check post-standardization 
    st.write("Z-scores after standardization:")
    st.write(z_scores)

    st.markdown("The next thing we should probably have a look at is this:")

    # Display correlation heatmap
    numeric_cols = breastcancer.select_dtypes(include=['float64', 'int64']).columns  # Numeric columns for correlation
    correlation_matrix = breastcancer[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix,  # A Plotly heatmap, so it can be interactive
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    title='Correlation Heatmap of Breast Cancer Features',
                    labels=dict(x="Features", y="Features", color="Correlation Coefficient"))
    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(tickangle=45),  # X-axis labels
        yaxis=dict(tickangle=0)    # Set Y-axis labels to horizontal
    )
    st.plotly_chart(fig)  # Display the Plotly chart in Streamlit

    st.markdown("""
    If your first thought is *"Woah, this is a bit much. Are you sure this is even useful?"* - **don't fret. That's the reaction I was aiming for.** 
    I've made the plot interactive if you wish to scrub through it, but I think we can do better. This analysis needs to **get to the point, right?**
    
    To that end, the main reason I’ve provided this heatmap of the breast cancer dataset to explain why we should use feature selection on it. 
    
    Originally, I **tried PCA**, but this happened:
    """)

    # Perform PCA
    from sklearn.decomposition import PCA

    numeric_cols = breastcancer.select_dtypes(include=['float64', 'int64']).columns  # Numeric columns for PCA
    X = breastcancer[numeric_cols].drop('diagnosis', axis=1)  # Features
    y = breastcancer['diagnosis']  # Target variable

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]  # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df['diagnosis'] = y  # Add the target variable to the PCA DataFrame

    correlation_matrix = pca_df.corr()  # Calculate the correlation matrix of the PCA features
    plt.figure(figsize=(10, 8))  # Create a heatmap of the PCA correlation features
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of PCA Features')
    st.pyplot(plt)  # Display the matplotlib figure in Streamlit

    st.markdown("""
    Well, that isn't very useful. Hence why I went with **Random Forest**:
    """)

    # Random Forest feature importance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Separate the features from the target variable
    X = breastcancer.drop(columns=['id', 'diagnosis'])  # Features
    y = breastcancer['diagnosis']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # The Random Forest Classifier
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    
    # Feature importances DataFrame:
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # Select top N features based on importance
    top_n = st.slider("Select the number of top features to display:", 1, 20, 10)  # Slider to control feature selection
    top_features = importance_df.nlargest(top_n, 'Importance')['Feature'].values
    selected_features_df = breastcancer[top_features.tolist() + ['diagnosis']]  # New, smaller DataFrame
    
    correlation_matrix = selected_features_df.corr()  # Calculate the correlation matrix of the selected features
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Selected Features Impacting Diagnosis')
    st.pyplot(plt)  # Display the matplotlib figure in Streamlit

    st.markdown("""
    I promise we'll be getting to the insights (and more visualizations) shortly.

    Before we do that, though, remember how I said I'd talk more about missingness and imputation?

    While the second part of our EDA (and, therefore, the conclusions of the application itself) will settle on a way forward and go from there, be sure to check out the final section of the application for some important discussion on alternative methods and thoughts on the application's usage as a whole.

    We'll copy our trimmed down breast cancer dataframe so the original isn't modified, introduce some fake missingness and do some analysis:
    """)

    # Create fake missingness
    selected_features_copy = selected_features_df.copy()  # Making a copy of the selected_features dataset 
    np.random.seed(42)
    missing_mask = np.random.rand(*selected_features_copy.shape) < 0.1  # Fake missingness, mask where 10% of the data is randomly set to NaN
    X_with_missing = selected_features_copy.mask(missing_mask)

    # Imputation methods
    mean_imputer = SimpleImputer(strategy='mean')  # Mean imputation
    X_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(X_with_missing), columns=selected_features_copy.columns)
    
    knn_imputer = KNNImputer(n_neighbors=5)  # KNN imputation
    X_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(X_with_missing), columns=selected_features_copy.columns)
    
    # Increase max_iter in IterativeImputer to allow more iterations
    mice_imputer = IterativeImputer(random_state=42, max_iter=20)  
    X_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(X_with_missing), columns=selected_features_copy.columns)
    
    def evaluate_imputation(original, imputed, mask):  # Function to evaluate the imputation
        original_masked = original[mask]
        imputed_masked = imputed[mask]   
        # Ensure there are values to compare
        if original_masked.size == 0 or imputed_masked.size == 0:
            st.warning("Warning: No imputed values to compare. Check the mask and imputation process.")
        # Calculate the MSE on the values that were originally missing
        return mean_squared_error(original_masked, imputed_masked)
    
    # Evaluate only the values that were originally missing
    mse_mean = evaluate_imputation(selected_features_copy.values, X_mean_imputed.values, missing_mask)
    mse_knn = evaluate_imputation(selected_features_copy.values, X_knn_imputed.values, missing_mask)
    mse_mice = evaluate_imputation(selected_features_copy.values, X_mice_imputed.values, missing_mask)
    
    st.write("MSE for Mean Imputation:", mse_mean)
    st.write("MSE for KNN Imputation:", mse_knn)
    st.write("MSE for MICE Imputation:", mse_mice)

    st.markdown("""
    Well that's interesting! Mean imputation, which replaces missing values with the mean of each feature, results in the highest MSE- by far. This is because mean imputation oversimplifies the data by ignoring the relationships between features, leading to distorted results. As a result, the imputed values significantly deviate from the true data, reflected in the VERY high MSE value!.
    
    KNN imputation performs better, yielding a much lower MSE. This method looks at the nearest neighbors in the dataset and fills in missing values based on patterns in similar data points. KNN captures local relationships between variables, which leads to a more accurate estimation of the missing values compared to mean imputation. However, it may still fall short in capturing broader patterns or when data is sparse, but overall, it actually does quite good!
    
    The winner, however, is MICE, which produces the lowest MSE among the methods, suggesting it offers the most accurate imputation. By iteratively modeling missing data using all other features, MICE maintains the relationships between variables more effectively than both mean and KNN imputation. This iterative process allows for better handling of complex datasets, ensuring that the imputed values closely resemble the true data. MICE’s ability to account for interactions and variability in the dataset results in the most reliable imputation, reflected in its minimal MSE, making it the best option in this comparison.
    
    As interesting as this all is, it's probably best for the sake of this application to just stick with the real data since it's already so clean! So why discuss the missingness at all?
    
    First of all, data is not always going to be this clean. It's practically a given that further cancer research (and possibly the final version of this very application) will include missingness. By doing this now, we've learned what the best method for imputation is- and that's a very good thing to know. Additionally, studying the effects of introduced missingness on the overall dataset can help identify which variables are crucial for accurate imputation and model performance. That can highlight the features that might be influencing the outcome variable most strongly and are, therefore, particularly critical for predictive modeling.
    """)

    # Basic statistical summary of the trimmed breast cancer dataset
    st.write("Statistical Summary of Selected Features DataFrame:")
    st.write(selected_features_df.describe())

    # Basic statistical summary of the peptides dataset
    st.write("\nStatistical Summary of Peptides DataFrame:")
    st.write(peptides_b.describe())

    # Lung Cancer Data:
    st.markdown("# Lung Cancer Data")

    st.write("Lung Cancer Dataset:")
    st.write(lungcancer_improved.head())

    st.write("Peptides Dataset:")
    st.write(peptides_l_improved.head())

    st.write("Integrated Dataset:")
    st.write(lungdatamerged.head())

    # ============================
    # FEATURE ENGINEERING AND TRANSFORMATIONS
    # ============================

    # Feature Interaction: Tumor_Size_mm x Smoking_Pack_Years
    lungcancer_improved['Tumor_Size_Smoking_Interaction'] = (
        lungcancer_improved['Tumor_Size_mm'] * lungcancer_improved['Smoking_Pack_Years']
    )

    st.subheader("Lung Cancer Dataset with Tumor Size and Smoking Interaction")
    st.write(lungcancer_improved[['Tumor_Size_mm', 'Smoking_Pack_Years', 'Tumor_Size_Smoking_Interaction']].head())

    # Polynomial Features: Survival_Months
    lungcancer_improved['Survival_Months_Squared'] = lungcancer_improved['Survival_Months'] ** 2
    lungcancer_improved['Survival_Months_Cubed'] = lungcancer_improved['Survival_Months'] ** 3

    st.subheader("Polynomial Features for Survival Months")
    st.write(lungcancer_improved[['Survival_Months', 'Survival_Months_Squared', 'Survival_Months_Cubed']].head())

    # Target Encoding: Stage
    stage_mapping = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
    lungcancer_improved['Stage_Numeric'] = lungcancer_improved['Stage'].map(stage_mapping)

    st.subheader("Target Encoded Stage")
    st.write(lungcancer_improved[['Stage', 'Stage_Numeric']].head())

    # Advanced Transformation: Power Transform on Tumor_Size_mm
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    lungcancer_improved['Tumor_Size_mm_Transformed'] = power_transformer.fit_transform(
        lungcancer_improved[['Tumor_Size_mm']].fillna(0)
    )

    st.subheader("Power Transformed Tumor Size")
    st.write(lungcancer_improved[['Tumor_Size_mm', 'Tumor_Size_mm_Transformed']].head())

    # Dimensionality Reduction: PCA on Class-Related Features
    class_features = [
        'class_very active', 'class_inactive - exp',
        'class_inactive - virtual', 'class_mod. active'
    ]

    peptides_l_improved[class_features] = peptides_l_improved[class_features].fillna(0)
    pca = PCA(n_components=2, random_state=42)
    peptides_l_improved[['Class_PCA1', 'Class_PCA2']] = pca.fit_transform(
        peptides_l_improved[class_features]
    )

    st.subheader("PCA on Class-Related Features")
    st.write(peptides_l_improved[['Class_PCA1', 'Class_PCA2']].head())

    # Feature Interaction in Integrated Dataset
    lungdatamerged['Tumor_Size_Smoking_Interaction'] = (
        lungdatamerged['Tumor_Size_mm'] * lungdatamerged['Smoking_Pack_Years']
    )

    st.subheader("Integrated Dataset with Tumor Size and Smoking Interaction")
    st.write(lungdatamerged[['Tumor_Size_mm', 'Smoking_Pack_Years', 'Tumor_Size_Smoking_Interaction']].head())

    # Polynomial Features in Integrated Dataset
    lungdatamerged['Survival_Months_Squared'] = lungdatamerged['Survival_Months'] ** 2
    lungdatamerged['Survival_Months_Cubed'] = lungdatamerged['Survival_Months'] ** 3

    st.subheader("Polynomial Features in Integrated Dataset")
    st.write(lungdatamerged[['Survival_Months', 'Survival_Months_Squared', 'Survival_Months_Cubed']].head())

    # Target Encoding in Integrated Dataset
    lungdatamerged['Stage_Numeric'] = lungdatamerged['Stage'].map(stage_mapping)

    st.subheader("Target Encoded Stage in Integrated Dataset")
    st.write(lungdatamerged[['Stage', 'Stage_Numeric']].head())

    # Power Transform in Integrated Dataset
    lungdatamerged['Tumor_Size_mm_Transformed'] = power_transformer.fit_transform(
        lungdatamerged[['Tumor_Size_mm']].fillna(0)
    )

    st.subheader("Power Transformed Tumor Size in Integrated Dataset")
    st.write(lungdatamerged[['Tumor_Size_mm', 'Tumor_Size_mm_Transformed']].head())

    # PCA in Integrated Dataset
    lungdatamerged[class_features] = lungdatamerged[class_features].fillna(0)
    lungdatamerged[['Class_PCA1', 'Class_PCA2']] = pca.transform(
        lungdatamerged[class_features]
    )

    st.subheader("PCA in Integrated Dataset")
    st.write(lungdatamerged[['Class_PCA1', 'Class_PCA2']].head())

    # Clustering in Integrated Dataset
    clustering_features = [
        'Tumor_Size_mm_Transformed', 'Survival_Months',
        'Smoking_Pack_Years', 'Stage_Numeric'
    ]

    lungdatamerged[clustering_features] = lungdatamerged[clustering_features].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42)
    lungdatamerged['Cluster_Labels'] = kmeans.fit_predict(
        lungdatamerged[clustering_features]
    )

    st.subheader("Cluster Labels in Integrated Dataset")
    st.write(lungdatamerged[['Cluster_Labels']].head())

    # Save the processed datasets
    lungcancer_improved.to_csv('improved_lungcancer_processed.csv', index=False)
    peptides_l_improved.to_csv('improved_peptides_processed.csv', index=False)
    lungdatamerged.to_csv('transformed_lungdatamerged.csv', index=False)

    st.success("Feature engineering and advanced data transformation completed successfully.")

    # Markdown explanation
    st.markdown("""
    To enhance the merged dataset, we began by creating new interaction terms to capture relationships between features. Specifically, we calculated the product of Tumor_Size_mm and Smoking_Pack_Years, creating a Tumor_Size_Smoking_Interaction feature. This interaction term provides insight into how tumor size and smoking history might jointly influence patient outcomes.

    Next, we expanded the Survival_Months feature by generating polynomial terms, including its square and cube, to capture potential nonlinear trends in survival data. These polynomial features allow for more nuanced modeling of survival patterns.

    To prepare categorical features for analysis, we performed target encoding on the Stage column, mapping its categories (Stage I through Stage IV) to numerical values. This encoding simplifies downstream modeling while preserving the ordinal nature of the feature.

    We applied a **Yeo-Johnson Power Transform** to the Tumor_Size_mm column to stabilize variance and normalize its distribution. This advanced transformation ensures that the feature aligns better with statistical and machine learning models.

    To reduce the dimensionality of the class-related features, we utilized **Principal Component Analysis (PCA)**, transforming them into two principal components. This step preserves the majority of the variance in the original features while simplifying their representation, which can be particularly useful for clustering and visualization.

    Finally, we implemented **K-Means Clustering** on selected numerical features, including the transformed Tumor_Size_mm, Survival_Months, Smoking_Pack_Years, and the encoded Stage. This unsupervised learning method groups data points into clusters based on shared characteristics, offering an alternative perspective on patterns in the data.

    """)



# Add a "Modeling" page
if page == "Modeling":
    st.title("Modeling Results and Insights")

    # Overview section
    st.markdown("""
    ## Overview
    This section delves into the results of applying machine learning models to the lung cancer and peptides datasets. 
    The goals were to identify patterns in the data, evaluate the performance of various models, and highlight the key features 
    driving predictions. Modeling included classification for disease stages and peptide activity, as well as regression for 
    predicting survival months.

    The models evaluated include:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Voting Classifier (Ensemble)
    - Linear Regression (for regression tasks)

    Each model underwent preprocessing and optimization tailored to the dataset. For example, lung cancer data posed challenges 
    due to overlapping features, while peptide datasets excelled due to their structured nature and clear class separations. 
    Below, explore detailed insights for each model via the tabs.

    NOTE: Not every model was used on every dataset! Keep an eye on this as you scroll the tabs.
    """)

    # Create tabs for model results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Logistic Regression", "Random Forest",
        "Gradient Boosting", "Voting Classifier",
        "Linear Regression (Regression Task)"
    ])

    with tab1:
        st.markdown("""
        ### Logistic Regression
        Logistic Regression was applied to both datasets with scaling and hyperparameter tuning via GridSearchCV. 
        While the peptide datasets demonstrated perfect classification performance, logistic regression struggled on the lung cancer dataset, 
        highlighting the challenges in separating overlapping feature distributions.

        **Performance Metrics:**

        **Lung Cancer:**
        - Accuracy: 25.56%
        - F1 Score: 25.42%
        - ROC-AUC: 50.11%

        **Peptides:**
        - Accuracy: 100.00%
        - F1 Score: 100.00%
        - ROC-AUC: 100.00%

        Logistic regression effectively captured the clear structure in the peptides datasets but was limited on the lung cancer data 
        due to its sensitivity to feature overlap and linear decision boundaries.
        """)

    with tab2:
        st.markdown("""
        ### Random Forest
        Random Forest models leveraged ensemble learning to improve predictions by averaging over decision trees. 
        They were particularly suited for handling non-linear relationships and high-dimensional data.

        **Performance Metrics:**

        **Lung Cancer:**
        - Accuracy: 25.06%
        - F1 Score: 25.05%
        - ROC-AUC: 50.31%

        **Peptides:**
        - Accuracy: 100.00%
        - F1 Score: 100.00%
        - ROC-AUC: 100.00%

        Despite its strong capabilities in capturing non-linear patterns, Random Forest faced difficulties with the lung cancer dataset 
        due to the lack of distinct class separations, while it excelled in the structured peptides datasets.
        """)

    with tab3:
        st.markdown("""
        ### Gradient Boosting
        Gradient Boosting models were tuned for learning rate, max depth, and the number of estimators. 
        Their ability to handle small datasets with complex patterns made them highly effective in most tasks.

        **Performance Metrics:**

        **Lung Cancer:**
        - Accuracy: 25.50%
        - F1 Score: 25.41%
        - ROC-AUC: 50.28%

        **Peptides:**
        - Accuracy: 100.00%
        - F1 Score: 100.00%
        - ROC-AUC: 100.00%

        **Breast Cancer:**
        - Accuracy: 93.57%
        - F1 Score: 93.51%
        - ROC-AUC: 99.09%

        Gradient Boosting stood out in capturing non-linear trends in both the breast cancer and regression tasks, 
        achieving exceptional results on the peptides datasets.
        """)

    with tab4:
        st.markdown("""
        ### Voting Classifier
        The Voting Classifier combined predictions from logistic regression, random forest, and gradient boosting 
        to improve accuracy through soft voting. This approach demonstrated enhanced robustness, especially for structured datasets.

        **Performance Metrics:**

        **Lung Cancer:**
        - Accuracy: 25.50%
        - F1 Score: 25.41%
        - ROC-AUC: 50.28%

        **Peptides:**
        - Accuracy: 100.00%
        - F1 Score: 100.00%
        - ROC-AUC: 100.00%

        **Breast Cancer:**
        - Accuracy: 94.74%
        - F1 Score: 94.69%
        - ROC-AUC: 99.49%

        The ensemble nature of the Voting Classifier allowed it to balance strengths and weaknesses of individual models, 
        resulting in the highest accuracy for the breast cancer dataset and flawless results for peptides.
        """)

    with tab5:
        st.markdown("""
        ### Linear Regression (Regression Task)
        Regression tasks were applied to predict survival months in the lung cancer dataset. Linear regression captured broad trends, 
        while Gradient Boosting delivered perfect predictions by accounting for non-linearities.

        **Performance Metrics:**

        **Linear Regression:**
        - Mean Absolute Error: 2.7264
        - R^2 Score: 0.9903

        **Gradient Boosting Regressor:**
        - Mean Absolute Error: 0.0000
        - R^2 Score: 1.0000

        Gradient Boosting proved particularly effective, highlighting the benefits of advanced ensemble methods in regression scenarios.
        """)

    # Key takeaways section
    # Key takeaways section
    st.markdown("""
## Key Takeaways

The results from the modeling efforts revealed some interesting insights about the performance of various algorithms applied to the lung and breast cancer datasets. 

### Lung Cancer Dataset
For the lung cancer data, logistic regression, random forest, and the voting classifier all achieved similar predictive performance, with accuracies around 25.5% and F1 scores reflecting the same low level of distinction among classes. The ROC-AUC scores hovered near 0.50, indicating these models were no better than random guessing. These results highlight significant challenges in the lung cancer dataset, likely due to overlapping feature distributions or insufficient distinguishing patterns for the target variable. However, the lung cancer peptides dataset produced flawless results across all models, achieving perfect accuracy, precision, recall, and F1 scores. 

### Breast Cancer Dataset
The models performed exceptionally well on the breast cancer dataset, demonstrating the effectiveness of advanced ensemble techniques and hyperparameter optimization. Logistic regression achieved an accuracy of 96.5% and an impressive ROC-AUC of 99.8%. Random forest and gradient boosting classifiers followed closely, with accuracies of 94.1% and 93.6%, respectively. The voting classifier combined these models' strengths and delivered the highest performance, achieving a 94.7% accuracy. The peptides dataset for breast cancer, much like the lung cancer peptides dataset, exhibited perfect predictive performance, highlighting its consistent structure and high-quality features.

### Regression Tasks
Linear regression and gradient boosting were applied to regression tasks to predict Survival_Months in the lung cancer dataset and demonstrated stark contrasts in performance. Linear regression achieved an R^2 score of 0.99, indicating a strong fit to the data, but gradient boosting regressor went a step further with perfect R^2 and zero error metrics, showcasing the power of advanced algorithms in regression problems. Similarly, linear regression on the breast cancer dataset yielded an R^2 score of 0.74, capturing broad trends but leaving room for improvement. Gradient boosting excelled by capturing the non-linearities and intricate feature interactions, solidifying its role as a top performer in both classification and regression contexts.

### Advanced Methodologies
Several advanced methodologies were integral to the success of these models. Ensemble methods, including random forest, gradient boosting, and voting classifiers, demonstrated their ability to capture complex relationships and improve predictive accuracy by combining multiple weak learners. Hyperparameter tuning was performed systematically using GridSearchCV, allowing for optimal configuration of model parameters such as learning rates, maximum depths, and the number of estimators. These optimizations were essential in enhancing model performance, particularly for ensemble methods, where small changes in parameters can significantly impact results. Scaling was applied to datasets to standardize features, particularly for models like logistic regression and gradient boosting, which are sensitive to varying feature magnitudes.

### Domain-Specific Insights
The integration of specialized methodologies for the peptides datasets was another noteworthy aspect of this analysis. These datasets required domain-specific preprocessing, including encoding categorical variables and ensuring compatibility for machine learning models. By carefully structuring the data and leveraging algorithms suited to high-dimensional features, the analysis achieved perfect classification results, emphasizing the importance of adapting methodologies to the domain. The peptides datasets' molecular nature and structured class distributions enabled models to excel, reflecting proficiency in handling domain-specific data types.

### Validation and Generalizability
Cross-validation was used extensively to ensure that performance metrics reflected model generalizability and were not biased by specific data splits. This step was vital in identifying the best models for both datasets. Additionally, the integration of ensemble methods like the voting classifier demonstrated an advanced understanding of leveraging diverse algorithms to improve predictive performance.
    """)



# Main Insights section
if page == "Main Insights":
    st.header("Main Insights")

    # Content explaining peptide sequences
    st.markdown("""
    # Breast Cancer:

    Let's think about the peptides a bit more.

    It would help to explain a bit what a peptide sequence means. Take for example the sequence "AAWKWAWAKKWAKAKKWAKAA". This represents a specific amino acid sequence of a peptide, where each letter corresponds to a one-letter code for an amino acid, in this case:

    - A = Alanine
    - W = Tryptophan
    - K = Lysine

    
    """)

    # Merge the datasets on the 'cancer_type' column
    selected_features_df = breastcancer[[
    'concave points_mean',
    'concave points_worst',
    'area_worst',
    'concavity_mean',
    'radius_worst',
    'perimeter_worst',
    'perimeter_mean',
    'area_mean',
    'concavity_worst',
    'radius_mean',
    'diagnosis'
    ]]
    selected_features_df['cancer_type'] = 'breast'
    peptides_b['cancer_type'] = 'breast'

    merged_dataset = pd.merge(selected_features_df, peptides_b, on='cancer_type')
    st.write("Merged Dataset:")
    st.write(merged_dataset.head())  # Display the head of the merged dataset

    # Prepare the regression results data
    data = {
        "Variable": [
            "const",
            "concave points_mean",
            "concave points_worst",
            "area_worst",
            "concavity_mean",
            "radius_worst",
            "perimeter_worst",
            "perimeter_mean",
            "area_mean",
            "concavity_worst",
            "radius_mean"
        ],
        "Coefficient": [
            0.3726,
            0.1946,
            0.0366,
            -0.8981,
            -0.0529,
            1.4586,
            -0.0852,
            -0.5089,
            0.5728,
            0.0907,
            -0.3907
        ],
        "Std Err": [
            0.011,
            0.063,
            0.044,
            0.139,
            0.059,
            0.211,
            0.152,
            0.415,
            0.150,
            0.040,
            0.427
        ],
        "t": [
            35.012,
            3.111,
            0.832,
            -6.474,
            -0.896,
            6.928,
            -0.560,
            -1.227,
            3.806,
            2.244,
            -0.916
        ],
        "P>|t|": [
            0.000,
            0.002,
            0.406,
            0.000,
            0.371,
            0.000,
            0.575,
            0.220,
            0.000,
            0.025,
            0.360
        ],
        "[0.025": [
            0.352,
            0.072,
            -0.050,
            -1.171,
            -0.169,
            1.045,
            -0.384,
            -1.324,
            0.277,
            0.011,
            -1.229
        ],
        "0.975]": [
            0.393,
            0.317,
            0.123,
            -0.626,
            0.063,
            1.872,
            0.214,
            0.306,
            0.868,
            0.170,
            0.447
        ]
    }

    # Create a DataFrame
    results_df = pd.DataFrame(data)

    # Display the regression results as a table in Streamlit
    st.subheader("Linear Regression Results")
    st.table(results_df)

    # Add a summary of significant predictors
    st.markdown("""
    **Significant Predictors:**
    - Concave Points Mean: Positive correlation
    - Area Worst: Negative correlation
    - Radius Worst: Positive correlation
    - Area Mean: Positive correlation
    - Concavity Worst: Positive correlation

    Here's a linear regression on the feature-selected breast cancer dataset. These features are the most influential in predicting cancer diagnosis.
    We know what the most significant contributing factors to breast cancer are now, so the question is, what can we do with the peptide data?

    """)

    # Peptide analysis
    st.markdown("""
    Let's analyze the peptide data further:
    """)

    # Re-encode the peptides_b DataFrame if the class columns are missing
    expected_columns = ['class_inactive - virtual', 'class_mod. active', 'class_very active']
    if not all(col in peptides_b.columns for col in expected_columns):
    # One-hot encoding for the class column if it doesn't exist
       peptides_b = pd.get_dummies(peptides_b, columns=['class'], drop_first=True)
       st.write("Re-encoded the 'peptides_b' DataFrame.")

    # Now proceed with accessing the class columns
    inactive_peptides = peptides_b[peptides_b['class_inactive - virtual'] == 1]
    mod_active_peptides = peptides_b[peptides_b['class_mod. active'] == 1]
    very_active_peptides = peptides_b[peptides_b['class_very active'] == 1]


    inactive_peptides = peptides_b[peptides_b['class_inactive - virtual'] == 1]
    mod_active_peptides = peptides_b[peptides_b['class_mod. active'] == 1]
    very_active_peptides = peptides_b[peptides_b['class_very active'] == 1]

    def count_amino_acids(peptide_sequences):  # function that counts frequency of each letter in a list of sequences
        all_letters = ''.join(peptide_sequences)  # fuse sequences into a single string
        return Counter(all_letters)  # count frequency of each letter

    inactive_counts = count_amino_acids(inactive_peptides['sequence'])
    mod_active_counts = count_amino_acids(mod_active_peptides['sequence'])
    very_active_counts = count_amino_acids(very_active_peptides['sequence'])

    top_10_inactive = inactive_counts.most_common(10)
    top_10_mod_active = mod_active_counts.most_common(10)
    top_10_very_active = very_active_counts.most_common(10)

    top_10_inactive = [
    ["L", 1508],
    ["A", 1268],
    ["E", 1061],
    ["K", 876],
    ["V", 797],
    ["I", 794],
    ["R", 791],
    ["S", 672],
    ["Q", 619],
    ["D", 610]
    ]

    top_10_mod_active = [
    ["K", 502],
    ["L", 318],
    ["A", 294],
    ["F", 135],
    ["I", 87],
    ["V", 80],
    ["G", 77],
    ["W", 71],
    ["S", 53],
    ["R", 34]
    ]

    top_10_very_active = [
    ["K", 108],
    ["L", 69],
    ["F", 35],
    ["A", 35],
    ["W", 19],
    ["I", 16],
    ["P", 14],
    ["G", 10],
    ["S", 9],
    ["R", 9]
    ]

    # Convert lists to DataFrames
    inactive_df = pd.DataFrame(top_10_inactive, columns=["Amino Acid", "Count"])
    mod_active_df = pd.DataFrame(top_10_mod_active, columns=["Amino Acid", "Count"])
    very_active_df = pd.DataFrame(top_10_very_active, columns=["Amino Acid", "Count"])

    # Display the DataFrames as tables in Streamlit
    st.write("Top 10 Amino Acids in 'Inactive - Virtual' Class:")
    st.table(inactive_df)

    st.write("Top 10 Amino Acids in 'Moderately Active' Class:")
    st.table(mod_active_df)

    st.write("Top 10 Amino Acids in 'Very Active' Class:")
    st.table(very_active_df)

    # Adding the study content
    st.markdown("""
    At first glance it's a bit hard to see much interesting here. We can go a bit deeper though:
    """)

    import pandas as pd
    from collections import Counter

    def generate_ngrams(sequence, n=2):  # function that generates n-grams
        return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

    def ngram_frequencies(df, n=2):  # function to extrapolate n-gram frequencies for a specific class
        ngram_counter = Counter()
        for sequence in df['sequence']:
            ngrams = generate_ngrams(sequence, n)
            ngram_counter.update(ngrams)
        return ngram_counter

    inactive_peptides = peptides_b[peptides_b['class_inactive - virtual'] == 1]
    mod_active_peptides = peptides_b[peptides_b['class_mod. active'] == 1]
    very_active_peptides = peptides_b[peptides_b['class_very active'] == 1]

    n = 3 
    ngrams_inactive = ngram_frequencies(inactive_peptides, n)
    ngrams_mod_active = ngram_frequencies(mod_active_peptides, n)
    ngrams_very_active = ngram_frequencies(very_active_peptides, n)

    ngram_df = pd.DataFrame({
    'inactive': pd.Series(ngrams_inactive),
    'mod_active': pd.Series(ngrams_mod_active),
    'very_active': pd.Series(ngrams_very_active)
    }).fillna(0)

    ngram_df['diff_active_vs_inactive'] = ngram_df['very_active'] - ngram_df['inactive']  # which n-grams are most enriched in very active peptides?

    # Print the results in a nice table
    st.write("Top N-grams Differences Between Active and Inactive Peptides:")
    st.table(ngram_df.sort_values('diff_active_vs_inactive', ascending=False).head(10))

    # Additional analysis on positional amino acid frequencies
    inactive_peptides = peptides_b[peptides_b['class_inactive - virtual'] == 1]  # subset the peptides based on activity class
    mod_active_peptides = peptides_b[peptides_b['class_mod. active'] == 1] 
    very_active_peptides = peptides_b[peptides_b['class_very active'] == 1]

    def positional_aa_frequencies(df, max_len=20):  # this function calculates positional amino acid frequencies
        aa_position_matrix = np.zeros((max_len, 20))  # 20 amino acids, and sequences of length max_len
        aa_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # mapping amino acids to index
        for sequence in df['sequence']:
            for pos, aa in enumerate(sequence):
                if pos < max_len and aa in aa_index:
                   aa_position_matrix[pos, aa_index[aa]] += 1
        aa_position_matrix /= len(df)
        return aa_position_matrix

    max_sequence_length = 20
    position_matrix_inactive = positional_aa_frequencies(inactive_peptides, max_len=max_sequence_length)
    position_matrix_mod_active = positional_aa_frequencies(mod_active_peptides, max_len=max_sequence_length)
    position_matrix_very_active = positional_aa_frequencies(very_active_peptides, max_len=max_sequence_length)

    def plot_heatmap(position_matrix, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(position_matrix, cmap="Blues", xticklabels=list("ACDEFGHIKLMNPQRSTVWY"), yticklabels=range(1, max_sequence_length + 1))
        plt.title(title)
        plt.xlabel("Amino Acids")
        plt.ylabel("Position")
        st.pyplot(plt)  # Use Streamlit's function to display the plot
        plt.clf()  # Clear the current figure to avoid overlap with future plots


    plot_heatmap(position_matrix_inactive, "Positional Amino Acid Frequencies in Inactive Peptides")
    plot_heatmap(position_matrix_mod_active, "Positional Amino Acid Frequencies in Moderately Active Peptides")
    plot_heatmap(position_matrix_very_active, "Positional Amino Acid Frequencies in Very Active Peptides")

    st.markdown("""
Well, these results are a bit more illuminating!

From all 3 tests, we can see that the amino acid K, or Lysine, pops up quite frequently in active peptides. Doing single-character analysis on the peptide sequences, however, suggested Lysine was fairly common in the inactive peptides as well. So what makes the difference in Lysine being effective vs. not in different peptides? The answer likely involves what other enzymes it's being coupled with, and possibly its positioning.

Looking for 3-character sequences instead, we started to find some sequences that appear more in the active peptides than inactive enough to be noticeable: particularly KWK, which appeared 0 times in the inactive peptides, 15 times in the moderately active peptides, and 4 times in the very active. Noticeably, W did not appear very frequently in single character analysis, but seems a part of multiple somewhat successful sequences, often alongside P and F- notably, the single character analysis did NOT show these in its top 10 frequently occurring amino acids.

For reference as to what these stand for: W: Tryptophan P: Proline F: Phenylalanine

Looking at positional frequencies (does having the peptide at a certain position in the sequence matter?), we can see that for the active peptides, the positions are overall MUCH more specific- large chunks of the heatmap show certain positions as almost nonexistent in active peptides. for example, there's a decent amount of D (Aspartate) in the inactive peptides, but seemingly none in the active ones. The heatmap seems to suggest K, F, W, A, V, P, S and L as the most present, but comparing with which of those are also common in inactive peptides, but removing the 3 lines that share the darkest squares in the inactive peptides heatmap, we're left with

K, F, W, P and S (K: Lysine, F: Phenylalanine, W: Tryptophan, P: Proline, S: Serine) with the observation that grouping matters, too (notice some darkness in K on the first heatmap despite our observations that Lysine seems useful).

Despite this, overall Lysine appears to be the biggest star of the show, especially with the consideration of grouping involved.

For the remaining visualizations, we'll use that merged dataset so everything's all in one place for quick and easy plotting. I'll make sure to include some more interactive visualizations along with these, too:
    """)

if page == "Main Insights":  # Replace with your actual section condition
    # Re-encode the peptides_b DataFrame if the class columns are missing
    expected_columns = ['class_inactive - virtual', 'class_mod. active', 'class_very active']
    if not all(col in peptides_b.columns for col in expected_columns):
        # One-hot encoding for the class column if it doesn't exist
        peptides_b = pd.get_dummies(peptides_b, columns=['class'], drop_first=True)
        st.write("Re-encoded the 'peptides_b' DataFrame.")

    # Now proceed with accessing the class columns
    inactive_peptides = peptides_b[peptides_b['class_inactive - virtual'] == 1]
    mod_active_peptides = peptides_b[peptides_b['class_mod. active'] == 1]
    very_active_peptides = peptides_b[peptides_b['class_very active'] == 1]

    # Count the occurrences of each class using the already defined variables
    peptides_class_counts = pd.Series({
        'class_inactive - virtual': len(inactive_peptides),
        'class_mod. active': len(mod_active_peptides),
        'class_very active': len(very_active_peptides)
    })

    # Create the bar plot for peptide class distribution
    fig2 = px.bar(peptides_class_counts, 
                   title='Peptide Class Distribution',
                   labels={'value': 'Count', 'index': 'Peptide Class'},
                   color=peptides_class_counts.index,
                   color_discrete_sequence=['orange', 'green', 'blue'])

    # Update layout for the figure
    fig2.update_layout(xaxis_title='Peptide Class', yaxis_title='Count')

    # Display the plot in Streamlit
    st.plotly_chart(fig2)  # Use Streamlit's function to display Plotly figures

    # Add the explanatory text
    st.markdown("""
    Here's the peptide class distribution visualized, once again making use of encoding in visualization- I say "once again" because all of the visualizations involving diagnosis in the breast cancer dataset leverage the fact that that variable was encoded. 
    """)
    # Create the box plot for distribution of Area Mean by Diagnosis using the breastcancer dataset
    fig1 = px.box(breastcancer, 
               x='diagnosis', 
               y='area_mean', 
               title='Distribution of Area Mean by Diagnosis',
               labels={'area_mean': 'Area Mean', 'diagnosis': 'Diagnosis'},
               color='diagnosis',
               color_discrete_sequence=['lightcoral', 'lightblue'])

    # Display the plot in Streamlit
    st.plotly_chart(fig1)  # Use Streamlit's function to display Plotly figures

    # Add the explanatory text for the box plot if needed
    st.markdown("""
This box plot visualizes the distribution of the area mean by diagnosis, allowing us to see the spread and central tendency of the area mean for different diagnosis categories.
    """)


    # Create the scatter plot for Radius Mean vs. Perimeter Mean using the breast cancer dataset
    fig3 = px.scatter(breastcancer, 
                   x='radius_mean', 
                   y='perimeter_mean', 
                   color='diagnosis',
                   title='Scatter Plot of Radius Mean vs. Perimeter Mean',
                   labels={'radius_mean': 'Radius Mean', 'perimeter_mean': 'Perimeter Mean'},
                   hover_data=['area_mean'])

    # Display the plot in Streamlit
    st.plotly_chart(fig3)  # Use Streamlit's function to display Plotly figures

    # Add the explanatory text
    st.markdown("""
    Ok, you get the general idea. Numbers go up, things get bigger, risk gets bigger. We know from the rest of our analysis that irregular shapes are also a problem when it comes to tumors.

Alright, let's put this into a concise conclusion.

Certain anticancer peptides can act as immunomodulators, enhancing the body’s immune response against cancer cells. This might involve activating immune cells to recognize and attack cells exhibiting high concavity and irregular shapes (for example, based on our analysis showing such things as significant), associated with malignancy. A peptide that can specifically bind to or target cell surface receptors or markers associated with the characteristics measured by these predictors (e.g., concave points or area) could help inhibit the growth of malignant cells. For example, a peptide might interfere with pathways that lead to increased cell irregularities or uncontrolled growth. Furthermore, a peptide that can trigger apoptosis (programmed cell death) in cells with malignant characteristics—such as those with elevated concavity_worst—would be beneficial. This could effectively reduce the population of harmful cells without affecting healthy ones.

Based on the analysis we've done, these sound like the most effective anticancer solutions. And based on our specific peptide analysis, important ways to concoct such peptide involve careful sequence choosing, amino acid placement over just throwing in useful acids willy-nilly. However, you might want to include some Lysine!

Note also that a peptide is a smaller chunk of a protein. When we are analyzing the "sequence" variable, in this context we are analyzing more of a part than a whole. Therefore, the three-acid chunks we started identifying may tell us the active acid(s) in the successful peptides.

For transparency's sake, the peptides were tested on breast cancer cell lines as opposed to active human trials.

# Lung Cancer:


    """)

    # =========================================================
    # Lung Cancer and Peptide Analysis
    # =========================================================

    st.subheader("Lung Cancer and Peptide Data Analysis")

    # =========================================================
    # PCA Visualization: Lung Cancer Dataset
    # =========================================================
    st.subheader("3D PCA Scatter Plot (Lung Cancer Dataset)")

    # PCA Visualization Code
    lung_numeric = lungcancer.select_dtypes(include=['float64', 'int64'])
    columns_to_drop = ['ID', 'class_inactive - exp', 'class_inactive - virtual', 
                       'class_mod. active', 'class_very active']
    lung_numeric = lung_numeric.drop(columns=columns_to_drop, errors='ignore')
    lung_numeric = lung_numeric.dropna(axis=1, how='all')

    if lung_numeric.empty:
        st.warning("No numeric data available for PCA after handling missing values.")
    else:
        pca = PCA(n_components=3)
        lung_pca = pca.fit_transform(lung_numeric)
        lung_pca_df = pd.DataFrame(lung_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        lung_pca_df['Stage'] = lungcancer.get('Stage', pd.Series([None] * len(lung_pca_df)))

        fig = px.scatter_3d(
            lung_pca_df,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='Stage',
            title='3D PCA Scatter Plot (Lung Cancer)',
            labels={'Stage': 'Cancer Stage'}
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig)

    st.markdown("""
    The fact that the cancer stage markers (indicated by the colors) are spread all over without clear clustering implies that the variation in the data captured by PCA is not strongly aligned with the cancer stage labels.
    """)

    # =========================================================
    # Simplified t-SNE Visualization: Lung Cancer Dataset
    # =========================================================
    st.subheader("Simplified t-SNE Visualization (Lung Cancer Dataset)")

    # t-SNE Visualization Code
    if 'Stage_Numeric' not in lungcancer.columns:
        if 'Stage' in lungcancer.columns:
            stage_mapping = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
            lungcancer['Stage_Numeric'] = lungcancer['Stage'].map(stage_mapping)
        else:
            st.error("Neither 'Stage_Numeric' nor 'Stage' exists in the dataset. Cannot create t-SNE visualization.")
            lung_tsne_df = None

    if 'Stage_Numeric' in lungcancer.columns:
        lung_numeric = lungcancer.select_dtypes(include=['float64', 'int64'])
        lung_numeric = lung_numeric.drop(columns=columns_to_drop, errors='ignore')
        lung_numeric = lung_numeric.dropna(axis=1, how='all')

        # Simplified t-SNE: Use sampling and fewer iterations
        sample_size = min(500, len(lung_numeric))  # Cap at 500 samples for faster computation
        lung_sampled = lung_numeric.sample(n=sample_size, random_state=42)
        stage_sampled = lungcancer.loc[lung_sampled.index, 'Stage_Numeric']

        imputer = SimpleImputer(strategy='mean')
        lung_sampled_imputed = imputer.fit_transform(lung_sampled)
        scaler = StandardScaler()
        lung_sampled_scaled = scaler.fit_transform(lung_sampled_imputed)

        tsne = TSNE(n_components=2, perplexity=20, random_state=42, n_iter=500)  # Reduced perplexity and iterations
        lung_tsne = tsne.fit_transform(lung_sampled_scaled)

        lung_tsne_df = pd.DataFrame(lung_tsne, columns=['t-SNE1', 't-SNE2'])
        lung_tsne_df['Stage'] = stage_sampled.reset_index(drop=True)

        fig = px.scatter(
            lung_tsne_df,
            x='t-SNE1',
            y='t-SNE2',
            color='Stage',
            color_continuous_scale='Viridis',
            title='Simplified t-SNE Visualization of Lung Cancer Dataset',
            labels={'Stage': 'Cancer Stage'},
            hover_data=['Stage']
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(height=600, width=800)
        st.plotly_chart(fig)

    st.markdown("""
    This simplified t-SNE visualization represents a two-dimensional embedding of a sampled high-dimensional lung cancer dataset. Using a smaller subset of the data, it maintains the insights while significantly improving load times. The plot reveals a relatively uniform cloud of data points without clear clusters or patterns, suggesting that the dataset does not naturally group patients into well-defined categories based on the provided numerical features.
    """)

    # =========================================================
    # Heatmap: Correlation with Survival_Months
    # =========================================================
    st.subheader("Correlation Heatmap with Survival_Months")

    # Correlation Heatmap Code
    lung_numeric = lungcancer.select_dtypes(include=['float64', 'int64'])
    lung_numeric = lung_numeric.drop(columns=columns_to_drop, errors='ignore')
    lung_numeric = lung_numeric.dropna(axis=1, how='all')
    lung_numeric_imputed = imputer.fit_transform(lung_numeric)
    lung_numeric_imputed_df = pd.DataFrame(lung_numeric_imputed, columns=lung_numeric.columns)

    correlation_matrix = lung_numeric_imputed_df.corr()
    survival_corr = correlation_matrix[['Survival_Months']].sort_values(by='Survival_Months', ascending=False)

    fig = px.imshow(
        survival_corr,
        color_continuous_scale="Viridis",
        title="Correlation Heatmap of Features with Survival_Months",
        labels=dict(color="Correlation"),
        text_auto=True,
        aspect=0.25
    )
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Correlation with Survival_Months",
        height=700,
        width=1000
    )
    st.plotly_chart(fig)

    st.markdown("""
    Not counting the direct derivations of the target variable, Tumor Size, Smoking Interaction, Age, and phosphorus level seem to be strongest correlators. However, they're all very weak.
    """)

    # =========================================================
    # Peptide Analysis: Lung Cancer
    # =========================================================

    st.subheader("Peptide Analysis: Lung Cancer")

    # Create a copy of the peptides_l dataset
    peptides_copy = peptides_l.copy()

    # Ensure the required peptide class columns exist in the copied dataset
    required_columns = ['class_inactive - virtual', 'class_mod. active', 'class_very active']
    if not all(col in peptides_copy.columns for col in required_columns):
        if 'class' in peptides_copy.columns:
            # Re-encode the class column using one-hot encoding
            peptides_copy = pd.get_dummies(peptides_copy, columns=['class'], prefix='class', drop_first=False)

            # Check again if required columns exist after encoding
            missing_columns = [col for col in required_columns if col not in peptides_copy.columns]
            if missing_columns:
                st.error(f"Re-encoding failed. The following columns are still missing: {missing_columns}")
                st.stop()
        else:
            st.error("The 'class' column is missing from the peptides dataset. Cannot generate required columns.")
            st.stop()

    # Define global variables for active peptide classes
    inactive_peptides = peptides_copy[peptides_copy['class_inactive - virtual'] == 1]
    mod_active_peptides = peptides_copy[peptides_copy['class_mod. active'] == 1]
    very_active_peptides = peptides_copy[peptides_copy['class_very active'] == 1]

    # Function to calculate positional amino acid frequencies
    def positional_aa_frequencies(df, max_len=20):
        aa_position_matrix = np.zeros((max_len, 20))
        aa_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        for sequence in df['sequence']:
            for pos, aa in enumerate(sequence):
                if pos < max_len and aa in aa_index:
                    aa_position_matrix[pos, aa_index[aa]] += 1
        if len(df) > 0:
            aa_position_matrix /= len(df)
        return aa_position_matrix

    # Generate heatmaps for positional amino acid frequencies
    max_sequence_length = 20
    position_matrix_inactive = positional_aa_frequencies(inactive_peptides, max_len=max_sequence_length)
    position_matrix_mod_active = positional_aa_frequencies(mod_active_peptides, max_len=max_sequence_length)
    position_matrix_very_active = positional_aa_frequencies(very_active_peptides, max_len=max_sequence_length)

    # Function to plot heatmaps
    def plot_heatmap(position_matrix, title):
        fig = go.Figure(
            data=go.Heatmap(
                z=position_matrix,
                x=list("ACDEFGHIKLMNPQRSTVWY"),
                y=list(range(1, max_sequence_length + 1)),
                colorscale="Blues",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Amino Acids",
            yaxis_title="Position",
            width=800,
            height=600,
        )
        st.plotly_chart(fig)

    # Plot heatmaps for each peptide class
    plot_heatmap(position_matrix_inactive, "Positional Amino Acid Frequencies in Inactive Peptides")
    plot_heatmap(position_matrix_mod_active, "Positional Amino Acid Frequencies in Moderately Active Peptides")
    plot_heatmap(position_matrix_very_active, "Positional Amino Acid Frequencies in Very Active Peptides")

    # =========================================================
    # Advanced Visualization: t-SNE
    # =========================================================

    # Select numeric features and preprocess
    numeric_features = peptides_l.select_dtypes(include=['float64', 'int64']).drop(columns=['class_encoded'], errors='ignore')
    numeric_features = numeric_features.dropna(axis=1, how='all')  # Drop columns with all NaNs

    # Ensure the dataset has enough rows and columns
    if numeric_features.shape[0] < 2 or numeric_features.shape[1] < 2:
        # Duplicate the data to artificially increase its size
        while numeric_features.shape[0] < 10:  # Ensure at least 10 rows
            numeric_features = pd.concat([numeric_features, numeric_features], ignore_index=True)
        while numeric_features.shape[1] < 2:  # Ensure at least 2 features
            numeric_features[f'Extra_Feature_{numeric_features.shape[1]}'] = np.random.rand(numeric_features.shape[0])

    # Handle missing values and standardize
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
    numeric_features_imputed = imputer.fit_transform(numeric_features)
    numeric_features_scaled = StandardScaler().fit_transform(numeric_features_imputed)  # Standardize features

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(numeric_features_scaled)

    # Create DataFrame for visualization
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Class'] = pd.concat([peptides_l['class']] * (tsne_df.shape[0] // len(peptides_l) + 1), ignore_index=True)[:tsne_df.shape[0]]

    # Plot t-SNE
    fig = px.scatter(
        tsne_df,
        x='t-SNE1',
        y='t-SNE2',
        color='Class',
        title="t-SNE Visualization of Peptide Dataset",
        labels={'Class': 'Peptide Class'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_data=['Class']
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(height=600, width=800, title_font_size=20)
    st.plotly_chart(fig)


    st.markdown("""
For transparency's sake, it is important to be clear that the main lung cancer dataset (not the peptide data, however) was synthetically generated. It was engineered to be as close to reality as possible, but its synthetic nature will undoubtedly skew analysis. That being said, the results from the heatmap are unsurprising. It is not a stunner that most features on the heatmap have very poor correlation to survival duration when 90% of lung cancers are caused specifically by smoking, according to research from Johns Hopkins, leaving little room for other factors. Since peptides cannot be used to convince a person to quit their smoking habit, that cause is irrelevant to our research, leaving only the less common and minor causes behind. While some smoking-based variables do exist in the data and can be seen in the heatmap, their low correlation likely stems from their nature—Tumor_Size_Smoking_Interaction relates to smoking when the cancer has already developed, and Smoking_Pack_Years unfortunately contains some values that were generated improperly by the dataset's author (for example, in one entry, a 58-year-old patient has that variable set to 93.270893), making both of these variables poor predictors for their own reasons. The other variables have far less margin of error and can be taken more seriously.

If we were to remove smoking's immense real-life influence for a moment, our data suggests that, similarly to breast cancer, tumor size is an important indicator. This was expected prior to starting analysis based on the breast cancer data results.

But another real-life reason likely impacting which features impact survival is that lung cancer is harder to treat than breast cancer, and only around 20% of cases respond to immunotherapies. This makes it more challenging to identify effective options for treatment.

However, not all is lost!

For the peptide data, peptides classified as 'very active' exhibit distinct sequence motifs and amino acid preferences. Lysine ('K') is consistently overrepresented in highly active peptides, appearing prominently in positional frequency analyses and n-gram enrichment studies. Motifs like 'KWK' and 'KKK' are enriched in very active peptides, suggesting that positively charged residues and specific structural motifs may enhance activity. This pattern indicates that charge interactions, sequence alignment, and residue positioning are critical for optimizing peptide functionality. The presence of tryptophan ('W') in enriched motifs also suggests that hydrophobic and aromatic interactions may play a key role in binding efficacy.

We've seen much of this before in the breast cancer data. The continuing emergence of similar results is promising!

### Based on our analysis, a realistic plan for treatment could look something like this:

1. **Sequence Optimization**: Focus on incorporating motifs like 'KWK' and 'KKK' while maintaining a balance of hydrophobic and polar residues. This combination likely enhances structural stability and interaction specificity.

2. **Targeting Key Factors**: Design peptides to bind or modulate targets associated with tumor size or aggressive tumor locations (e.g., upper and lower lobes).

3. **Positional Importance**: Ensure that lysine and tryptophan residues are strategically positioned within peptides to optimize binding efficiency.

However, an additional cure presented by our new analysis is that the difficulties of finding a strictly medical cure for lung cancer do outline the importance of living a healthy lifestyle. The best way to prevent lung cancer is to avoid smoking, and in general, living healthier reduces cancer risk.

For the time being, anyway, no treatment is a substitute for a healthy lifestyle.
    """)


# References page
if page == "References":
    st.header("References")
    
    st.markdown("""

    1. **Wang, Z., & Zhang, J. (2021)**. [PMC8618108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8618108/)
    2. **Gonzalez-Moreno, J. et al. (2020)**. [PMC7071719](https://pmc.ncbi.nlm.nih.gov/articles/PMC7071719/#:~:text=At%20cell%20and%20disease%20levels,disease%20%2D%20kwashiorkor%20%5B32%5D.)
    3. **Keller, L., et al. (2019)**. [Annals of Oncology](https://www.annalsofoncology.org/article/S0923-7534(19)62291-X/pdf)
    4. **Johns Hopkins Medicine.** [Lung Cancer Risk Factors](https://www.hopkinsmedicine.org/health/conditions-and-diseases/lung-cancer/lung-cancer-risk-factors).
    5. **Medical News Today.** [How Lung Cancer Compares to Other Cancers](https://www.medicalnewstoday.com/articles/321817#:~:text=Each%20year%2C%20more%20people%20die,significantly%20lower%20than%20other%20cancers.).
    6. **Cancer Research UK.** [Can Cancer Be Prevented?](https://www.cancerresearchuk.org/about-cancer/causes-of-cancer/can-cancer-be-prevented-0#:~:text=Can%20I%20make%20sure%20I,a%20family%20history%20of%20cancer.).
    """)
