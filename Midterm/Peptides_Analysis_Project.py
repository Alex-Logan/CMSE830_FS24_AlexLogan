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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # needed for MICE
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error  # Import mean_squared_error for evaluation
from collections import Counter

# Set the title of the Streamlit app
st.title("Anti-Cancer Peptides: A Medical Hopeful?")
st.markdown("### A data research project by Alex Logan")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Introduction", "Data Loading", "Handling the Data", "Main Insights", "Closing Notes"])
# Sidebar - About This Section
st.sidebar.header("About This:")
st.sidebar.markdown("""
This dashboard is a condensed version of a documented dataset research project that can be viewed in full, all steps included, at the following link: [GitHub Project Link](https://github.com/Alex-Logan/CMSE830_FS24_AlexLogan/tree/main/Midterm)

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
except FileNotFoundError:
    # If the first attempt fails, try loading from the GitHub path
    try:
        breastcancer = pd.read_csv('Midterm/Raw_Data/breastcancer.csv')
        peptides_b = pd.read_csv('Midterm/Raw_Data/peptides_b.csv')
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
    In general terms, the purpose of this project is to determine how anti-cancer peptides may relate to fighting the common causes of cancer. However, there are many types of cancer, and so for the time being, this project's scope will be somewhat more limited (or if you prefer, focused), specifically to breast cancer. My datasets come from two sources:

    1. **Breast Cancer Dataset**
       - **Title:** Breast Cancer Wisconsin (Diagnostic) Data Set
       - **Source:** UCI Machine Learning Repository
       - **Data Format:** CSV
       - **Description:** This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes measurements like radius, texture, and smoothness, which are used to classify tumors as benign or malignant.
       - **Link:** [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

    2. **Anticancer Peptides Dataset**
       - **Title:** Anticancer Peptides Dataset
       - **Source:** UCI Machine Learning Repository
       - **Data Format:** CSV
       - **Description:** This dataset includes a collection of peptides and their anticancer activity against various types of cancer, providing valuable information for research in medicinal chemistry.
       - **Link:** [Anticancer Peptides Data](https://www.kaggle.com/datasets/uciml/anticancer-peptides-dataset)

    For the anticancer peptides dataset, the download actually includes datasets of peptides for two types of cancer, breast cancer and lung cancer. However, there wasn't a good way to involve the lung anticancer peptides with the project in its current state. A potential goal for future research, perhaps?

    The work involved in the project is two-fold: to examine the most common effects leading to breast cancer and also to examine the efficacy of peptides, with the ultimate goal being to find the most effective anti-cancer solutions. More specifically, what might make a peptide effective in the first place, and what would a "good" anticancer peptide do to the cells in order to be effective?

    Bringing the two goals together into a cohesive sum requires a bit of thinking, and we'll get to that soon. For now, there is some work to do to make this data more usable.

    While the information presented here will likely be of most usage to those in the medical community, I hope for the project to be understandable by all - I have tried to keep the language I use very casual!!
    """)

# Data Loading and Preparation page
elif page == "Data Loading":
    st.header("Data Loading and Preparation")
    
    # Load datasets
    

    # Display heads of both datasets
    st.subheader("Preview of Datasets")
    st.write("### Breast Cancer Dataset")
    st.write(breastcancer.head())
    st.write("### Peptides Dataset")
    st.write(peptides_b.head())

    st.markdown("""
    For the purposes of our analysis, one of the most important parts of the peptides dataset is the class column, which categorizes peptides into four classes based on their activity levels:

    - **Very Active:** EC/IC/LD/LC50 ≤ 5 μM
    - **Moderately Active:** EC/IC/LD/LC50 ≤ 50 μM
    - **Experimental Inactive**
    - **Virtual Inactive**

    For the inactive peptides, the only thing we care about at the moment is that they are **Inactive**.

    Active peptides are those that are either Very Active or Moderately Active, as they show significant anticancer activity. This will be important soon!

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
    We start with some encoding- in the interest of demonstrating something a bit more advanced, the peptides dataset is encoded with an if/else system that observes the data and picks one of the following based upon what is found:
    
    **Label Encoding:** If the class column had only two unique classes, a simple label encoding was applied, converting the classes into binary values (0 or 1).
    
    **One-Hot Encoding:** If there were more than two unique classes, one-hot encoding was applied to create separate binary columns for each class. This allows the model to interpret each class independently without imposing an ordinal relationship.
    
    The print statement that follows indicates that One-Hot encoding was applied, which isn't a surprise- the earlier IDA indicated that the dataset was multi-class. Still, you can never go wrong with building a more advanced multi-purpose method to handle more options, especially for the future.
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
    Ok, we've technically demonstrated one way to handle missingness now, but we'll talk on it a bit more in a moment. We already know there are no duplicates that need to be removed- the code to do so is provided for the sake of demonstrating how it would be done should the need arise in the final project.
    
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

    While the second part of our EDA (and, therefore, the conclusions of the project itself) will settle on a way forward and go from there, be sure to check out the final section of the project for some important discussion on alternative methods and thoughts on the project's application as a whole.

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
    Well that's interesting! Mean imputation, which replaces missing values with the mean of each feature, results in the highest MSE- by far. This is because mean imputation oversimplifies the data by ignoring the relationships between features, leading to distorted results. As a result, the imputed values significantly deviate from the true data, reflected in the MSE value of nearly 1- that's basically Completely Wrong.
    
    KNN imputation performs better, yielding a much lower MSE. This method looks at the nearest neighbors in the dataset and fills in missing values based on patterns in similar data points. KNN captures local relationships between variables, which leads to a more accurate estimation of the missing values compared to mean imputation. However, it may still fall short in capturing broader patterns or when data is sparse, but overall, it actually does quite good!
    
    The winner, however, is MICE, which produces the lowest MSE among the methods, suggesting it offers the most accurate imputation. By iteratively modeling missing data using all other features, MICE maintains the relationships between variables more effectively than both mean and KNN imputation. This iterative process allows for better handling of complex datasets, ensuring that the imputed values closely resemble the true data. MICE’s ability to account for interactions and variability in the dataset results in the most reliable imputation, reflected in its minimal MSE, making it the best option in this comparison.
    
    As interesting as this all is, it's probably best for the sake of this project to just stick with the real data since it's already so clean! So why discuss the missingness at all?
    
    First of all, data is not always going to be this clean. It's practically a given that further cancer research (and possibly the final version of this very project) will include missingness. By doing this now, we've learned what the best method for imputation is- and that's a very good thing to know. Additionally, studying the effects of introduced missingness on the overall dataset can help identify which variables are crucial for accurate imputation and model performance. This process can highlight key features that may strongly influence the outcome variable and are, therefore, critical for predictive modeling.
    """)

    # Basic statistical summary of the trimmed breast cancer dataset
    st.write("Statistical Summary of Selected Features DataFrame:")
    st.write(selected_features_df.describe())

    # Basic statistical summary of the peptides dataset
    st.write("\nStatistical Summary of Peptides DataFrame:")
    st.write(peptides_b.describe())


# Main Insights section
elif page == "Main Insights":
    st.header("Main Insights")

    # Content explaining peptide sequences
    st.markdown("""
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
    Here's the peptide class distribution visualized, once again making use of encoding in visualization- I say "once again" because all of the visualizations involving diagnosis in the breast cancer dataset leverage the fact that that variable was encoded. Does this bar chart perhaps indicate I should've used SMOTE? Check the end of the notebook for my answer.
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
    """)

# Closing Notes page
elif page == "Closing Notes":
    st.header("Closing Notes")
    
    st.markdown("""
    Let's address the elephant in the room here: this project is not perfect!

    The main challenge was finding a meaningful way to **incorporate the datasets together**- outside the type of cancer, there wasn't a meaningful column to merge on. Because of that, assumptions made about the peptides' potential effects on the causes of breast cancer from a completely different dataset have to be taken as **hypothetical**- though there **is** some reason to believe they are worth considering- namely that we are looking **at the same type of cancer in both datasets** - these are anticancer peptides for **breast cancer** so if they work, it's not entirely unreasonable to suggest certain potential conclusions based on what we know about breast cancer. I'd have liked an even stronger leg to stand on, but making data science projects out of merging separate datasets together isn't easy! **Aside from some visualizations at the end**, the merged dataset never really got used- there never really ended up being a **meaningful reason to have it** and it was easier to just keep things separate for analysis. This project taught me many valuable lessons on how far we can expect traditional data science to take us on its own- and how challenging it is to do truly meaningful research! No wonder we've all heard that sobering statistic that the large majority of data science projects allegedly fail.

    Luckily, I don't think this project can be called a failure- we seem to have arrived at some useful information, at least. Some after-the-fact googling shows that other studies suggest Lysine to be effective in cancer fighting: 

    ## References

    1. **Wang, Z., & Zhang, J. (2021)**. [PMC8618108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8618108/)
    2. **Gonzalez-Moreno, J. et al. (2020)**. [PMC7071719](https://pmc.ncbi.nlm.nih.gov/articles/PMC7071719/#:~:text=At%20cell%20and%20disease%20levels,disease%20%2D%20kwashiorkor%20%5B32%5D.)
    3. **Keller, L., et al. (2019)**. [Annals of Oncology](https://www.annalsofoncology.org/article/S0923-7534(19)62291-X/pdf)

    The bottom source specifically discusses breast cancer, which means we've hopefully made some useful headway after all!

    However, it's less clear whether or not we were onto quite as much with the other specific amino acids. Earlier in the project I refused to use SMOTE because I was worried about too much fake data. But those **same class imbalances** made things much harder to analyze in the end when it came to the peptides, especially ones that weren't Lysine. Lesson learned for the future- SMOTE has its place! Ultimately, I think I should've stuck with it, but it was perhaps even more valuable to learn **why** I should have.

    Introducing artificial missingness was another method I ultimately chose to not pursue for the final analysis, but throwing this wrinkle into the data could've made certain key observations more prevelant if they remained appearing despite the missingness. Once again, a valuable lesson to learn.

    In the end, I think the project still managed to prove mostly beneficial.
    """)

