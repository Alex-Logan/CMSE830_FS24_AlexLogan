# Data Dictionary 

## 1. Breast Cancer Dataset
- **File Name:** `breastcancer.csv`
- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Description:** 
  - This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
  - The features describe characteristics of the cell nuclei present in the image.
  - The target variable is `diagnosis`, indicating whether the tumor is benign (B) or malignant (M).
- **Attributes:**
  - `ID`: Unique identifier for each observation.
  - `radius_mean`, `texture_mean`, `perimeter_mean`, etc.: Mean, standard error, and worst values for various measurements.
  - `diagnosis`: Target label (B for benign, M for malignant).

## 2. Peptides Dataset (Breast Cancer)
- **File Name:** `peptides_b.csv`
- **Source:** [Anticancer Peptides Dataset](https://www.kaggle.com/datasets/uciml/anticancer-peptides-dataset)
- **Description:**
  - Contains a collection of peptides and their anticancer activity.
  - Peptides are labeled based on their activity levels.
- **Attributes:**
  - `ID`: Unique peptide identifier.
  - `sequence`: Amino acid sequence of the peptide.
  - `class`: Activity classification of the peptide (e.g., `very active`, `inactive`, etc.).

## 3. Lung Cancer Dataset
- **File Name:** `lungcancer.csv`
- **Source:** [Lung Cancer Prediction Data Set](https://www.kaggle.com/datasets/rashadrmammadov/lung-cancer-prediction)
- **Description:**
  - Includes features relevant to lung cancer prediction, such as smoking habits, alcohol consumption, and age.
  - The target variable indicates the presence or absence of lung cancer.
- **Attributes:**
  - `ID`: Unique identifier.
  - `Age`, `Smoking_Pack_Years`, `Alcohol_Consumption`: Lifestyle-related features.
  - `Stage`: Cancer stage (Stage I–IV).

## 4. Peptides Dataset (Lung Cancer)
- **File Name:** `peptides_l.csv`
- **Source:** [Anticancer Peptides Dataset](https://www.kaggle.com/datasets/uciml/anticancer-peptides-dataset)
- **Description:**
  - Similar to `peptides_b.csv`, but with a focus on lung cancer peptides.
  - Includes additional processed columns.
- **Attributes:**
  - `ID`: Unique peptide identifier.
  - `sequence`: Amino acid sequence.
  - `class_inactive - virtual`, `class_mod. active`, `class_very active`: One-hot encoded activity classes.

---

## Modified/Processed Datasets

### 1. `improved_lungcancer_dataset.csv`
- **Description:** Preprocessed version of the original lung cancer dataset with added feature engineering.
- **Modifications:**
  - Features such as `Tumor_Size_Smoking_Interaction` and polynomial terms for survival months.
  - Target encoding for `Stage`.

### 2. `improved_peptides_dataset.csv`
- **Description:** Preprocessed version of the lung cancer peptides dataset.
- **Modifications:**
  - Dimensionality reduction via PCA.
  - Additional class encoding.

### 3. `cleaned_integrated_lungcancer_peptides_advanced.csv`
- **Description:** Integrated dataset combining lung cancer and peptides datasets.
- **Modifications:**
  - Combined and aligned on shared cancer-related attributes.
  - Feature engineering applied to the integrated dataset.

### 4. `improved_breastcancer_dataset.csv`
- **Description:** Preprocessed version of the breast cancer dataset.
- **Modifications:**
  - Normalization and encoding of categorical variables.
  - Removal of missing values.

### 5. `improved_peptides_b_dataset.csv`
- **Description:** Preprocessed version of the breast cancer peptides dataset.
- **Modifications:**
  - One-hot encoding for activity classes.
  - Sequence-based feature extraction.

### 6. `transformed_lungdatamerged.csv`
- **Description:** Further processed version of the integrated lung cancer dataset.
- **Modifications:**
  - Additional transformations such as clustering and dimensionality reduction.

---
