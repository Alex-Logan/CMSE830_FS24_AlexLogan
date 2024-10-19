# Anti-Cancer Peptides Analysis Project

This project aims to explore the relationship between anti-cancer peptides and their effectiveness in combating breast cancer. It focuses on two main aspects: identifying the primary factors contributing to breast cancer and assessing the efficacy of various peptides. The ultimate goal is to discover the most effective anti-cancer solutions by understanding the characteristics that make a peptide successful and how such peptides can influence cancerous cells for effective treatment.

## Project Overview

For this project, the file(s) inside the `Project_Notebook.zip` file are very useful to look through. It is helpful to view the notebook (in either its `.ipynb` or `.html` form; reading both would likely be redundant) to fully understand this project! This project had a rubric for its development: the notebook makes it very clear specifically what I have done to address each step of that rubric, minus the GitHub part (which is covered by this repository) and the Streamlit part (which is of course covered by the Streamlit app itself).

The Streamlit app is essentially a simplified, distilled version of what is present in the notebook. The link to that app is here - [Streamlit App Link](#).

The `.ipynb` (Jupyter Notebook) file is somewhat large. I uploaded a `.zip` file so that 1.) the `.ipynb` file could be uploaded to this repo at all, and 2.) so that I could include an `.html` file version of the notebook if you do not have a particular interest in running the Jupyter code cells yourself (likely isn't really necessary for this project). If you have any errors with the `.ipynb`, just take a look at the `.html` instead; it includes all the same information.

Make sure you have Jupyter installed on your machine if you wish to run the notebook locally. You may also encounter errors while running the notebook if you have not yet installed certain Python packages—this can generally be resolved by running `pip install 'package name'` in an empty code cell, then reloading the file. The `.html` file, on the other hand, should be completely self-contained and not require you to install anything to view it. You can find a more specific list of Python requirements inside the `requirements.txt` file.

## Midterm Section Overview

This "Midterm" section of the repo has a few important components:

### Folders:

- **`.streamlit`**: This folder stores the Streamlit app's config file so that the app has some cool color customization options. You don't need to worry much about it.
  
- **`Detailed_Instructions`**: Contains more formal documentation on how to install and run the project if you wish to do so locally.
  
- **`Raw_Data`**: Contains the raw datasets the project needs to run and is built off of. The cleaning, processing, altering, etc. of these datasets is done within the code of the app/notebook itself.

### Files:

- **`Peptides_Analysis_Project.py`**: This is the code of the Streamlit app itself—it's what's being run and opened when you access the app.

- **`Project_Notebook.zip`**: A zip file containing both a Jupyter Notebook (`.ipynb` file) that shows every step of the project in more detail, including the programming, and an `.html` version of that same notebook.

- **`requirements.txt`**: The packages the app needs to work.

- **`README.md`**: The readme file that you are reading right now!
