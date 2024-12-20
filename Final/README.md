# Anti-Cancer Peptides Analysis Project

This project aims to explore the relationship between anti-cancer peptides and their effectiveness in combating breast cancer. It focuses on two main aspects: identifying the primary factors contributing to breast cancer and assessing the efficacy of various peptides. The ultimate goal is to discover the most effective anti-cancer solutions by understanding the characteristics that make a peptide successful and how such peptides can influence cancerous cells for effective treatment.

## Project Overview

For this project, the file(s) inside the `Project_Notebook.zip` file are very useful to look through. It is helpful to view the notebook to fully understand this project! This project had a rubric for its development: the notebook makes it very clear specifically what I have done to address each step of that rubric, minus the GitHub part (which is covered by this repository) and the Streamlit part (which is of course covered by the Streamlit app itself). Additionally, the notebook contains some additional visualizations, sections, discussion etc. that either didn't play nice with Streamlit integration, or simply bloated the app too much. I **highly** reccomend viewing the notebook to see this project in its true entirety.

The Streamlit app is essentially a simplified, distilled version of what is present in the notebook. The link to that app is here - https://alexlogan-peptidesresearch.streamlit.app/

This is the link to the final version of the project. For archival purposes, the midterm version is preserved on its own separate link which you can find in the appropriate folder.

The `.ipynb` (Jupyter Notebook) file is somewhat large. I uploaded a `.zip` file so that the `.ipynb` file could be uploaded to this repo at all. For the Midterm, I had previously included an `.html` file version of the notebook if the end user did not have a particular interest in running the Jupyter code cells themself (likely isn't really necessary for this project). Unfortunately, plots weren't playing nice with the `.html` this time around, so is just the `.ipynb` this time.

Make sure you have Jupyter installed on your machine if you wish to run the notebook locally. You may also encounter errors while running the notebook if you have not yet installed certain Python packages—this can generally be resolved by running `pip install 'package name'` in an empty code cell, then reloading the file. You can find a more specific list of Python requirements inside the `requirements.txt` file.

## Final Section Overview

This "Final" section of the repo has a few important components:

### Folders:

- **`.streamlit`**: This folder stores the Streamlit app's config file so that the app has some cool color customization options. You don't need to worry much about it.
  
- **`Detailed_Instructions`**: Contains more formal documentation on how to install and run the project if you wish to do so locally. 
  
- **`Raw_Data`**: Contains the raw datasets the project needs to run and is built off of. The cleaning, processing, altering, etc. of these datasets is done within the code of the app/notebook itself, however, the underlying code re-loads some of those cleaned datasets when necessary due to the need for archival versions of the datasets at certain points in the demonstration, so the folder contains all of the various modified versions as well.

- **`Data_Dictionary`**: Contains a **data dictionary file** that I **reccomend reading if you want to fully understand the folder's contents**.

- **`Modeling_Info`**: Documentation on the modeling approaches used in this project.

### Files:

- **`ResearchApp.py`**: This is the final code of the Streamlit app itself—it's what's being run and opened when you access the app. The filename no longer refers to it as a project.

- **`Project_Notebook_Final.zip`**: A zip file containing both a Jupyter Notebook (`.ipynb` file) that shows every step of the project in more detail, including the programming. In the Midterm folder, you can find an older version of this.

- **`requirements.txt`**: The packages the app needs to work.

- **`README.md`**: The readme file that you are reading right now!
