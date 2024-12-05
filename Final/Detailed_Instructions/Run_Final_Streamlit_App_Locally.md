# Running the App Locally - A Simple Guide

## What You Need

Before diving in, ensure you have:

- **Python** installed (version 3.12.6).
- Additionally, install Jupyter if you wish to open the `.ipynb` notebook file.

### Essential Packages

You’ll need the following libraries to get everything running smoothly:

- `matplotlib==3.8.4`
- `numpy==1.26.1`
- `pandas==2.2.3`
- `plotly==5.22.0`
- `scikit_learn==1.4.2`
- `scipy==1.14.1`
- `seaborn==0.13.2`
- `streamlit==1.32.0`
- `toml==0.10.2`

## Let’s Set It Up

1. **Bring the Project to Your Machine**:  
   Start by cloning the repository to your local system with this command:
   ```bash
   git clone https://github.com/Alex-Logan/CMSE830_FS24_AlexLogan.git
   cd CMSE830_FS24_AlexLogan/Final
   ```

2. **Install What You Need**:  
   Use the provided `requirements.txt` file to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
   If you prefer to install them individually, you can do so with the following command:
   ```bash
   pip install matplotlib==3.8.4 numpy==1.26.1 pandas==2.2.3 plotly==5.22.0 scikit_learn==1.4.2 scipy==1.14.1 seaborn==0.13.2 streamlit==1.32.0 toml==0.10.2
   ```

3. **Prepare Your Data**:  
   Make sure that the necessary data files (`breastcancer.csv`, `peptides_b.csv`, etc.) are available in the appropriate folder. If they’re not in the repository, you’ll need to download them.

4. **Time to Launch**:  
   Now that everything is ready, kick off the Streamlit app with:
   ```bash
   streamlit run ResearchApp.py
   ```

5. **Explore The App**:  
   Once the command is executed:
   - Your Streamlit dashboard should open automatically in your web browser at `http://localhost:8501`.
   - If it doesn't, simply navigate to `http://localhost:8501` in your browser to view the dashboard.
