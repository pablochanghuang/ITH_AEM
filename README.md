**"Sounding inversions from the Inuvik-Tuktoyaktuk Highway (ITH) Airborne Electromagnetics (AEM) survey"**

EOSC 556 Final Project

Pablo Chang Huang

The Inuvik-Tuktoyaktuk Highway (ITH) is a 138 km all-season highway that opened in 2017 [1] in the north of the Northwest Territories under a contract with the Department of Infrastructure. It is the first Canadian highway connecting to the Arctic Ocean. This highway is essential for the residents of Tuktoyaktuk for tourism, commerce, and business purposes. Due to the regional conditions, this highway is at risk of being affected by permafrost, perennially frozen ground, degradation.
In this context, data on permafrost layers are essential for the planning and design of resilient infrastructure [2]. Geophysics offers valuable insight through efficiently collected data that can be used to map the permafrost boundaries along the highway. In this sense, Airborne Electromagnetic (AEM) methods are widely used to map subsurface electrical conductivity for large-scale problems like this. Due to the shallow depth resolution, a multi-frequency domain electromagnetic (FDEM) survey was conducted in August 2024 along and across the road using the FDEM Resolve 6 system, covering a total of 1,142 km in flight lines. Additionally, complemen- tary datasets from the same region—such as surficial geology, electrical resistivity tomography, and airborne amplitude radar (InSAR)—provide further insights.

The main goal is to perform 1D sparse inversions of single soundings along selected lines in Block 53 of the ITH data, comparing the results with previous inversions conducted by Marco Couto from the GIF group. Additionally, if time permits, I will explore different methods for integrating supporting datasets to improve the inversion results. This may include incorporating surface constraints or even performing joint inversions with InSAR data. A key challenge is accurately mapping the shallow permafrost areas, making the inclusion of complementary datasets essential for refining the results. These additions also lay the groundwork for future research directions.

### **Setting up environment:**
This project uses a custom Conda environment to ensure all required packages are available for geophysical simulation, data analysis, and visualization. To create and activate the environment:
conda env create -f ITH_AEM_Project.yml
conda activate ITH_AEM_Project

- **`environment.yml`**: Environment configuration file to set up all necessary dependencies as described above. Enables running all Jupyter notebooks included in the project.


### **Project Contents:**

1. `1D_FDEM_REALDATA_least_square_inversion.ipynb`**: Notebook with a cleaner workflow to read and plot the original data. Data is separated by lines and soundings in Block 53 of the described survey, read from a CSV file (provided in the Google Drive link below). Runs the original L2-norm inversion by Marco Couto.

2. `1D_Forward_Simulation.ipynb`**: Forward simulation of a 1D model with features that mimic the expected permafrost structure from the project. The survey design is set up to match the one described above.

3. `1D_Simulation_Inversion.ipynb`**: L2 and sparse inversion performed on the synthetic data generated from the forward simulation. *Note: there is currently a bug affecting the inversion results.*


**Complementary:**
- **`fdem_1d_least_square_inversion_original.ipynb`**: Original inversion workflow notebook by Marco Couto. May not run correctly due to format differences and experimental code.


### **Data Drive:**

Real data csv file: https://drive.google.com/file/d/1Y1yTpyrHJLez2_wiJxF1hLWxP7nustEx/view?usp=share_link



### **References:**

[1] Department of Infrastructure, Government of Northwest Territories, "Inuvik Tuktoyaktuk Highway Project," [Online]. Available: \url{https://www.inf.gov.nt.ca/en/ITH}. [Accessed: Mar-2025].


[2] ARCUS, "How is permafrost degradation affecting infrastructure?", *ARCUS*, [Online]. Available: \url{https://www.arcus.org/search-program/arctic-answers/permafrost-and-infrastructure/briefs}. [Accessed: Mar-2025].
