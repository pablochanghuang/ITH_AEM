# ITH_AEM
EOSC 556 Final Project
Pablo Chang Huang

The Inuvik-Tuktoyaktuk Highway (ITH) is a 138 km all-season highway that opened in 2017 [1] in the north of the Northwest Territories under a contract with the Department of Infrastructure. It is the first Canadian highway connecting to the Arctic Ocean. This highway is essential for the residents of Tuktoyaktuk for tourism, commerce, and business purposes. Due to the regional conditions, this highway is at risk of being affected by permafrost, perennially frozen ground, degradation.
In this context, data on permafrost layers are essential for the planning and design of resilient infrastructure [2]. Geophysics offers valuable insight through efficiently collected data that can be used to map the permafrost boundaries along the highway. In this sense, Airborne Electromagnetic (AEM) methods are widely used to map subsurface electrical conductivity for large-scale problems like this. Due to the shallow depth resolution, a multi-frequency domain electromagnetic (FDEM) survey was conducted in August 2024 along and across the road using the FDEM Resolve 6 system, covering a total of 1,142 km in flight lines. Additionally, complemen- tary datasets from the same region—such as surficial geology, electrical resistivity tomography, and airborne amplitude radar (InSAR)—provide further insights.

The main goal is to perform 1D sparse inversions of single soundings along selected lines in Block 53 of the ITH data, comparing the results with previous inversions conducted by Marco Couto from the GIF group. Additionally, if time permits, I will explore different methods for integrating supporting datasets to improve the inversion results. This may include incorporating surface constraints or even performing joint inversions with InSAR data. A key challenge is accurately mapping the shallow permafrost areas, making the inclusion of complementary datasets essential for refining the results. These additions also lay the groundwork for future research directions.

**Setting up environment:**
This project uses a custom Conda environment to ensure all required packages are available for geophysical simulation, data analysis, and visualization. To create and activate the environment:
conda env create -f ITH_AEM_Project.yml
conda activate ITH_AEM_Project


**Project Contents:**
* 
