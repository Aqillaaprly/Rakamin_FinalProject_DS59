# 📚 Final Project: Model Development & Deployment Documentation (DS 59 Group 2)

This repository serves as the central hub for documentation and core artifact storage for the Data Science Final Project (DS 59) Group 2. Its purpose is to ensure that the model development and deployment process is structured, reproducible, and well-documented.

## 👥 Team Members

* **Mercy Claudia Menayang (Project Manager):** Responsible for setting project goals, managing timelines, coordinating deployment, and ensuring the project delivers on time and meets business objectives.

* **Jason Williem Candra (Business & Data Analyst):** Responsible for defining business problems, setting analytical goals, translating data insights into actionable business recommendations, and ensuring results align with stakeholder needs.

* **Sigit Suhardyadi (Data Engineer):** Responsible for handling data preparation and integration, cleaning, transforming, combining data from various sources, and performing Exploratory Data Analysis (EDA) to find insights.

* **Aqilla Aprilly Kurnia Sari (Data Scientist):** Responsible for developing and optimizing analytical models, building and testing machine learning models according to project needs, and validating performance through collaboration with the Data Engineer.



##  Repository Structure

This repository is organized into several key folders to facilitate easy navigation and management of project assets:

### 📂 `stages/`

This folder is used to save all documentation and code related to every stage of model development, ranging from data preprocessing and model training to experiment evaluation.

* **Purpose:** To store the code that reflects the sequential workflow (pipeline) of Model Development and Experimentation.

### 📂 `models/`

This folder stores the model files that are **ready for deployment**. The models saved here are the best-performing models that have passed through the validation and performance comparison stages.

* **Purpose:** To house the trained model artifacts that will be utilized in the production/deployment phase.

### 📂 `views/`

This folder contains all the interface pages that are shown during the deployment stage. This includes the code for the web application's front-end or other user interfaces.

* **Purpose:** To save the source code for the visual interface presented to the end-users.

### **`app.py`:** 
The main execution script (Streamlit deployment) that connects the models and the `views/` interface.
### **`recruitment_efficiency_improved.csv`:** 
The core dataset used for training and testing the models.
### **`requirements.txt`:** 
Lists all necessary Python libraries and their specific versions to ensure the project runs correctly in any environment.
