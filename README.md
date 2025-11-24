
Income Inequality in Developing Nations ( Challenges and Solutions )
# 💵 Income Inequality Prediction - Machine Learning Project

**Course:** Artificial Intelligence & Data Science  
**Instructor:** Miss Aqsa Moiz  
**Student:** Zahabia Ahmed  
**Institution:** S.M.I.T (Saylani Mass IT Training)  
**Project Type:** Machine Learning Assignment

## 📋 Project Overview

This machine learning project addresses the growing problem of income inequality by developing predictive models that can determine whether an individual's income is above or below a specific threshold. The project was assigned as part of the AI and Data Science curriculum at S.M.I.T under the guidance of Miss Aqsa Moiz.

### 🎯 Problem Statement
**Binary Classification Task**: Predict the target feature `income_above_limit` which indicates whether an individual earns above or below a certain amount.

**Primary Evaluation Metric**: F1-Score

## 🏗️ Project Structure

```
income_inequality_predictor/
│
├── 📊 Google Colab Files/
│   ├── Income_Inequality_Solution.ipynb  # Provided 
│   └── Income_Inequality_Prediction.ipynb # My Implementation
│
├── 💻 Local Implementation/
│   ├── app.py                          # Main Streamlit Application
│   ├── train_model.py                  # Model Training Script
│   ├── requirements.txt                # Python Dependencies
│   ├── income_data.csv                 # Dataset
│   ├── income_predictor_model.pkl      # Trained Model
│   └── model_info.pkl                  # Model Metadata
│
└── 📚 Documentation/
    ├── README.md
    └── Project_Report.pdf
```

## 📁 File Descriptions

### Google Colab Notebooks
- **`Income_Inequality_Solution.ipynb`** - Original assignment notebook provided by Miss Aqsa Moiz containing problem statement, dataset description, and requirements
- **`Income_Inequality_Prediction.ipynb`** - Complete implementation by student Zahabia Ahmed with data preprocessing, model training, and evaluation

### Local Application Files
- **`app.py`** - Interactive Streamlit web application for income prediction
- **`train_model.py`** - Script to train machine learning models on the dataset
- **`requirements.txt`** - List of Python packages required to run the project
- **`income_data.csv`** - Dataset containing demographic and employment features
- **`income_predictor_model.pkl`** - Serialized trained model for predictions
- **`model_info.pkl`** - Metadata about the trained model and features

## 🚀 Quick Start

### Google Colab & VS code Implementation
1. Open the provided Colab notebook in Google Colaboratory
2. Upload the dataset to your environment
3. Run all cells sequentially to:
   - Load and preprocess data
   - Train machine learning models
   - Evaluate model performance
   - Generate predictions

### Local Implementation
```bash
# 1. Clone or download the project files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (ensure income_data.csv is in directory)
python train_model.py

# 4. Launch the web application
streamlit run app.py
```

## 🎯 Model Details

### Algorithms Implemented
- **Random Forest Classifier** - Primary model with hyperparameter tuning
- **Additional models** total no. of Model implemented 07 as per assignment requirements

### Features Used
- Demographic information (age, gender, education, marital status)
- Employment details (occupation, work class, hours per week)
- Economic factors (capital gain, capital loss)
- Geographic and migration information

### Performance Metrics
- **Primary**: F1-Score (harmonic mean of precision and recall)
- **Secondary**: Accuracy, Precision, Recall, ROC-AUC

## 📊 Dataset Information

The dataset contains 43 columns with mixed data types including:
- **Target Variable**: `income_above_limit` (Binary: Below limit/Above limit)
- **Features**: Demographic, employment, economic, and geographic attributes
- **Size**: Multiple records with real-world income distribution patterns

## 🌟 Key Features

### Web Application (`app.py`)
- **Single Prediction**: Interactive form for individual income prediction
- **Batch Processing**: CSV upload for multiple predictions
- **Real-time Results**: Instant prediction with confidence scores
- **Visual Analytics**: Charts and graphs for model interpretation
- **User-Friendly Interface**: Streamlit-based responsive design

### Model Training (`train_model.py`)
- **Data Preprocessing**: Handling missing values and categorical encoding
- **Feature Engineering**: Standardization and transformation
- **Model Selection**: Algorithm comparison and hyperparameter tuning
- **Performance Evaluation**: Comprehensive metrics and validation
- **Model Persistence**: Save/load trained models for deployment

## 🛠️ Technical Stack

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Web Framework**: Streamlit
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Model Serialization**: Joblib
- **Development Environment**: Google Colab & VS Code

## 📈 Learning Outcomes

This project demonstrates proficiency in:

1. **Data Preprocessing** - Handling real-world datasets with missing values and mixed data types
2. **Feature Engineering** - Transforming raw data into meaningful model inputs
3. **Model Selection** - Choosing appropriate algorithms for binary classification
4. **Hyperparameter Tuning** - Optimizing model performance through grid search
5. **Model Evaluation** - Using appropriate metrics for imbalanced classification
6. **Web Deployment** - Creating interactive applications for model demonstration
7. **Project Documentation** - Professional reporting and code organization

## 👩‍🏫 Instructor Requirements

 
- Comprehensive problem statement and dataset
- Clear evaluation criteria and success metrics
- F1 Score on machine learning best practices
- Streamlit application (if possible)

## 👩‍🎓 Student Implementation

**Zahabia Ahmed** delivered:
- Complete data preprocessing pipeline
- Multiple machine learning model implementations
- Comprehensive model evaluation and comparison
- Interactive web application for demonstration
- Professional documentation and code comments

## 🎓 Academic Context

This project was developed as part of the **Artificial Intelligence and Data Science** curriculum at **S.M.I.T (Saylani Mass IT Training)**, representing the practical application of machine learning concepts taught in the course.

## 📝 Usage Instructions

### For Educators/Reviewers
1. Examine the Colab notebooks for implementation details
2. Review the model training process and evaluation metrics
3. Test the Streamlit application with sample data
4. Check the code quality and documentation

### For Students
1. Study the data preprocessing techniques
2. Understand the model selection process
3. Learn how to evaluate classification models
4. See practical implementation of Streamlit for ML apps

## 🔮 Future Enhancements

Potential improvements identified during development:
- Integration of more advanced ensemble methods
- Real-time model performance monitoring
- Additional visualization dashboards
- API development for external integrations
- Automated model retraining pipeline

## 📞 Contact Information

* **Student**: **Zahabia Ahmed**  
* **Course**: **AI and Data Science**  
* [**GitHub**]( https://github.com/Zahab163)
* [**Facebook**](https://www.facebook.com/share/1KBwSz91no/)
* [**Instagram**](https://www.instagram.com/zahabiaahmed?igsh=MXkwNzkzdGJsMzJqOA==)
* [**YouTube**](https://www.youtube.com/@ZahabiaAhmed)
* [**X-Twitter**](https://x.com/AhmedZahabia?t=yAAjSTYTwRRQsXCeomBMuQ&s=08)
* [**WhatsApp**](+923323924734)
* [**Gmail**](zahabia0ahmed@gmail.com)
* [**Gmail2**](bintesahmed8@gmail.com)
* **Zoom**: **Zahabia Ahmed**
* **Teams**: **Zahabia Ahmed**




## 🙏 Acknowledgments

- **Miss Aqsa Moiz** for guidance and assignment design
- **S.M.I.T** for providing the learning platform and resources
- **Open-source community** for the Python libraries and tools used
- **Dataset providers** for the real-world income data
- **Zahabia Ahmed** sound like giving self credit ( Deserved for the completion and survival). Keep going Zahabia there's always a work in pending and in progress. Wish me Luck!
---

<div align="center">

### 🎓 Machine Learning Assignment  
### 💻 Artificial Intelligence & Data Science Course  
### 👩‍🎓 Implemented by Zahabia Ahmed  


</div>
