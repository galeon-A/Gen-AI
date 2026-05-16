# AI and Data Engineering Projects by Atharva Kulkarni  

Welcome to my GitHub repository! Here, you'll find a collection of projects showcasing my expertise and passion for Artificial Intelligence, Machine Learning, and Deep Learning. These projects demonstrate my skills in building predictive models, exploring generative AI applications, and implementing advanced deep learning techniques.  

---

## 📁 Project Categories    

### 🤖 **Generative AI**  
- Custom language models for text generation and summarization.  
- Exploration of cutting-edge models like GPT and diffusion models.  
- Creative projects involving AI-generated art and text.  

### 🖥️ **Deep Learning**  
- Neural network architectures for image and speech recognition.  
- Computer vision projects using Convolutional Neural Networks (CNNs).  
- Experimentation with cutting-edge frameworks like TensorFlow and PyTorch.


  
## Data Engineering

# Engine Telemetry Data Quality Pipeline

## Overview
Designed and implemented an end-to-end PySpark + Databricks ETL pipeline that processes simulated IoT engine telemetry from 10K+ units across CSV/JSON sources.

## Business Problem
Cummins-style engine fleets generate dirty, incomplete, and duplicate telemetry data. Poor data quality can lead to:
- Incorrect predictive maintenance
- Engine downtime
- Warranty cost inflation
- Sensor reliability issues

## Solution
Built a medallion architecture:
### Bronze:
Raw ingestion from CSV + JSON

### Silver:
Data cleansing, deduplication, validation

### Gold:
Data quality KPIs + dashboard metrics

---

## Tech Stack
- Python
- PySpark
- Databricks
- Delta Lake
- Great Expectations
- SQL
- Faker
- Pandas

---

## Key Features
- Simulated 100K+ telemetry records
- 15% null injection
- 10% duplicate injection
- 5% outlier injection
- Multi-source schema normalization
- Delta Lake Bronze/Silver/Gold
- DQ dashboard

---

## Impact
- Engineered scalable ETL for industrial IoT
- Reduced bad telemetry propagation by >95%
- Built operational data quality monitoring

## Future Enhancements
- Kafka streaming ingestion
- Airflow orchestration
- ML anomaly detection
- Unity Catalog governance 

---

## 📌 Highlights  
- 🌟 **Large Language Models (LLMs):** Development and fine-tuning of LLMs for specialized applications.  
- 📈 **Data Visualization:** Interactive dashboards and insightful visualizations for AI model performance.  
- 🌌 **Innovation:** Pioneering ideas in generative models and AI-powered creativity.  

---

## 🚀 Tech Stack  
- **Programming Languages:** Python.  
- **Frameworks and Libraries:** TensorFlow, PyTorch, Scikit-learn, Hugging Face Transformers.  
- **Tools:** Streamlit, Jupyter Notebook, Google Colab, Docker.  

---

## 📫 Contact  
Feel free to reach out if you'd like to discuss any of my projects or collaborate:  
- **Email:** [atharvakulkarni329@gmail.com]  
   

---

### ⭐ Star this repository if you find my work interesting and useful!  

