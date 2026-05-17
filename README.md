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
# Pharma Commercial Patient Journey Lakehouse

Overview

This project simulates a real-world biopharma commercial data engineering use case inspired by enterprise requirements such as Bristol Myers Squibb’s commercial data platforms.

The solution designs and implements an end-to-end ELT lakehouse pipeline that ingests, cleanses, transforms, and curates synthetic claims, patient/HUB, provider, and specialty pharmacy datasets into analytics-ready commercial data products using Databricks, Delta Lake, SQL, and PySpark.

Business Problem

Biopharma commercial organizations often struggle with fragmented patient and provider data spread across:

Claims systems
HUB/patient support programs
Specialty pharmacy fulfillment
Provider master systems
Key business objective:

Track patient progression from prescription → claim approval → HUB enrollment → therapy fulfillment → adherence, while identifying drop-off points and operational bottlenecks.

Solution Architecture
Medallion Architecture:
Bronze Layer:
Raw CSV/JSON ingestion
Schema preservation
Source lineage tracking
Metadata enrichment
Silver Layer:
Cleansing
Standardization
Deduplication
PHI masking
Data validation
SCD Type 2 provider modeling
Gold Layer:
Patient Journey Funnel
Patient Drop-off Analysis
Provider Performance Analytics
Tech Stack
Core:
Databricks
Delta Lake
PySpark
SQL
Python
Git/GitHub
Data Engineering Concepts:
ELT Pipelines
Medallion Architecture (Bronze/Silver/Gold)
Slowly Changing Dimensions (SCD Type 2)
Delta MERGE / Upserts
Data Quality Frameworks
Governance & PHI Masking
Visualization:
Power BI / Tableau (Optional)
Synthetic Datasets
1. Claims Data

Includes:

Claim status
NDC codes
Diagnosis codes
Payer IDs
Rejection reasons
2. Patient / HUB Data

Includes:

Enrollment
Therapy start
Support programs
Consent
3. Specialty Pharmacy Data

Includes:

Shipments
Delivery delays
Refill adherence
4. Provider Master

Includes:

Specialty
Territory
Active status
Key Features
Data Engineering:
Batch ingestion pipelines
Incremental-ready architecture
Cross-domain joins
Curated business marts
Data Quality:
Null checks
Duplicate detection
Reconciliation
Delay anomaly flags
Standardization
Governance:
PHI masking
Lineage
Documentation
Role-based conceptual access controls
Gold Layer Outputs
Patient Journey Funnel

Tracks:

Prescription initiation
Claim approval
HUB enrollment
Shipment fulfillment
Adherence
Patient Drop-off Analysis

Identifies:

Rejected claims
Missing HUB engagement
Delayed shipments
Provider Performance

Measures:

Approval rates
Patient volume
Delivery success
Territory trends

Raw Pharma Data Sources
     ↓
Bronze Layer (Raw Delta)
     ↓
Silver Layer (Clean + Standardized + SCD + Governance)
     ↓
Gold Layer (Commercial Analytics Data Products)
     ↓
Power BI / Tableau
##############################################################################################################################

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

