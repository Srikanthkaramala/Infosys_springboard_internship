# Quantum Policy Assistant (Infosys Internship Project)

This repository was created as part of the **Infosys Springboard Internship (Batch 3, Team 1)**. 
It demonstrates how **machine learning**, **policy datasets**, and **quantum-inspired techniques** can be combined to build an intelligent assistant for exploring and analyzing public policies.


## Key Features
- **Policy Dataset Integration** — Includes structured CSV dataset (`policies.csv`)  
- **Machine Learning Models** — Scripts and saved models for policy classification and retrieval  
- **Quantum-Inspired Search** — Experimental module for enhanced similarity scoring  
- **Web Application** — Application logic under `app/` for serving results  
- **Extensible Design** — Modular structure for adding new models or datasets  

## Folder Structure
```
Quantam_Infosys/
│
├── app/                # Application logic and API layer
├── models/             # Trained ML models and scripts
├── policies.csv        # Policy dataset
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup Guide

### 1. Create and Activate Virtual Environment
```bash
cd Quantam_Infosys
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
uvicorn app.main:app --reload

```

### 4. Access in Browser
* API root: http://127.0.0.1:8000/
* Swagger UI: http://127.0.0.1:8000/docs
* ReDoc: http://127.0.0.1:8000/redoc

---

## Dataset Format
The included `policies.csv` file is structured with fields such as:

```
policy_id,title,full_text,state,year,category,status,region
1,Education Reform Act,Full text of the policy...,Delhi,2021,Education,Active,North
```

---

## Tech Stack
| Component       | Description                                |
|-----------------|--------------------------------------------|
| Python 3.x      | Core programming language                  |
| scikit-learn    | Machine learning model training            |
| Pandas / NumPy  | Data preprocessing and manipulation        |
| FastAPI / Flask | Backend web framework (depending on setup) |
| Quantum Module  | Experimental quantum-inspired search       |

---

## Credits
- Infosys Springboard Internship Program (Batch 3, Team 1)  
- Python ML and Quantum Computing open-source ecosystem  

