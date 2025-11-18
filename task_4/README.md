# Voice-to-Text Policy Assistant (Web Module)

**Submodule:** Voice_to_text — AI-powered voice-to-text assistant for policy navigation and visualization

This module implements a speech-enabled intelligent assistant that allows users to query and explore public policies. It integrates voice recognition, text-based search, and interactive data visualizations to make policy information more accessible.

# Key Features
- **Voice + Text Input** — Search policies by speaking or typing queries  
- **Policy Categories** — Education, Poverty, and Quantum-enhanced Education search  
- **Interactive Visualizations** — Bar charts, scatter plots, and pie charts using Chart.js  
- **CSV Export** — Download search results for offline analysis  
- **Web-based Interface** — Simple and responsive design for browser use  

# Folder Structure
```
project_root/
│
├── app.py              # Backend server (FastAPI/Flask style)
├── index.html          # Main frontend page
└── README.md
```
---

## Setup Guide

### 1. Create and Activate Virtual Environment
```bash
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
uvicorn app:app --reload
# or alternatively
python app.py
```

### 4. Access in Browser
Open:
```
http://localhost:8000
```

---

## CSV Input Format
The application expects policy data in CSV format with the following schema:

```
policy_id,title,full_text,state,year,category,status,region
1,Education Reform Act,Full text of the policy...,Delhi,2021,Education,Active,North
```

Ensure your dataset is stored in the appropriate data folder or configured path.

---

## Tech Stack
| Component            | Description                                |
|----------------------|--------------------------------------------|
| Python (Flask/FastAPI) | Backend server for API and routing        |
| HTML, CSS, JavaScript | Frontend interface                        |
| Web Speech API        | Voice recognition for queries             |
| Chart.js              | Interactive data visualizations           |
| CSV                   | Policy dataset input format               |

---

## Example Queries
- “Show active education policies in Delhi”  
- “List poverty alleviation schemes from 2020”  
- “Compare education policies by region”  

---

## Credits
- Web Speech API  
- Chart.js  
- Flask / FastAPI  
- Infosys Springboard Internship Program  
