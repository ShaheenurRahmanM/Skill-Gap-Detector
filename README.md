# SkillFit: Smart Skill Analyzer & Role Matcher

SkillFit is a smart web-based tool that analyzes a user's resume or LinkedIn profile (PDF format), extracts their skills, and predicts the top 5 job roles they’re most suited for – along with the match percentage and upskilling guidance.

It’s designed to help users understand their current strengths, discover suitable career paths, identify skill gaps, and receive personalized learning resources – all in one place.

Live App: [https://skillfit.streamlit.app/](https://skillfit.streamlit.app/)

---

## Features

- Upload LinkedIn PDF or Resume
- Skill Extraction using NLP
- Predicts Top 5 Job Roles based on your skills
- Visualizes Role Match Percentage using graphs
- Shows missing skills for top 3 matched roles
- Provides upskilling links (courses/resources)
- Shares job openings from LinkedIn, Indeed, and Glassdoor
- Downloadable skill analysis report

---

## How It Works

1. Upload a resume or LinkedIn profile PDF
2. Extract skills using text processing and NLP
3. Match with a curated job role-skill dataset (synthetic)
4. Predict top 5 job roles using a trained Random Forest classifier (97% accuracy)
5. Display results including visual graph, match scores, missing skills, learning links, and job links
6. Download report with all findings

---

## Tech Stack

- Python  
- Machine Learning (Random Forest, SVM, Logistic Regression)  
- Natural Language Processing (NLP)  
- Streamlit  
- Pandas, Scikit-learn, Matplotlib  
- PDF Parsing with PyMuPDF

---

## Dataset

We used a synthetic dataset that includes job roles, domains, and associated skillsets curated from various online job portals and job descriptions. This dataset was used to train and test the machine learning models.

---

## Try It Out

You can try the live app here:  
[https://skillfit.streamlit.app/](https://skillfit.streamlit.app/)

---

## Screenshots

![Screenshot 2025-05-23 150308](https://github.com/user-attachments/assets/f5f92096-8268-4de4-a2cc-a708f79a44bb)
![Screenshot 2025-05-23 150446](https://github.com/user-attachments/assets/ff025ace-2cc2-4d0f-a263-9dfd7823c0ec)
![Screenshot 2025-05-23 151547](https://github.com/user-attachments/assets/b410214a-9b1a-42e5-a611-08c8c132cfef)
![Screenshot 2025-05-23 150521](https://github.com/user-attachments/assets/fe7f026f-db30-4fe7-b6c8-6a2bbda2cfff)


## Contact

If you have any questions, suggestions, or would like to collaborate, feel free to reach out.

- **Name**: Shaheenur Rahman M
- **Email**: [shaheenur2005@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/shaheenurrahman](www.linkedin.com/in/shaheenur-rahman-m-2263b3290)
- **Project Demo**: [https://skillfit.streamlit.app](https://skillfit.streamlit.app)
