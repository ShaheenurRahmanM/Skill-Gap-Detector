import numpy as np
import pandas as pd
import joblib
import ast

# Load pre-trained components
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
le_combined = joblib.load("model/label_encoder.pkl")

# Load dataset
df = pd.read_csv("dataset/final.csv")
df.drop_duplicates(inplace=True)
df['Skills'] = df['Skills'].apply(ast.literal_eval)
df['Skills'] = df['Skills'].apply(lambda x: ', '.join(x))
df['Combined_Label'] = df['Role'].str.strip() + " || " + df['Domain'].str.strip()

# Skill normalization dictionary
skill_mapping = {
    "ml": "machine learning", "dl": "deep learning", "ai": "artificial intelligence",
    "rest api": "rest api", "rest apis": "rest api", "restful api": "rest api",
    "restful": "rest api", "rest": "rest api", "springboot": "spring boot",
    "spring-boot": "spring boot", "apis": "api", "large language models": "llms",
    "large language model": "llms", "llm": "llms", "natural language understanding": "natural language processing",
    "natural language generation": "natural language processing", "nlp": "natural language processing",
    "Natural language processing": "natural language processing", "viz": "visualization",
    "data viz": "data visualization", "tensorflow 2.0": "tensorflow", "py": "python",
    "react": "react", "react js": "react", "react.js": "react", "js": "javascript",
    "c plus plus": "c++", "cpp": "c++", "csharp": "c#", "rdbms": "relational database",
    "sql server": "sql", "postgressql": "postgresql", "nosql db": "nosql",
    "xgboost": "gradient boosting", "gboost": "gradient boosting", "pytorch": "deep learning",
    "prompting": "prompt engineering", "prompt": "prompt engineering", "AI prompt": "prompt engineering",
    "AI prompting": "prompt engineering", "convolutional neural network": "cnn",
    "convolutional neural networks": "cnn", "convolutional neural net": "cnn",
    "convolutional neural nets": "cnn", "recurrent neural network": "rnn",
    "recurrent neural networks": "rnn", "recurrent neural net": "rnn",
    "recurrent neural nets": "rnn", "long short term memory": "lstm", "long short term memory networks": "lstm",
    "long short term memory net": "lstm", "Genarative adversarial networks": "gans",
    "Generative adversarial network": "gans", "ML pipeline": "ml pipelines", "MLpipeline": "ml pipelines",
    "MLOps": "ml ops", "stats": "statistics", "stat": "statistics", "maths": "mathematics",
    "math": "mathematics", "algorithm": "algorithms", "Data structures": "data structures",
    "Data structure": "data structures", "DSA": "dsa", "System designing": "system design",
    "System design": "system design", "Oops": "oop", "Object oriented programming": "oop",
    "Object oriented programming language": "oop"
}

# Normalize skills string
def normalize_skills(skill_string):
    skill_phrases = [s.strip().lower() for s in skill_string.split(',')]
    normalized_phrases = [skill_mapping.get(p, p) for p in skill_phrases]
    return ', '.join(normalized_phrases)

# Normalize list of skills
def normalize_skill_list(skill_list):
    normalized_skills = set()
    for skill in skill_list:
        normalized = skill_mapping.get(skill.lower().strip(), skill.lower().strip())
        normalized_skills.add(normalized)
    return normalized_skills

# Predict top roles/domains
def predict_top_roles_domains(user_skills, top_n=5):
    input_str = ', '.join(user_skills)
    normalized_input = normalize_skills(input_str)
    input_vector = vectorizer.transform([normalized_input])
    probs = model.predict_proba(input_vector)[0]
    top_indices = np.argsort(probs)[-top_n:][::-1]

    predictions = []
    for idx in top_indices:
        combined_label = le_combined.inverse_transform([idx])[0]
        role, domain = combined_label.split(" || ")
        predictions.append((role, domain, round(probs[idx], 3)))
    return predictions

# Get skill gaps
def get_gap_skills(user_skills, role, domain):
    user_tokens = normalize_skill_list(user_skills)

    filtered = df[(df['Role'].str.lower() == role.lower()) &
                  (df['Domain'].str.lower() == domain.lower())]

    if filtered.empty:
        return user_tokens, set(), set()

    required_tokens = set()
    for skills_str in filtered['Skills']:
        normalized = normalize_skills(skills_str)
        required_tokens.update([s.strip() for s in normalized.split(',')])

    gap_tokens = required_tokens - user_tokens
    return user_tokens, required_tokens, gap_tokens
