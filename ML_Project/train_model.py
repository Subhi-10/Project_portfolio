import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Define the dataset (expand as necessary)
data = {
    'word': ['python', 'java', 'ruby', 'html', 'css', 'php', 'javascript', 'typescript', 'scala', 'rust', 
             'machinelearning', 'deeplearning', 'react', 'angular', 'vue', 'graphql', 'nextjs', 'flask', 
             'django', 'tensorflow', 'cybersecurity', 'devops'],
    'difficulty': ['Easy', 'Easy', 'Easy', 'Easy', 'Easy', 'Easy', 'Medium', 'Medium', 'Medium', 'Medium',
                   'Difficult', 'Difficult', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium',
                   'Medium', 'Difficult', 'Difficult', 'Difficult']
}

# Step 2: Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Step 3: Vectorize the words (converts words into numerical data for the model)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['word'])

# Step 4: Define the target variable (difficulty levels)
y = df['difficulty']

# Step 5: Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Step 6: Save the model and the vectorizer to a file
with open('word_difficulty_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, model), file)

print("Model training complete, model saved as 'word_difficulty_model.pkl'")
