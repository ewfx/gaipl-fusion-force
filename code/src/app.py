import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app, specifying the correct path to the templates folder
app = Flask(__name__,
            template_folder=os.path.join('C:/Users/snrah/PycharmProjects/gaipl-fusion-force/code/src/Templates'))

# Load pre-trained Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can also use other models such as 'paraphrase-MiniLM-L6-v2'

# Define your Gemini API Key
GEMINI_API_KEY = "your-gemini-api-key"  # Replace with your actual Gemini API key
GEMINI_API_URL = "https://api.gemini.com/v1/query"  # Example URL, adjust based on actual Gemini API endpoint


# Function to load the QAPair.xlsx document into memory
def load_qa_pairs(file_path):
    df = pd.read_excel(file_path)
    qa_pairs = dict(zip(df['Question'], df['Answer']))
    return df


# Specify the exact path to the QAPair.xlsx file
qa_file_path = "C:/Users/snrah/Downloads/QAPair.xlsx"  # Update with your path

# Load the Q&A pairs from the QAPair.xlsx file
qa_pairs_df = load_qa_pairs(qa_file_path)


# Function to get an answer from the loaded Q&A pairs based on text similarity
def get_similar_answer(user_question):
    # Generate embeddings for the user question and all available questions in the QAPair file
    all_questions = qa_pairs_df['Question'].tolist()
    all_questions.append(user_question)  # Add the user question to the list for comparison

    # Generate embeddings for all questions and the user input
    question_embeddings = model.encode(all_questions)

    # Compute cosine similarity between the user's question and all available questions
    cosine_sim = cosine_similarity([question_embeddings[-1]], question_embeddings[:-1])

    # Get the index of the most similar question
    similar_question_index = cosine_sim.argmax()


    # Retrieve the answer corresponding to the most similar question
    similar_answer = qa_pairs_df.iloc[similar_question_index]['Answer']

    return similar_answer


# Function to get the response from the Gemini API
def get_gemini_response(query):
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "query": query,
        "max_tokens": 150,
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("text", "Sorry, I couldn't get a valid response.")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return "Sorry, there was an issue reaching the Gemini API."


@app.route('/')
def home():
    return render_template('index.html')  # Make sure Flask can find index.html in the specified template folder


def get_similar_answer(user_question):
    # List of generic words/phrases that don't need to be answered
    generic_phrases = ['hello', 'hi', 'how are you', 'hey', 'good morning', 'bye', 'thank you', 'thanks', 'what\'s up']

    # Normalize the input by converting to lower case and removing any extra spaces
    user_question_normalized = user_question.strip().lower()

    # If the user input is a generic greeting, return a default greeting response
    if any(phrase in user_question_normalized for phrase in generic_phrases):
        return "Hello! How can I assist you today?"

    # Generate embeddings for the user question and all available questions in the QAPair file
    all_questions = qa_pairs_df['Question'].tolist()
    all_questions.append(user_question)  # Add the user question to the list for comparison

    # Generate embeddings for all questions and the user input
    question_embeddings = model.encode(all_questions)

    # Compute cosine similarity between the user's question and all available questions
    cosine_sim = cosine_similarity([question_embeddings[-1]], question_embeddings[:-1])

    # Get the index of the most similar question
    similar_question_index = cosine_sim.argmax()

    # Check if the similarity score is above a certain threshold (e.g., 0.7)
    if cosine_sim[0][similar_question_index] < 0.7:  # This threshold can be adjusted
        return "I'm sorry, I couldn't find an answer to that. Could you please rephrase your question?"

    # Retrieve the answer corresponding to the most similar question
    similar_answer = qa_pairs_df.iloc[similar_question_index]['Answer']

    return similar_answer

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']

    # Look up answer based on similar questions in the Q&A pairs
    bot_response = get_similar_answer(user_input)

    # If the answer isn't found in Q&A, use the Gemini API for a response
    if not bot_response:
        bot_response = get_gemini_response(user_input)

    # Return the response back to the frontend
    return jsonify({'bot_response': bot_response})


if __name__ == "__main__":
    app.run(debug=True)
