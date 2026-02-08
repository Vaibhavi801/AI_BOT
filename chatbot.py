import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# NLTK DOWNLOADS (RUN ONCE)
# -----------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# -----------------------------
# INTENTS
# -----------------------------
intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey"],
        "responses": [
            "Hello! How can I help you?",
            "Hi there! What can I do for you?"
        ]
    },
    "how_are_you": {
        "patterns": ["how are you", "how are you doing"],
        "responses": [
            "I'm doing great, thanks for asking!"
        ]
    },
    "capabilities": {
        "patterns": ["what can you do", "your features", "help me"],
        "responses": [
            "I can answer questions using NLP techniques.",
            "I use Natural Language Processing to understand text."
        ]
    },
    "nlp": {
        "patterns": ["what is nlp", "explain nlp"],
        "responses": [
            "NLP stands for Natural Language Processing.",
            "It allows machines to understand human language."
        ]
    },
    # ONLY explicit continuation
    "continue": {
        "patterns": ["go ahead", "tell me more", "continue"],
        "responses": []
    },
    # Neutral acknowledgement (NO continuation)
    "acknowledgement": {
        "patterns": ["okay", "ok", "hmm", "alright"],
        "responses": ["👍", "Okay!", "Got it.", "Sure."]
    },
    "goodbye": {
        "patterns": ["bye", "exit", "quit"],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later!"
        ]
    }
}

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation
    ]
    return " ".join(tokens)

# -----------------------------
# TRAINING DATA
# -----------------------------
patterns = []
intent_tags = []

for intent, data in intents.items():
    for pattern in data["patterns"]:
        patterns.append(preprocess(pattern))
        intent_tags.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# -----------------------------
# CONTEXT MEMORY
# -----------------------------
last_intent = None
last_response_index = {}

# -----------------------------
# RESPONSE FUNCTION
# -----------------------------
def get_response(user_input):
    global last_intent, last_response_index

    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)

    best_match = similarities.argmax()
    confidence = similarities[0][best_match]

    if confidence < 0.2:
        return "Sorry, I didn't understand that. Can you rephrase?"

    intent = intent_tags[best_match]

    # Acknowledgement → do NOT continue topic
    if intent == "acknowledgement":
        return random.choice(intents["acknowledgement"]["responses"])

    # Explicit continuation
    if intent == "continue" and last_intent:
        responses = intents[last_intent]["responses"]
        index = last_response_index.get(last_intent, 0)

        if index < len(responses):
            reply = responses[index]
            last_response_index[last_intent] = index + 1
            return reply
        else:
            return "That's all I can tell you about this. Ask me something else 😊"

    # New intent
    last_intent = intent
    last_response_index[intent] = 1

    return intents[intent]["responses"][0]

# -----------------------------
# CHAT LOOP
# -----------------------------
print("🤖 AI Chatbot (Context Aware)")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["bye", "exit", "quit"]:
        print("AI Chatbot:", random.choice(intents["goodbye"]["responses"]))
        break

    reply = get_response(user_input)
    print("AI Chatbot:", reply)
