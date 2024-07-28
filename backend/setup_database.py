# setup_database.py

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['my_database']
collection = db['collection']

# Example document
candidate = {
    "name": "John Doe",
    "skills": ["HTML", "CSS", "JavaScript", "React"],
    "experience": "5 years",
    "profile": "Experienced UI Developer."
}

# Insert document
collection.insert_one(candidate)
