#comments tells about the steps
# Import necessary libraries

import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Load spaCy English model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Attempt to download NLTK punkt tokenizer data
try:
    nltk.download('punkt', quiet=True)  # Downloads the necessary NLTK data for sentence tokenization
    sent_tokenize("Test")  # Test the sentence tokenizer from NLTK
except:
    # If the download or tokenizer fails, fall back to spaCy for sentence tokenization
    print("NLTK sentence tokenization failed. Using spaCy instead.")

# Function to check if a sentence describes a task
def is_task(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    print(f"\nChecking Sentence: {sentence}")  
    
    # Check if the sentence contains an imperative verb (command-like)
    imperative = any(token.tag_ in {"VB", "VBG", "VBN"} for token in doc)
    
    # Define a set of words that indicate modality (e.g., necessity, obligation)
    modality_keywords = {"must", "should", "need to", "has to", "required to", "is supposed to"}
    modality = any(token.text.lower() in modality_keywords for token in doc)
    
    # Check if the sentence mentions a person (using named entity recognition)
    contains_person = any(ent.label_ == "PERSON" for ent in doc.ents)
    
    # Check if the sentence contains an action (verb)
    contains_action = any(token.pos_ == "VERB" for token in doc)

    # Output details for debugging
    print(f"  - Imperative: {imperative}, Modality: {modality}, Contains Person: {contains_person}, Contains Action: {contains_action}")
    
    # A sentence is considered a task if it is either imperative or has modality, and contains an action
    return (imperative or modality) and contains_action

# Function to extract task-related information from a sentence
def extract_task_info(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)
    person, deadline, action = None, None, sentence  # Default values for task info

    # Search for a person entity in the sentence
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person = ent.text  # Assign the first person entity found
            break  # Stop after the first person is found

    # Keywords indicating time (e.g., "by", "before", "tomorrow")
    time_keywords = {"by", "before", "until", "at", "on", "today", "tomorrow"}
    
    # Search for date or time-related entities or time keywords
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME"} or any(word in ent.text.lower() for word in time_keywords):
            deadline = ent.text  # Assign the date or time entity found
            break  # Stop after the first deadline is found

    # Return a dictionary with extracted task details
    return {"task": action, "who": person, "deadline": deadline}

# Function to extract tasks from the entire text
def extract_tasks_from_text(text):
    try:
        # Try using NLTK to tokenize the text into sentences
        sentences = sent_tokenize(text)  
    except:
        # If NLTK fails, fall back to using spaCy to tokenize sentences
        print("NLTK sentence tokenization failed. Using spaCy instead.")
        sentences = [sent.text for sent in nlp(text).sents]  # spaCy sentence tokenizer
    
    extracted_tasks = []  # List to store all extracted tasks
    
    # Loop through each sentence in the text
    for sentence in sentences:
        # Check if the sentence is a task
        if is_task(sentence):
            # If it's a task, extract task details (action, person, deadline)
            extracted_tasks.append(extract_task_info(sentence))
    
    # Return the list of extracted tasks
    return extracted_tasks

# Sample input text containing task descriptions
text = """he needs to submit the project report by Monday. He should also review the team's progress. 
         has to organize the client meeting tomorrow. Everyone must complete their assigned tasks before the deadline."""

# Extract tasks from the input text
tasks = extract_tasks_from_text(text)

# Print the extracted tasks with details
print("\nExtracted Tasks:")
for task in tasks:
    print(task)  # Print each task's details (action, person, and deadline)
