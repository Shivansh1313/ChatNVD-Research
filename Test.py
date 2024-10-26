# in this date has been parsed into a common format to check it properly

import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from fuzzywuzzy import fuzz
import pandas as pd

def load_json(file_path):
    """Loads the JSON file with questions, expected answers, and actual answers."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_answer(answer):
    """Basic preprocessing to convert answers to lowercase and strip unnecessary characters."""
    return answer.lower().strip()

def compare_textual_answers(expected, actual):
    """Token-based comparison for textual answers using cosine similarity (for descriptions)."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([expected, actual])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def parse_date(date_str):
    """Attempts to parse the date string into a standard format for comparison."""
    date_formats = [
        "%Y-%m-%dT%H:%MZ", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", 
        "%B %d, %Y", "%b %d, %Y"  # Adding long and short month formats (e.g., "October 25, 2023")
    ]
    for fmt in date_formats:
        try:
            #return datetime.strptime(date_str, fmt).date()  # Parse to date only
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")  # Parse to date only
        except ValueError:
            continue
    return None


def parse_date_2(date_str):

    # Regular expression to match the date pattern
    date_pattern = r"([A-Za-z]+\s\d{1,2},\s\d{4})"

    # Find the date in the sentence
    match = re.search(date_pattern, date_str)

    # Check if a match is found
    if match:
        extracted_date = match.group(0)
        # Convert the extracted date to a datetime object
        try:
            date_obj = datetime.strptime(extracted_date, "%B %d, %Y")
            # Convert to ISO format
            iso_date = date_obj.strftime("%Y-%m-%d")
            print("Extracted date in ISO format:", iso_date)
            return iso_date
        except ValueError:
            #print("Date format is incorrect.")
            return
            
    # else:
    #     print("No date found in the sentence.")

def compare_dates(expected, actual):
    """Compares dates by parsing them into a standard format and ignoring time."""
    expected_date = parse_date(expected)
    actual_date = parse_date_2(actual)
    print("Expected data: ")
    print(expected_date)
    print("\n Actual data: ")
    print(actual_date)
    if expected_date and actual_date:
        return expected_date == actual_date
    return False



from fuzzywuzzy import fuzz

# def evaluate_answers(data, threshold=0.75):
#     """Evaluates the actual answers against the expected answers and displays results for each question."""
#     y_true = []
#     y_pred = []
    
#     print("Detailed Results:\n")
    
#     for idx, item in enumerate(data):
#         expected_answer = str(item['expected_answer']).lower()
#         actual_answer = str(item['actual_answer']).lower()
#         correct = False
#         score = 0
        
#         # Special case for description: Use token-based comparison
#         if "description" in item['question'].lower():
#             score = compare_textual_answers(preprocess_answer(expected_answer), preprocess_answer(actual_answer))
#             correct = score >= threshold
        
#         # Special case for date: Use date parsing and comparison
#         elif "date" in item['question'].lower():
#             correct = compare_dates(expected_answer, actual_answer)
#             if correct:
#                 score = 1.0
#             else:
#                 correct = expected_answer in actual_answer
#                 score = 1.0 if correct else 0.0
        
#         # Check if expected answer is "no data found" and if the actual answer contains similar phrases
#         elif expected_answer == "no data available":
#             # Define possible variations of "no data found"
#             no_data_variants = [
#                 "no data found", "data not found", "no data available", 
#                 "information not found", "data unavailable", "no information available"
#             ]
            
#             # Fuzzy matching threshold
#             fuzzy_threshold = 80  # Percentage similarity threshold for approximate matching
            
#             # Check if any of the variations appear in the actual answer
#             for variant in no_data_variants:
#                 if fuzz.partial_ratio(variant, actual_answer) >= fuzzy_threshold:
#                     correct = True
#                     score = 1.0
#                     break
#             else:
#                 correct = False
#                 score = 0.0
        
#         # For all other fields, check if the expected answer string is present in the actual answer
#         else:
#             correct = expected_answer in actual_answer
#             score = 1.0 if correct else 0.0
        
#         y_true.append(1)  # Expected to match
#         y_pred.append(1 if correct else 0)
        
#         # Display the result for this question
#         print(f"Question {idx + 1}: {item['question']}")
#         print(f"Expected Answer: {item['expected_answer']}")
#         print(f"Actual Answer: {item['actual_answer']}")
#         print(f"Score: {score:.2f}")
#         print(f"Correct: {'Yes' if correct else 'No'}\n")
    
#     # Calculate precision, recall, and F1 score
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
    
#     return precision, recall, f1

# # Load the JSON file
# file_path = 'ollama_output_results.json'  # Replace with your JSON file path
# data = load_json(file_path)

# # Evaluate the answers and display detailed results
# precision, recall, f1 = evaluate_answers(data, threshold=0.75)

# # Print overall precision, recall, and F1 score
# print("Overall Results:")
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 Score: {f1:.2f}')
#######################################################################
# here no data available is working fine
# from sklearn.metrics import precision_score, recall_score, f1_score
# from fuzzywuzzy import fuzz

# def preprocess_answer(answer):
#     """Basic preprocessing to convert answers to lowercase, remove extra spaces, and strip unnecessary characters."""
#     return " ".join(answer.lower().strip().split())

# def compare_textual_answers(expected, actual):
#     """Token-based comparison for textual answers using cosine similarity (for descriptions)."""
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([expected, actual])
#     similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
#     return similarity

# def evaluate_answers(data, threshold=0.75, batch_size=5):
#     """Evaluates the actual answers against the expected answers and displays results in batches."""
#     y_true = []
#     y_pred = []
    
#     batch_y_true = []
#     batch_y_pred = []
    
#     print("Detailed Results:\n")
    
#     for idx, item in enumerate(data):
#         expected_answer = str(item['expected_answer']).lower()
#         actual_answer = str(item['actual_answer']).lower()
#         correct = False
#         score = 0
        
#         # Special case for description: Use token-based comparison
#         if "description" in item['question'].lower():
#             processed_expected = preprocess_answer(expected_answer)
#             processed_actual = preprocess_answer(actual_answer)
#             score = compare_textual_answers(processed_expected, processed_actual)
#             correct = score >= threshold
        
#         # Special case for date: Use date parsing and comparison
#         elif "date" in item['question'].lower():
#             correct = compare_dates(expected_answer, actual_answer)
#             if correct:
#                 score = 1.0
#             else:
#                 correct = expected_answer in actual_answer
#                 score = 1.0 if correct else 0.0
        
#         # Check if expected answer is "no data found" and if the actual answer contains similar phrases
#         elif expected_answer == "no data available":
#             no_data_variants = [
#                 "no data found", "data not found", "no data available", 
#                 "information not found", "data unavailable", "no information available"
#             ]
            
#             fuzzy_threshold = 80  # Percentage similarity threshold for approximate matching
            
#             for variant in no_data_variants:
#                 if fuzz.partial_ratio(variant, actual_answer) >= fuzzy_threshold:
#                     correct = True
#                     score = 1.0
#                     break
#             else:
#                 correct = False
#                 score = 0.0
        
#         # For all other fields, check if the expected answer string is present in the actual answer
#         else:
#             correct = expected_answer in actual_answer
#             score = 1.0 if correct else 0.0
        
#         y_true.append(1)  # Expected to match
#         y_pred.append(1 if correct else 0)
#         batch_y_true.append(1)
#         batch_y_pred.append(1 if correct else 0)
        
#         # Display the result for this question
#         print(f"Question {idx + 1}: {item['question']}")
#         print(f"Expected Answer: {item['expected_answer']}")
#         print(f"Actual Answer: {item['actual_answer']}")
#         print(f"Score: {score:.2f}")
#         print(f"Correct: {'Yes' if correct else 'No'}\n")
        
#         # After every batch of 5 questions, calculate precision, recall, and F1 for that batch
#         if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
#             batch_precision = precision_score(batch_y_true, batch_y_pred)
#             batch_recall = recall_score(batch_y_true, batch_y_pred)
#             batch_f1 = f1_score(batch_y_true, batch_y_pred)
            
#             print(f"Batch {idx // batch_size + 1} Results:")
#             print(f'Precision: {batch_precision:.2f}')
#             print(f'Recall: {batch_recall:.2f}')
#             print(f'F1 Score: {batch_f1:.2f}\n')
            
#             # Clear batch lists for next batch
#             batch_y_true.clear()
#             batch_y_pred.clear()
    
#     # Calculate precision, recall, and F1 score for all questions
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
    
#     return precision, recall, f1

# # Load the JSON file
# file_path = 'ollama_output_results.json'  # Replace with your JSON file path
# data = load_json(file_path)

# # Evaluate the answers and display detailed results
# precision, recall, f1 = evaluate_answers(data, threshold=0.75)

# # Print overall precision, recall, and F1 score
# print("Overall Results:")
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 Score: {f1:.2f}')

#####################################################
#here code is added to add in excell file




def evaluate_answers_to_excel(data, threshold=0.75, output_file="evaluation_results.xlsx"):
    """Evaluates the actual answers against the expected answers and saves results to an Excel file."""
    y_true = []
    y_pred = []
    
    results = []  # List to store the results
    
    print("Detailed Results:\n")
    
    for idx, item in enumerate(data):
        expected_answer = str(item['expected_answer']).lower()
        actual_answer = str(item['actual_answer']).lower()
        correct = False
        score = 0
        
        # Special case for description: Use token-based comparison
        if "description" in item['question'].lower():
            score = compare_textual_answers(preprocess_answer(expected_answer), preprocess_answer(actual_answer))
            correct = score >= threshold
        
        # Special case for date: Use date parsing and comparison
        elif "date" in item['question'].lower():
            correct = compare_dates(expected_answer, actual_answer)
            if correct:
                score = 1.0
            else:
                correct = expected_answer in actual_answer
                score = 1.0 if correct else 0.0
        
        # For all other fields, check if the expected answer string is present in the actual answer
        else:
            correct = expected_answer in actual_answer
            score = 1.0 if correct else 0.0
        
        y_true.append(1)  # Expected to match
        y_pred.append(1 if correct else 0)
        
        # Append the result for this question to the results list
        results.append({
            "Question": item['question'],
            "Expected Answer": item['expected_answer'],
            "Actual Answer": item['actual_answer'],
            "Score": f"{score:.2f}",
            "Correct": "Yes" if correct else "No"
        })
    
    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    df_results.to_excel(output_file, index=False)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1

# Load the JSON file
file_path = 'gemini_output_results.json'  # Replace with your JSON file path
data = load_json(file_path)

# Evaluate the answers, save the results to an Excel file, and display detailed results
output_file = "evaluation_results.xlsx"  # Output Excel file path
precision, recall, f1 = evaluate_answers_to_excel(data, threshold=0.75, output_file=output_file)

# Print overall precision, recall, and F1 score
print("Overall Results:")
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Results saved to {output_file}')
