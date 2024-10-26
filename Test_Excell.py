import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from fuzzywuzzy import fuzz
import pandas as pd
import time


def preprocess_answer(answer):
    """Basic preprocessing to convert answers to lowercase, remove extra spaces, and strip unnecessary characters."""
    return " ".join(answer.lower().strip().split())

def compare_textual_answers(expected, actual):
    """Token-based comparison for textual answers using cosine similarity (for descriptions)."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([expected, actual])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

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




def evaluate_answers_to_excel(file_paths, threshold=0.40, batch_size=5, output_file="evaluation_batch_results.xlsx"):
    """Evaluates the actual answers from multiple files, displays results, and saves to Excel."""
    y_true = []
    y_pred = []
    
    batch_y_true = []
    batch_y_pred = []
    
    results = []  # To store results for Excel output
    batch_count = 0  # Track the number of batches
    
    print("Detailed Results:\n")
    
    # Iterate over all file paths and process each file
    for file_path in file_paths:
        data = load_json(file_path)
        print(f"Processing file: {file_path}")
        
        for idx, item in enumerate(data):
            expected_answer = str(item['expected_answer']).lower()
            actual_answer = str(item['actual_answer']).lower()
            correct = False
            score = 0
            
            # Special case for description: Use token-based comparison
            if "description" in item['question'].lower():
                processed_expected = preprocess_answer(expected_answer)
                processed_actual = preprocess_answer(actual_answer)
                score = compare_textual_answers(processed_expected, processed_actual)
                correct = score >= threshold
            
            # Special case for date: Use date parsing and comparison
            elif "date" in item['question'].lower():
                correct = compare_dates(expected_answer, actual_answer)
                if correct:
                    score = 1.0
                else:
                    correct = expected_answer in actual_answer
                    score = 1.0 if correct else 0.0
            
            # Check if expected answer is "no data found" and if the actual answer contains similar phrases
            elif expected_answer == "no data available":
                no_data_variants = [
                    "no data found", "data not found", "no data available", "not aware of any information","doesn't specifically mention","couldn't find any information",
                    "information not found", "data unavailable", "no information available","data not available","information is not available"
                ]
                
                fuzzy_threshold = 50  # Percentage similarity threshold for approximate matching
                
                for variant in no_data_variants:
                    if fuzz.partial_ratio(variant, actual_answer) >= fuzzy_threshold:
                        correct = True
                        score = 1.0
                        break
                else:
                    for variant in no_data_variants:
                        correct = variant in actual_answer
                        if correct:
                            score = 1.0
                            break 
                        else:
                            score = 0.0

            
                    
                    # correct = False
                    # score = 0.0
            
            # For all other fields, check if the expected answer string is present in the actual answer
            else:
                correct = expected_answer in actual_answer
                score = 1.0 if correct else 0.0
            
            y_true.append(1)  # Expected to match
            y_pred.append(1 if correct else 0)
            batch_y_true.append(1)
            batch_y_pred.append(1 if correct else 0)
            
            # Append the result for this question to the results list for Excel output
            results.append({
                "Question": item['question'],
                "Expected Answer": item['expected_answer'],
                "Actual Answer": item['actual_answer'],
                "Score": f"{score:.2f}",
                "Correct": "Yes" if correct else "No"
            })
            
            # After every batch of batch_size questions, calculate precision, recall, and F1 for that batch
            if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
                batch_count += 1
                batch_precision = precision_score(batch_y_true, batch_y_pred)
                batch_recall = recall_score(batch_y_true, batch_y_pred)
                batch_f1 = f1_score(batch_y_true, batch_y_pred)
                
                # Print batch results
                print(f"Batch {batch_count} Results:")
                print(f'Precision: {batch_precision:.2f}')
                print(f'Recall: {batch_recall:.2f}')
                print(f'F1 Score: {batch_f1:.2f}\n')
                
                # Append batch results to the results list
                results.append({
                    "Question": f"Batch {batch_count} Results",
                    "Expected Answer": "",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                results.append({
                    "Question": "Precision",
                    "Expected Answer": f"{batch_precision:.2f}",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                results.append({
                    "Question": "Recall",
                    "Expected Answer": f"{batch_recall:.2f}",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                results.append({
                    "Question": "F1 Score",
                    "Expected Answer": f"{batch_f1:.2f}",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                
                # Clear batch lists for next batch
                batch_y_true.clear()
                batch_y_pred.clear()
                
                # Every 5 batches, show overall precision, recall, and F1 score so far
                if batch_count % 5 == 0 or idx == len(data) - 1:
                    overall_precision = precision_score(y_true, y_pred)
                    overall_recall = recall_score(y_true, y_pred)
                    overall_f1 = f1_score(y_true, y_pred)
                    
                    # Print overall results so far
                    print(f"Overall Results after {batch_count} Batches:")
                    print(f'Overall Precision: {overall_precision:.2f}')
                    print(f'Overall Recall: {overall_recall:.2f}')
                    print(f'Overall F1 Score: {overall_f1:.2f}\n')
                    
                    # Append overall results to the results list
                    results.append({
                        "Question": f"Overall Results after {batch_count} Batches",
                        "Expected Answer": "",
                        "Actual Answer": "",
                        "Score": "",
                        "Correct": ""
                    })
                    results.append({
                        "Question": "Overall Precision",
                        "Expected Answer": f"{overall_precision:.2f}",
                        "Actual Answer": "",
                        "Score": "",
                        "Correct": ""
                    })
                    results.append({
                        "Question": "Overall Recall",
                        "Expected Answer": f"{overall_recall:.2f}",
                        "Actual Answer": "",
                        "Score": "",
                        "Correct": ""
                    })
                    results.append({
                        "Question": "Overall F1 Score",
                        "Expected Answer": f"{overall_f1:.2f}",
                        "Actual Answer": "",
                        "Score": "",
                        "Correct": ""
                    })

    # Final overall precision, recall, and F1 score for all questions
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    df_results.to_excel(output_file, index=False)
    
    return precision, recall, f1

# List of file paths to process
# file_paths = ['gemini_output_results_1.json','gemini_output_results_2.json','gemini_output_results_3.json','gemini_output_results_4.json','gemini_output_results_5.json']  # Replace with your file paths
file_paths = ['ollama_output_results_1.json','ollama_output_results_2.json','ollama_output_results_3.json','ollama_output_results_4.json','ollama_output_results_5.json']  # Replace with your file paths
# Evaluate the answers, save the results to an Excel file, and display detailed results
output_file = "evaluation_batch_Llama_2.xlsx"  # Output Excel file path
precision, recall, f1 = evaluate_answers_to_excel(file_paths, threshold=0.40, batch_size=5, output_file=output_file)

# Print final overall precision, recall, and F1 score
print("Final Overall Results:")
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Results saved to {output_file}')