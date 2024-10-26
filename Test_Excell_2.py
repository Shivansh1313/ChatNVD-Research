from sklearn.metrics import accuracy_score
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


# def evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.40, batch_size=5, output_file="evaluation_batch_results.xlsx"):
#     """Evaluates the actual answers from multiple files, displays results, and saves accuracy and error rate to Excel."""
#     y_true = []
#     y_pred = []
    
#     batch_y_true = []
#     batch_y_pred = []
    
#     results = []  # To store results for Excel output
#     batch_count = 0  # Track the number of batches
    
#     print("Detailed Results:\n")
    
#     # Iterate over all file paths and process each file
#     for file_path in file_paths:
#         data = load_json(file_path)
#         print(f"Processing file: {file_path}")
        
#         for idx, item in enumerate(data):
#             expected_answer = str(item['expected_answer']).lower()
#             actual_answer = str(item['actual_answer']).lower()
#             correct = False
#             score = 0
            
#             # Special case for description: Use token-based comparison
#             if "description" in item['question'].lower():
#                 processed_expected = preprocess_answer(expected_answer)
#                 processed_actual = preprocess_answer(actual_answer)
#                 score = compare_textual_answers(processed_expected, processed_actual)
#                 correct = score >= threshold
            
#             # Special case for date: Use date parsing and comparison
#             elif "date" in item['question'].lower():
#                 correct = compare_dates(expected_answer, actual_answer)
#                 if correct:
#                     score = 1.0
#                 else:
#                     correct = expected_answer in actual_answer
#                     score = 1.0 if correct else 0.0
            
#             # Check if expected answer is "no data found" and if the actual answer contains similar phrases
#             elif expected_answer == "no data available":
#                 no_data_variants = [
#                     "no data found", "data not found", "no data available", 
#                     "information not found", "data unavailable", "no information available","data not available","information is not available"
#                 ]
                
#                 fuzzy_threshold = 50  # Percentage similarity threshold for approximate matching
                
#                 for variant in no_data_variants:
#                     if fuzz.partial_ratio(variant, actual_answer) >= fuzzy_threshold:
#                         correct = True
#                         score = 1.0
#                         break
#                 else:
#                     correct = False
#                     score = 0.0
            
#             # For all other fields, check if the expected answer string is present in the actual answer
#             else:
#                 correct = expected_answer in actual_answer
#                 score = 1.0 if correct else 0.0
            
#             y_true.append(1)  # Expected to match
#             y_pred.append(1 if correct else 0)
#             batch_y_true.append(1)
#             batch_y_pred.append(1 if correct else 0)
            
#             # Append the result for this question to the results list for Excel output
#             results.append({
#                 "Question": item['question'],
#                 "Expected Answer": item['expected_answer'],
#                 "Actual Answer": item['actual_answer'],
#                 "Score": f"{score:.2f}",
#                 "Correct": "Yes" if correct else "No"
#             })
            
#             # After every batch of batch_size questions, calculate accuracy and error rate for that batch
#             if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
#                 batch_count += 1
#                 batch_accuracy = accuracy_score(batch_y_true, batch_y_pred)
#                 batch_error_rate = 1 - batch_accuracy
                
#                 # Print batch results
#                 print(f"Batch {batch_count} Results:")
#                 print(f'Accuracy: {batch_accuracy:.2f}')
#                 print(f'Error Rate: {batch_error_rate:.2f}\n')
                
#                 # Append batch results to the results list
#                 results.append({
#                     "Question": f"Batch {batch_count} Results",
#                     "Expected Answer": "",
#                     "Actual Answer": "",
#                     "Score": "",
#                     "Correct": ""
#                 })
#                 results.append({
#                     "Question": "Accuracy",
#                     "Expected Answer": f"{batch_accuracy:.2f}",
#                     "Actual Answer": "",
#                     "Score": "",
#                     "Correct": ""
#                 })
#                 results.append({
#                     "Question": "Error Rate",
#                     "Expected Answer": f"{batch_error_rate:.2f}",
#                     "Actual Answer": "",
#                     "Score": "",
#                     "Correct": ""
#                 })
                
#                 # Clear batch lists for next batch
#                 batch_y_true.clear()
#                 batch_y_pred.clear()
                
#                 # Every 5 batches, show overall accuracy and error rate so far
#                 if batch_count % 5 == 0 or idx == len(data) - 1:
#                     overall_accuracy = accuracy_score(y_true, y_pred)
#                     overall_error_rate = 1 - overall_accuracy
                    
#                     # Print overall results so far
#                     print(f"Overall Results after {batch_count} Batches:")
#                     print(f'Overall Accuracy: {overall_accuracy:.2f}')
#                     print(f'Overall Error Rate: {overall_error_rate:.2f}\n')
                    
#                     # Append overall results to the results list
#                     results.append({
#                         "Question": f"Overall Results after {batch_count} Batches",
#                         "Expected Answer": "",
#                         "Actual Answer": "",
#                         "Score": "",
#                         "Correct": ""
#                     })
#                     results.append({
#                         "Question": "Overall Accuracy",
#                         "Expected Answer": f"{overall_accuracy:.2f}",
#                         "Actual Answer": "",
#                         "Score": "",
#                         "Correct": ""
#                     })
#                     results.append({
#                         "Question": "Overall Error Rate",
#                         "Expected Answer": f"{overall_error_rate:.2f}",
#                         "Actual Answer": "",
#                         "Score": "",
#                         "Correct": ""
#                     })

#     # Final overall accuracy and error rate for all questions
#     accuracy = accuracy_score(y_true, y_pred)
#     error_rate = 1 - accuracy
    
#     # Create a DataFrame from the results list
#     df_results = pd.DataFrame(results)
    
#     # Save the DataFrame to an Excel file
#     df_results.to_excel(output_file, index=False)
    
#     return accuracy, error_rate


# def evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.50, batch_size=5, output_file="evaluation_batch_results.xlsx"):
#     """Evaluates the actual answers from multiple files, displays results, and saves accuracy and error rate to Excel."""
#     y_true = []  # Track the expected correct answers
#     y_pred = []  # Track the actual correctness based on the "Correct" column
    
#     results = []  # To store results for Excel output
#     batch_count = 0  # Track the number of batches
    
#     print("Detailed Results:\n")
    
#     # Iterate over all file paths and process each file
#     for file_path in file_paths:
#         data = load_json(file_path)
#         print(f"Processing file: {file_path}")
        
#         for idx, item in enumerate(data):
#             expected_answer = str(item['expected_answer']).lower()
#             actual_answer = str(item['actual_answer']).lower()
#             correct = False
#             score = 0
            
#             # Special case for description: Use token-based comparison
#             if "description" in item['question'].lower():
#                 processed_expected = preprocess_answer(expected_answer)
#                 processed_actual = preprocess_answer(actual_answer)
#                 score = compare_textual_answers(processed_expected, processed_actual)
#                 correct = score >= threshold
            
#             # Special case for date: Use date parsing and comparison
#             elif "date" in item['question'].lower():
#                 correct = compare_dates(expected_answer, actual_answer)
#                 if correct:
#                     score = 1.0
#                 else:
#                     correct = expected_answer in actual_answer
#                     score = 1.0 if correct else 0.0
            
#             # Check if expected answer is "no data found" and if the actual answer contains similar phrases
#             elif expected_answer == "no data available":
#                 no_data_variants = [
#                     "no data found", "data not found", "no data available", 
#                     "information not found", "data unavailable", "no information available","data not available","information is not available"
#                 ]
                
#                 fuzzy_threshold = 50  # Percentage similarity threshold for approximate matching
                
#                 for variant in no_data_variants:
#                     if fuzz.partial_ratio(variant, actual_answer) >= fuzzy_threshold:
#                         correct = True
#                         score = 1.0
#                         break
#                 else:
#                     correct = False
#                     score = 0.0
            
#             # For all other fields, check if the expected answer string is present in the actual answer
#             else:
#                 correct = expected_answer in actual_answer
#                 score = 1.0 if correct else 0.0
            
#             # Append the result for this question to the results list for Excel output
#             results.append({
#                 "Question": item['question'],
#                 "Expected Answer": item['expected_answer'],
#                 "Actual Answer": item['actual_answer'],
#                 "Score": f"{score:.2f}",
#                 "Correct": "Yes" if correct else "No"
#             })
            
#             # Append to y_true (expected correctness) and y_pred (actual correctness)
#             y_true.append(1)  # Always expecting the answer to be correct
#             y_pred.append(1 if correct else 0)  # Append 1 for "Yes" (correct) and 0 for "No" (incorrect)
    
#     # Calculate final overall accuracy and error rate
#     accuracy = accuracy_score(y_true, y_pred)
#     error_rate = 1 - accuracy
    
#     # Print the final results
#     print(f"Final Overall Accuracy: {accuracy:.2f}")
#     print(f"Final Overall Error Rate: {error_rate:.2f}")
    
#     # Create a DataFrame from the results list
#     df_results = pd.DataFrame(results)
    
#     # Save the DataFrame to an Excel file
#     df_results.to_excel(output_file, index=False)
    
#     return accuracy, error_rate

# # List of file paths to process
# file_paths = ['ollama_output_results_1.json','ollama_output_results_2.json','ollama_output_results_3.json','ollama_output_results_4.json','ollama_output_results_5.json']  # Replace with your file paths

# # Evaluate the answers, save the results to an Excel file, and display detailed results
# output_file = "evaluation_batch_Llama_2_accuracy.xlsx"  # Output Excel file path
# accuracy, error_rate = evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.40, batch_size=5, output_file=output_file)

# # Print final overall accuracy and error rate
# print("Final Overall Results:")
# print(f'Accuracy: {accuracy:.2f}')
# print(f'Error Rate: {error_rate:.2f}')
# print(f'Results saved to {output_file}')


# def evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.50, batch_size=5, output_file="evaluation_batch_results.xlsx"):
#     """Evaluates the actual answers from multiple files, displays results, and saves accuracy and error rate to Excel."""
#     y_true = []
#     y_pred = []
    
#     batch_y_true = []
#     batch_y_pred = []
    
#     results = []  # To store results for Excel output
#     batch_count = 0  # Track the number of batches
    
#     print("Detailed Results:\n")
    
#     # Iterate over all file paths and process each file
#     for file_path in file_paths:
#         data = load_json(file_path)
#         print(f"Processing file: {file_path}")
        
#         for idx, item in enumerate(data):
#             expected_answer = str(item['expected_answer']).lower()
#             actual_answer = str(item['actual_answer']).lower()
            
#             # Use the "Correct" column to check if the answer is correct or not
#             correct = item['correct'].strip().lower() == "yes"
            
#             # Append the result for this question to the results list for Excel output
#             results.append({
#                 "Question": item['question'],
#                 "Expected Answer": item['expected_answer'],
#                 "Actual Answer": item['actual_answer'],
#                 "Correct": "Yes" if correct else "No"
#             })
            
#             # Append to y_true (expected correctness) and y_pred (actual correctness)
#             y_true.append(1)  # We always expect the answer to be correct
#             y_pred.append(1 if correct else 0)  # Append 1 for "Yes" and 0 for "No"
#             batch_y_true.append(1)
#             batch_y_pred.append(1 if correct else 0)
            
#             # After every batch of batch_size questions, calculate accuracy and error rate for that batch
#             if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
#                 batch_count += 1
#                 batch_accuracy = accuracy_score(batch_y_true, batch_y_pred)
#                 batch_error_rate = 1 - batch_accuracy
                
#                 # Print batch results
#                 print(f"Batch {batch_count} Results:")
#                 print(f'Accuracy: {batch_accuracy:.2f}')
#                 print(f'Error Rate: {batch_error_rate:.2f}\n')
                
#                 # Append batch results to the results list
#                 results.append({
#                     "Question": f"Batch {batch_count} Results",
#                     "Expected Answer": "",
#                     "Actual Answer": "",
#                     "Correct": ""
#                 })
#                 results.append({
#                     "Question": "Accuracy",
#                     "Expected Answer": f"{batch_accuracy:.2f}",
#                     "Actual Answer": "",
#                     "Correct": ""
#                 })
#                 results.append({
#                     "Question": "Error Rate",
#                     "Expected Answer": f"{batch_error_rate:.2f}",
#                     "Actual Answer": "",
#                     "Correct": ""
#                 })
                
#                 # Clear batch lists for the next batch
#                 batch_y_true.clear()
#                 batch_y_pred.clear()
                
#                 # Every 5 batches, show overall accuracy and error rate so far
#                 if batch_count % 5 == 0 or idx == len(data) - 1:
#                     overall_accuracy = accuracy_score(y_true, y_pred)
#                     overall_error_rate = 1 - overall_accuracy
                    
#                     # Print overall results so far
#                     print(f"Overall Results after {batch_count} Batches:")
#                     print(f'Overall Accuracy: {overall_accuracy:.2f}')
#                     print(f'Overall Error Rate: {overall_error_rate:.2f}\n')
                    
#                     # Append overall results to the results list
#                     results.append({
#                         "Question": f"Overall Results after {batch_count} Batches",
#                         "Expected Answer": "",
#                         "Actual Answer": "",
#                         "Correct": ""
#                     })
#                     results.append({
#                         "Question": "Overall Accuracy",
#                         "Expected Answer": f"{overall_accuracy:.2f}",
#                         "Actual Answer": "",
#                         "Correct": ""
#                     })
#                     results.append({
#                         "Question": "Overall Error Rate",
#                         "Expected Answer": f"{overall_error_rate:.2f}",
#                         "Actual Answer": "",
#                         "Correct": ""
#                     })

#     # Final overall accuracy and error rate for all questions
#     accuracy = accuracy_score(y_true, y_pred)
#     error_rate = 1 - accuracy
    
#     # Create a DataFrame from the results list
#     df_results = pd.DataFrame(results)
    
#     # Save the DataFrame to an Excel file
#     df_results.to_excel(output_file, index=False)
    
#     return accuracy, error_rate

# # Example usage:
# file_paths = ['ollama_output_results_1.json','ollama_output_results_2.json','ollama_output_results_3.json','ollama_output_results_4.json','ollama_output_results_5.json']  # Replace with your file paths
# output_file = "evaluation_batch_results_accuracy.xlsx"  # Output Excel file path
# accuracy, error_rate = evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.40, batch_size=5, output_file=output_file)

# # Print final overall accuracy and error rate
# print("Final Overall Results:")
# print(f'Accuracy: {accuracy:.2f}')
# print(f'Error Rate: {error_rate:.2f}')
# print(f'Results saved to {output_file}')

from sklearn.metrics import accuracy_score

def evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.40, batch_size=5, output_file="evaluation_batch_results_accuracy.xlsx"):
    """Evaluates the actual answers from multiple files, displays results, and saves accuracy and error rate to Excel."""
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
            
            # After every batch of batch_size questions, calculate accuracy and error rate for that batch
            if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
                batch_count += 1
                batch_accuracy = accuracy_score(batch_y_true, batch_y_pred)
                batch_error_rate = 1 - batch_accuracy
                
                # Print batch results
                print(f"Batch {batch_count} Results:")
                print(f'Accuracy: {batch_accuracy:.2f}')
                print(f'Error Rate: {batch_error_rate:.2f}\n')
                
                # Append batch results to the results list
                results.append({
                    "Question": f"Batch {batch_count} Results",
                    "Expected Answer": "",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                results.append({
                    "Question": "Accuracy",
                    "Expected Answer": f"{batch_accuracy:.2f}",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                results.append({
                    "Question": "Error Rate",
                    "Expected Answer": f"{batch_error_rate:.2f}",
                    "Actual Answer": "",
                    "Score": "",
                    "Correct": ""
                })
                
                # Clear batch lists for next batch
                batch_y_true.clear()
                batch_y_pred.clear()
                
                # Every 5 batches, show overall accuracy and error rate so far
                # if batch_count % 5 == 0 or idx == len(data) - 1:
                #     overall_accuracy = accuracy_score(y_true, y_pred)
                #     overall_error_rate = 1 - overall_accuracy
                    
                #     # Print overall results so far
                #     print(f"Overall Results after {batch_count} Batches:")
                #     print(f'Overall Accuracy: {overall_accuracy:.2f}')
                #     print(f'Overall Error Rate: {overall_error_rate:.2f}\n')
                    
                #     # Append overall results to the results list
                #     results.append({
                #         "Question": f"Overall Results after {batch_count} Batches",
                #         "Expected Answer": "",
                #         "Actual Answer": "",
                #         "Score": "",
                #         "Correct": ""
                #     })
                #     results.append({
                #         "Question": "Overall Accuracy",
                #         "Expected Answer": f"{overall_accuracy:.2f}",
                #         "Actual Answer": "",
                #         "Score": "",
                #         "Correct": ""
                #     })
                #     results.append({
                #         "Question": "Overall Error Rate",
                #         "Expected Answer": f"{overall_error_rate:.2f}",
                #         "Actual Answer": "",
                #         "Score": "",
                #         "Correct": ""
                #     })

    # Final overall accuracy and error rate for all questions
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    
    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    df_results.to_excel(output_file, index=False)
    
    return accuracy, error_rate

# List of file paths to process
# file_paths = ['ollama_output_results_1.json','ollama_output_results_2.json','ollama_output_results_3.json','ollama_output_results_4.json','ollama_output_results_5.json']  # Replace with your file paths
#file_paths = ['gpt_output_results_1.json','gpt_output_results_2.json','gpt_output_results_3.json','gpt_output_results_4.json','gpt_output_results_5.json']  # Replace with your file paths
file_paths = ['gemini_output_results_1.json','gemini_output_results_2.json','gemini_output_results_3.json','gemini_output_results_4.json','gemini_output_results_5.json']  # Replace with your file paths
# Evaluate the answers, save the results to an Excel file, and display detailed results
output_file = "evaluation_batch_gemini_2_accuracy.xlsx"  # Output Excel file path
accuracy, error_rate = evaluate_answers_to_excel_with_accuracy(file_paths, threshold=0.40, batch_size=5, output_file=output_file)

# Print final overall accuracy and error rate
print("Final Overall Results:")
print(f'Accuracy: {accuracy:.2f}')
print(f'Error Rate: {error_rate:.2f}')
print(f'Results saved to {output_file}')

