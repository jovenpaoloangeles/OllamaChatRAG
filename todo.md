Analyze @evaluation_results_20250320_131902.json, which contains true_answer. Compare the accuracy of each model_answers against true_answer.

Faithfulness Evaluation: Use Semantic Similarity Scoring.
Accuracy Evaluation: Use LLM-as-Judge (Gemini Pro 2.0).
Time Analysis: Compute the average time_taken per model.
Plot Generation: Create visualizations for faithfulness, accuracy, and average time per model.

If insufficient information is available to answer a question, assign a score of 0 automatically without processing through the LLM or semantic scoring.
Remove <thought>...</thought> blocks and their contents.
Strip unnecessary newline characters (\n) and other formatting artifacts.