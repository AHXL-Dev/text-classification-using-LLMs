# Text Classification Using LLMs
This repository contains programs for text classification tasks utilising open-source large language models (LLMs).

## Scripts

This section contains two primary folders for processing: one for **batch processing** and one for **sequential processing**. Each folder contains two versions of the same script – one version using the `Pydantic_instructor_*` naming convention and the other being the live version published to Streamlit (`text_classification_*`).

### **Batch Processing Folder**
1. **Pydantic_instructor_batch_classifier.py**:
   - Uses **Pydantic** and **Instructor** for batch processing with async capabilities.
   - Processes a list of tickets concurrently, optimizing the classification tasks by handling multiple requests simultaneously. This script aims to speed up processing through asynchronous execution.

2. **text_classification_batch_streamlit.py**:
   - This is the live Streamlit app version of the batch processing script.
   - It’s essentially the same as the above file but implemented with the Streamlit interface for end users.

**Note on Batch Processing with API**: After further research, I discovered that despite the script being designed for **batch processing**, the **API call limitation** (approximately one API call per second) means that the batch processing does not outperform sequential processing when relying on APIs like OpenAI or others with rate limits. Even though we use asynchronous requests to process multiple tickets at once, the bottleneck remains at the API level, as each request is processed in one-second intervals. This means that for API calls, the true benefit of async is diminished in scenarios where rate limits restrict parallelism. 

### **Sequential Processing Folder**
1. **Pydantic_instructor_sequential_classifier.py**:
   - A sequential version of the classifier using **Pydantic** and **Instructor**.
   - Each ticket is processed one at a time, without the concurrency features in the batch version. This can be useful for smaller sets of tickets or if you're processing tickets in order, such as for step-by-step verification.

2. **text_classification_sequential_streamlit.py**:
   - This version is the live Streamlit app version for sequential processing.
   - Similar to the batch version but using sequential processing to handle each ticket individually.


## Notebooks

- **Classifier with Swarm** (Not successful): This version attempted to integrate Swarm with the classifier, but it did not produce successful results.
- **Classifier without Swarm**: A simpler version of the classifier, tested with a local LLM (Deepseek R1:8b). This version yielded promising results using a binary 'Yes/No' classifier, which executes for each theme on each row. It works particularly well with reasoning models.
