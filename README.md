# Text Classification Using LLMs
This repository contains programs for text classification tasks utilizing open-source large language models (LLMs).

## Overview
There are two main versions of the program: one using Jupyter Notebooks and the other using Python scripts.

## Notebooks
- **Classifier with Swarm** (Not successful): This version attempted to integrate Swarm with the classifier, but it did not produce successful results.
- **Classifier without Swarm**: A simpler version of the classifier, tested with a local LLM (Deepseek R1:8b). This version yielded promising results using a binary 'Yes/No' classifier, which executes for each theme on each row. It works particularly well with reasoning models.

## Scripts
1. **Pydantic_instructor_classification_prototype_mistral_06.py**: 
   - Utilizes Pydantic and Instructor with both batch and async processing.
   - Includes a separate class method for theme category definitions, with a slightly different prompt.
   - Tested with Mistral:latest running locally, and the results were decent.

2. **Pydantic_instructor_text_classifier_cleaned_01.py**: 
   - Also uses Pydantic and Instructor with batch and async processing.
   - Removes the class method for theme category definitions, directly referencing the category definitions in the prompt.
   - After further investigation, I found that using a class method could complicate things by presenting category definitions as a dictionary (key-value pairs), which might confuse the model. I opted to merge the category definitions into one string, which improved the results.
   - Tested with Mistral:latest running locally. Since batch processing is used, this version works better with models like LLaMA and Mistral, but not as well with Deepseek.

3. **text_classification_structured_inputs_demo.py**: 
   - This is the current live demo version.
   - Based on `Pydantic_instructor_text_classifier_cleaned_01.py`, but it uses models accessed via an API.
