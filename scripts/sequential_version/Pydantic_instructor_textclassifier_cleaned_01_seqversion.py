import streamlit as st
import pandas as pd
from typing import List, Literal, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
import time
import os
from dotenv import find_dotenv, load_dotenv
import logging
import requests

logging.basicConfig(level=logging.INFO)
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)



if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
    
def increment_counter():
    st.session_state.usage_count += 1
    st.session_state.button_clicked = True

API_KEY = os.getenv("API_KEY")

MODEL_NAME = "mistral:latest"
#MODEL_NAME = "mistralai/mistral-small-24b-instruct-2501:free"

LABELS = Literal['TIME_WAITING', 'POLICY', 'SERVICE_PROCESS', 
                'QUALITY_OF_RESOLUTION', 'SELF_HELP_RESOURCES','AGENT_MANNERS', 
                'AGENT_KNOWLEDGE', 'TECHNOLOGY', 'REPEATED_FOLLOW_UP']

# Category definitions

CATEGORY_DEFINITIONS = """
    'TIME_WAITING': 'Feedback that EXPLICITY mentions long call waiting times, call queue lengths, or delays in receiving responses. This includes complaints about waiting for responses or resolution timeframes.'
    'POLICY': 'Feedback related to company policies, rules, or standard procedures that affect service delivery. This includes cases where policies are unclear, seem unfair, or limit service options.'
    'SERVICE_PROCESS': 'Feedback related to how processes in services are delivered or tasks are completed. This includes difficult processes or complicated workflows.'
    'QUALITY_OF_RESOLUTION': 'Feedback related to the customer\'s problem not being resolved, or answered. This also includes where the customer indicates the resolution was generic or incomplete.'
    'SELF_HELP_RESOURCES': 'Feedback related to QRG, website links, documentation, user guides, manuals, or other self-service materials. This includes unclear instructions, missing information, or difficult-to-use resources.'
    'AGENT_MANNERS': 'Feedback that EXPLICITLY mentions the agent\'s poor behavior towards customers. This includes specific mentions of rudeness, lack of empathy, being abrupt, dismissive, or any other unprofessional conduct. Do NOT apply this category for general complaints about resolution quality or service process.'
    'AGENT_KNOWLEDGE': 'Feedback related to the agent\'s expertise or understanding. This includes incorrect information, inability to explain clearly, or lack of technical knowledge.'
    'TECHNOLOGY': 'Feedback related to systems, software, or technical infrastructure. This includes system errors, software bugs, or lack of ease of use with digital tools.'
    'REPEATED_FOLLOW_UP': 'Feedback that EXPLICITLY mentions the customer having to follow-up multiple times on a request.'
"""



def create_sample_data():
    sample_data = [
        {
            "Row_ID": 1,
            "ProblemDescription": "I want to know how much my flight bonus will be for next year.",
            "Resolution": "Here’s a link to the bonus calculation guidelines.",
            "CustomerFeedback": "I wasn’t contacted and got a generic response with no specifics."
        },
        {
            "Row_ID": 2,
            "ProblemDescription": "Why is my ticket cost higher than usual?",
            "Resolution": "The price reflects new seasonal rates.",
            "CustomerFeedback": "I don’t understand why the cost increased. No detailed explanation."
        },
        {
            "Row_ID": 3,
            "ProblemDescription": "When will my refund for the canceled flight be processed?",
            "Resolution": "Refunds typically take 4-6 weeks to process.",
            "CustomerFeedback": "I received a generic timeline. No specific details for my case."
        },
        {
            "Row_ID": 4,
            "ProblemDescription": "I can’t access my flight booking details.",
            "Resolution": "You can access them directly on the website by logging in.",
            "CustomerFeedback": "The website is not working, and the advice didn’t help."
        },
        {
            "Row_ID": 5,
            "ProblemDescription": "How do I reset my password for my booking account?",
            "Resolution": "Follow the steps in the attached guide to reset your password.",
            "CustomerFeedback": "The guide wasn’t clear, I couldn’t reset my password."
        },
        {
            "Row_ID": 6,
            "ProblemDescription": "Why hasn’t my refund been processed yet?",
            "Resolution": "Refunds are processed at the end of each month.",
            "CustomerFeedback": "I wasn’t informed why my refund was delayed."
        },
        {
            "Row_ID": 7,
            "ProblemDescription": "Can you explain the process for claiming a flight delay compensation?",
            "Resolution": "Please refer to the compensation policy document linked here.",
            "CustomerFeedback": "The document is too complicated, and I didn’t find any clear answers."
        },
        {
            "Row_ID": 8,
            "ProblemDescription": "I need help with making a group booking for my department.",
            "Resolution": "Here’s a guide for making group bookings on our website.",
            "CustomerFeedback": "The guide missed some important steps for large group bookings."
        },
        {
            "Row_ID": 9,
            "ProblemDescription": "Why hasn’t my booking for an extra seat been processed?",
            "Resolution": "It takes up to 48 hours to confirm extra seat requests.",
            "CustomerFeedback": "Agent didn’t check my specific request, just gave a standard response."
        },
        {
            "Row_ID": 10,
            "ProblemDescription": "I need help with changing my flight date.",
            "Resolution": "Here’s the link to change your booking online.",
            "CustomerFeedback": "The website process was unclear and I couldn’t change my flight."
        }]
    
    return sample_data
    

class SearchAnalysis(BaseModel):
    match: Literal['YES', 'NO'] = Field(..., description='Whether the ticket matches the search criteria')
    confidence: float = Field(..., ge=0, le=1, description='How confident are you with the match? (0-1)')
    reason: str = Field(..., description='Explanation for the match decision')
    Row_ID: int
    
class TicketClassification(BaseModel):
    '''Classification model for support tickets'''
    
    Category: List[LABELS] = Field(..., description='''
        Analyze the CustomerFeedback and select one or more labels that apply to categorise the feedback. Use the ProblemDescription and Resolution to provide context.
        Choose categories that best match the customer's feedback about their experience.
    ''')
    justification: str = Field(..., description='Explain why you selected these categories, referencing specific aspects of the ticket.')
    sentiment: float = Field(ge=0, le=1, description='What is the degree of negative sentiment in the feedback (0-1)')
    confidence: float = Field(ge=0, le=1, description='How confident are you in this classification (0-1)')
    Row_ID: int

def initialize_client():
    if 'client' not in st.session_state:
        st.session_state.client = instructor.patch(
            OpenAI(
                #base_url="https://openrouter.ai/api/v1",
                #api_key=API_KEY
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            ),
            mode=instructor.Mode.JSON)
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
        
def classify_single_ticket(ticket: Dict) -> TicketClassification:
        for attempt in range(3):
            try:
                prompt = (
                f"Analyze this support ticket and theme the CustomerFeedback. The Field structures are as follows:\n\n"
                f"1. CUSTOMER'S INITIAL REQUEST:\n"
                f"   {ticket['ProblemDescription']}\n"
                f"   (This is what the customer initially asked for or needed help with)\n\n"
                f"2. AGENT'S RESOLUTION TO REQUEST:\n"
                f"   {ticket['Resolution']}\n"
                f"   (This is how the support agent attempted to resolve the request)\n\n"
                f"3. CUSTOMER'S FEEDBACK:\n"
                f"   {ticket['CustomerFeedback']}\n"
                f"   (This is the customer's reaction to the support ticket, reflecting their satisfaction or dissatisfaction with the support experience, how they felt about the resolution, and any other comments regarding the service received)\n\n"
                f"Ticket ID: {ticket['Row_ID']}\n\n"
                f"Instructions for Classification:\n"
                f"1. FOCUS PRIMARILY ON THE CUSTOMERFEEDBACK AS IT IS THE PRIMARY DRIVER OF THE CLASSIFICATION\n"
                f"2. YOU MUST USE THE {CATEGORY_DEFINITIONS} to guide your classification. CONSIDER THESE CAREFULLY AND ADHERE TO THE DEFINITIONS WHEN CLASSIFYING.\n" 
                f"3. Compare the initial request (ProblemDescription) with the resolution provided and USE THIS AS CONTEXT\n"
                f"4. Provide classification that reflects one or more of the following categories that best describe the customer's feedback. If multiple categories apply, please list them all (separate categories with commas):\n\n"
            )
                result = st.session_state.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_model=TicketClassification
                )
                return result
        
            except Exception as e:
                if attempt == 2:
                    result = TicketClassification(
                        Row_ID=ticket['Row_ID'],
                        Category=[],
                        sentiment=0,
                        confidence=0,
                        justification=f"Classification failed after 3 attempts: {str(e)}"
                    )
                    return result

            time.sleep(1)



def analyze_ticket(ticket: Dict, search_criteria: str, example_phrases: str) -> SearchAnalysis:
        for attempt in range(3):
            try:
                prompt = (f"""
                        Analyze if this ticket matches the search criteria.

                        SEARCH CRITERIA: {search_criteria}
                        EXAMPLE PHRASES: {example_phrases if example_phrases else 'None provided'}

                        TICKET INFORMATION:
                        Ticket ID: {ticket['Row_ID']}
                        Problem: {ticket['ProblemDescription']}
                        Resolution: {ticket['Resolution']}
                        Feedback: {ticket['CustomerFeedback']}

                        Consider:
                        1. Direct matches with search criteria
                        2. Context from problem description and resolution
                        3. Example phrases provided (if any)

                        Provide your analysis in this format:
                        MATCH: [YES/NO]
                        CONFIDENCE: [0-100]
                        REASON: [Brief explanation of why this matches or doesn't match]
                        KEY_THEMES: [Main topics/issues identified]
                        """)
                result_custom = st.session_state.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_model=SearchAnalysis
                )
                print(result_custom)
                return result_custom
          
            except Exception as e:
                if attempt == 2: 
                    result_custom = SearchAnalysis(
                        match="NO",
                        confidence=0.0,
                        reason=f"Analysis failed: {str(e)}",
                        Row_ID=ticket['Row_ID']
                    )
                    return result_custom
            time.sleep(1) 


            
def process_tickets(all_tickets,number:int,search_criteria = "none", example_phrases = "none"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    classified_results = []
    
    for i in all_tickets:
        progress = i['Row_ID'] / len(all_tickets)
        status_text.text(f"Processing ticket {i['Row_ID']}...")
        progress_bar.progress(progress)
        if number == 1:
            classified_results.append(classify_single_ticket(i))
        elif number == 2:
            print(i)
            print(search_criteria)
            print(example_phrases)
            classified_results.append(analyze_ticket(i, search_criteria, example_phrases))
        
    return classified_results

def create_streamlit_app():
    st.set_page_config(layout="wide")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Introduction", "Process Tickets", "Search for theme"])
    initialize_client()
    
    # Display usage count and limit information in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Usage Limit")
    max_attempts = 2
    remaining_attempts = max_attempts - st.session_state.usage_count
    st.sidebar.write(f"Remaining attempts: {remaining_attempts}")
    
    if st.session_state.usage_count >= max_attempts:
        st.sidebar.error("You have reached the maximum number of attempts for this session.")

    if 'tickets' not in st.session_state:
        st.session_state.tickets = create_sample_data()
    tickets = st.session_state.tickets
    
    st.title("Process Support Tickets")
    st.subheader("This app is using a MOCK/fake dataset")

    if page == "Introduction":
        st.title("Text Classification using Large Language Models – Demonstration")
        st.markdown("*Created by Asanka*")
        
        st.header("Key Points:")
        st.subheader("This app is using a MOCK/fake dataset, this is NOT based on any real data, it is completely MADE UP")
        st.markdown("""
        * This app is a demonstration of how Generative AI can be used to classify customer feedback based on mock customer support tickets. The AI model analyzes various parts of the ticket (Problem Description, Resolution, and Customer Feedback) to categorize the feedback and determine customer sentiment.**
        * The app runs on a local LLM (Large Language Model), which is downloaded and hosted locally, meaning **no data is sent to an external model provider**. The model being used is Mistral AI 7b model, a lightweight model with 7 billion parameters.**
        * I am currently hosting this on a particular platform. I have not created this app for production , **this is purely based on my interest and learning of Gen AI applications**
        * This is purely for demonstration, and there are further ways to enhance the program or use a stronger model.
        """)
    elif page == "Process Tickets":
        # Disable the button if max attempts reached
        button_disabled = st.session_state.usage_count >= max_attempts
        
        # Use on_click to increment the counter BEFORE running the main logic
        if st.button("Process Sample Tickets", disabled=button_disabled, on_click=increment_counter):
            # Only increment if we're still under the limit (defensive check)
            if st.session_state.button_clicked:
                st.session_state.button_clicked = False
                
                start_time = time.time()
                results = process_tickets(tickets,1)
                results_df = pd.DataFrame([
                    {
                        'Row_ID': r.Row_ID,
                        'Categories': ', '.join(r.Category),
                        'Sentiment': r.sentiment,
                        'Confidence': r.confidence,
                        'Justification': r.justification
                    }
                    for r in results
                ])
                tickets_df = pd.DataFrame(st.session_state.tickets)
                final_df = pd.merge(
                    tickets_df[['Row_ID', 'ProblemDescription', 'Resolution', 'CustomerFeedback']],
                    results_df,
                    on='Row_ID',
                    how='inner'
                )
                
                if final_df['Categories'].isnull().any():
                    st.button("Reprocess Sample Tickets", on_click="")
                
                st.subheader("Classification Results")
                st.dataframe(final_df)
                csv = final_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "classification_results.csv",
                    "text/csv"
                )
                end_time = time.time()
                st.success(f"Processed {len(results)} tickets in {end_time - start_time:.2f} seconds")
                
                # Update remaining attempts display after processing
        if st.session_state.usage_count >= max_attempts:
                    st.error("You have reached the maximum number of attempts (2) for this session. Please refresh the page to start a new session.")
        elif st.session_state.usage_count > 0:
                    st.info(f"You have {remaining_attempts} attempt(s) remaining for this session.")
                
    elif page == "Search for theme":
        st.title("Search for theme")
        issue_description = st.text_area(
                "What kind of feedback are you looking for?",
                placeholder="Example: System being slow and unresponsive"
            )

        example_phrases = st.text_area(
            "Optional: Add example phrases",
            placeholder="Example phrases that customers might use"
        )
        # Disable the search button if max attempts reached
        search_button_disabled = st.session_state.usage_count >= max_attempts
        if st.button("Process Sample Tickets", disabled=search_button_disabled, on_click=increment_counter) and issue_description:
            if st.session_state.button_clicked:
                st.session_state.button_clicked = False
                start_time = time.time()
                results2 = process_tickets(tickets,2,issue_description,example_phrases)
                print(results2)
                results2_df = pd.DataFrame([{
                    'Row_ID': r.Row_ID,
                    'Match': r.match,
                    'Confidence': r.confidence,
                    'Reason': r.reason,
                } for r in results2])
                print(results2)
                tickets_df_2 = pd.DataFrame(st.session_state.tickets)
                final_df_2 = pd.merge(
                    tickets_df_2[['Row_ID', 'ProblemDescription', 'Resolution', 'CustomerFeedback']],
                    results2_df,
                    on='Row_ID',
                    how='inner'
                )
                st.subheader("Classification Results")
                st.dataframe(final_df_2)
                csv = results2_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "specific_search.csv",
                    "text/csv"
                )
                end_time = time.time()
                st.success(f"Processed {len(results2)} tickets in {end_time - start_time:.2f} seconds")
        if st.session_state.usage_count >= max_attempts:
            st.error("You have reached the maximum number of attempts (2) for this session. Please refresh the page to start a new session.")
        elif st.session_state.usage_count > 0:
            st.info(f"You have {remaining_attempts} attempt(s) remaining for this session.")
    

if __name__ == "__main__":
    create_streamlit_app()