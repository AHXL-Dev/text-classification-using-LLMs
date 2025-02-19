import streamlit as st
import pandas as pd
import json
from typing import List, Literal, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
import time
import datetime
import os
from dotenv import find_dotenv, load_dotenv



#dotenv_path = find_dotenv()
#load_dotenv(dotenv_path)

#API_KEY = os.getenv("API_KEY")

LABELS = Literal['TIME_WAITING', 'POLICY', 'SERVICE_PROCESS', 
                'QUALITY_OF_RESOLUTION', 'SELF_HELP_RESOURCES','AGENT_MANNERS', 
                'AGENT_KNOWLEDGE', 'TECHNOLOGY', 'REPEATED_FOLLOW_UP']


#Passing this into the prompt. Whether its more effective or not than just putting it into the prompt, I am not sure.
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


# Would like to later have this as a dataframe or a dataframe input maybe converted into a dictionary.
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
        },
        {
            "Row_ID": 11,
            "ProblemDescription": "Can I get an update on my refund request?",
            "Resolution": "There’s no update yet, but we’ll notify you soon.",
            "CustomerFeedback": "No real update, just delays in communication."
        },
        {
            "Row_ID": 12,
            "ProblemDescription": "How do I submit my request for special assistance during the flight?",
            "Resolution": "Refer to the assistance request page on our website.",
            "CustomerFeedback": "The webpage wasn’t user-friendly, and I couldn’t submit my request."
        },
        {
            "Row_ID": 13,
            "ProblemDescription": "Why was my request for priority boarding denied?",
            "Resolution": "Priority boarding is only available to premium passengers.",
            "CustomerFeedback": "I wasn’t told about the eligibility criteria before."
        },
        {
            "Row_ID": 14,
            "ProblemDescription": "Can you explain why my checked bag was delayed?",
            "Resolution": "Bag delays are often caused by high traffic during peak hours.",
            "CustomerFeedback": "Agent didn’t explain the specific cause for my delayed luggage."
        },
        {
            "Row_ID": 15,
            "ProblemDescription": "How do I get a copy of my flight receipt?",
            "Resolution": "You can download the receipt from the booking portal.",
            "CustomerFeedback": "The receipt wasn’t available on the portal as mentioned."
        },
        {
            "Row_ID": 16,
            "ProblemDescription": "Why is my seat upgrade request taking longer than expected?",
            "Resolution": "Upgrades are subject to availability and processing time.",
            "CustomerFeedback": "Agent didn’t explain the exact reason for the delay."
        },
        {
            "Row_ID": 17,
            "ProblemDescription": "How can I get help with using the airline’s mobile app?",
            "Resolution": "Please refer to the app user guide or attend our mobile app training session.",
            "CustomerFeedback": "No immediate support, had to wait for a session."
        },
        {
            "Row_ID": 18,
            "ProblemDescription": "Why was my frequent flyer status downgraded?",
            "Resolution": "Your status is based on recent travel activity, which may have decreased.",
            "CustomerFeedback": "No explanation of the specific travel activity that caused the downgrade."
        },
        {
            "Row_ID": 19,
            "ProblemDescription": "I need assistance with claiming my frequent flyer miles.",
            "Resolution": "Check the attached guide on how to claim miles from your recent flights.",
            "CustomerFeedback": "The guide was unclear and didn’t help with my claim."
        },
        {
            "Row_ID": 20,
            "ProblemDescription": "Why is my baggage fee higher than usual?",
            "Resolution": "Baggage fees are adjusted depending on flight routes and seasons.",
            "CustomerFeedback": "Agent didn’t explain why my fee was higher for this specific flight."
        },
        {
            "Row_ID": 21,
            "ProblemDescription": "How do I update my credit card details for booking?",
            "Resolution": "Update your card information through the booking portal.",
            "CustomerFeedback": "The portal didn’t allow me to update my details, poor service."
        },
        {
            "Row_ID": 22,
            "ProblemDescription": "I need help understanding my flight charge breakdown.",
            "Resolution": "Check the breakdown guide attached to your booking confirmation.",
            "CustomerFeedback": "The breakdown didn’t explain the charges I was concerned about."
        },
        {
            "Row_ID": 23,
            "ProblemDescription": "Why is it taking so long to process my flight change request?",
            "Resolution": "Flight changes depend on availability and schedule adjustments.",
            "CustomerFeedback": "No solution was offered to speed up the processing time."
        },
        {
            "Row_ID": 24,
            "ProblemDescription": "Can I get assistance with canceling my flight?",
            "Resolution": "Follow the cancellation steps outlined on our website.",
            "CustomerFeedback": "The website instructions were confusing and didn’t work."
        },
        {
            "Row_ID": 25,
            "ProblemDescription": "How do I register for the airline's loyalty program?",
            "Resolution": "Please visit our loyalty program registration page on the website.",
            "CustomerFeedback": "The page was hard to find and took too long to navigate."
        },
        {
            "Row_ID": 26,
            "ProblemDescription": "Can I change my seat selection after booking?",
            "Resolution": "You can modify seat selection through the booking portal.",
            "CustomerFeedback": "The portal was not responsive, and I couldn’t change my seat."
        },
        {
            "Row_ID": 27,
            "ProblemDescription": "Why does it take so long to get customer support for flight queries?",
            "Resolution": "Support is provided based on queue order and agent availability.",
            "CustomerFeedback": "Unacceptable wait time and poor customer service."
        },
        {
            "Row_ID": 28,
            "ProblemDescription": "How can I escalate my complaint about flight delays?",
            "Resolution": "Escalations are possible after 5 business days if not resolved.",
            "CustomerFeedback": "I wasn’t given immediate escalation options, felt ignored."
        },
        {
            "Row_ID": 29,
            "ProblemDescription": "Why do I need to resubmit my claim for flight damage compensation?",
            "Resolution": "Your previous claim was incomplete, please provide additional details.",
            "CustomerFeedback": "I wasn’t told what was missing, causing unnecessary delays."
        },
        {
            "Row_ID": 30,
            "ProblemDescription": "I need clarification on the airline's policy for pet travel.",
            "Resolution": "Please refer to the attached policy document for pet travel guidelines.",
            "CustomerFeedback": "The policy document was too vague and didn’t answer my questions."
        },
        {
            "Row_ID": 31,
            "ProblemDescription": "I need help getting my delayed baggage back.",
            "Resolution": "Your baggage is being tracked, and you will receive an update soon.",
            "CustomerFeedback": "No one followed up with me and the agent was very rude about it."
        },
        {
            "Row_ID": 32,
            "ProblemDescription": "Why is my seat not assigned despite checking in?",
            "Resolution": "Seats are assigned based on availability at check-in.",
            "CustomerFeedback": "Agent was dismissive and said it’s ‘just the way it is.’"
        }
    ]
    
    return sample_data


class TicketCategory(BaseModel):
    """Model for storing the classification results of a support ticket"""
    Row_ID: int
    categories: List[LABELS]
    sentiment: float = Field(ge=0, le=1,description='What is the degree of negative sentiment in the feedback (0 is neutral, 1 is very negative)')
    confidence: float = Field(ge=0, le=1,description='What is the degree of confidence in this classification (0 is very unconfident, 1 is very confident)')
    justification: str = Field(..., description='Explain why you selected these categories, referencing specific aspects of the ticket.')
    
def initialize_client():
    """Initialize OpenAI client once and store in the session state"""
    if 'client' not in st.session_state:
        st.session_state.client = instructor.patch(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.secrets["openrouter"]["api_key"]
                #base_url="http://localhost:11434/v1",
                #api_key="ollama"
            ),
            mode=instructor.Mode.JSON
        )

def classify_ticket(ticket: Dict) -> TicketCategory:
    """
    Classify a single support ticket using LLM
    Returns structured classification data
    """
    # Create a clear prompt for the LLM
    prompt = (
        f"Analyze this support ticket and theme the CustomerFeedback. The Field structures are as follows:\n\n"
        f"1. CUSTOMER'S INITIAL REQUEST:\n"
        f" {ticket['ProblemDescription']}\n"
        f" (This is what the customer initially asked for or needed help with)\n\n"
        f"2. AGENT'S RESOLUTION TO REQUEST:\n"
        f" {ticket['Resolution']}\n"
        f" (This is how the support agent attempted to resolve the request)\n\n"
        f"3. CUSTOMER'S FEEDBACK:\n"
        f" {ticket['CustomerFeedback']}\n"
        f" (This is the customer's reaction to the support ticket, reflecting their satisfaction or dissatisfaction with the support experience, how they felt about the resolution, and any other comments regarding the service received)\n\n"
        f"Ticket ID: {ticket['Row_ID']}\n\n"
        f"Instructions for Classification:\n"
        f"1. FOCUS PRIMARILY ON THE CUSTOMERFEEDBACK AS IT IS THE PRIMARY DRIVER OF THE CLASSIFICATION\n"
        f"2. YOU MUST USE THE {CATEGORY_DEFINITIONS} to guide your classification. CONSIDER THESE CAREFULLY AND ADHERE TO THE DEFINITIONS WHEN CLASSIFYING.\n" 
        f"3. Compare the initial request (ProblemDescription) with the resolution provided and USE THIS AS CONTEXT\n"
        f"4. Provide classification that reflects one or more of the following categories that best describe the customer's feedback. If multiple categories apply, please list them all (separate categories with commas):\n\n"
    )
    
    return st.session_state.client.chat.completions.create(
        #model="mistral:latest",
        model = "mistralai/mistral-small-24b-instruct-2501:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=TicketCategory  # Must confirm to our pydantic TicketCategory model
    )

def create_streamlit_app():
    """Create the Streamlit interface"""
    st.set_page_config(layout="wide")
    initialize_client()

    if 'tickets' not in st.session_state:
        st.session_state.tickets = create_sample_data() #So it persists in the session.


    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Introduction", "Process Tickets"])
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
        
        st.header("Classification Page")
        st.markdown("""
        This page showcases how the AI classifies customer support tickets using the **mock dataset**. The system automatically categorizes each ticket based on the customer's feedback, problem description, and agent's resolution.

        The categories used in this classification are designed to reflect different aspects of customer service and feedback.  
        This is a **fake dataset** with **30 number of rows** of mock tickets.

        ### Categories Used in Classification:
        - **TIME_WAITING**: Feedback about long wait times or delays.
        - **POLICY**: Issues with company policies or procedures.
        - **SERVICE_PROCESS**: Feedback on how services or tasks are delivered.
        - **QUALITY_OF_RESOLUTION**: Effectiveness of the resolution provided.
        - **SELF_HELP_RESOURCES**: Issues with self-help materials like guides or manuals.
        - **AGENT_MANNERS**: Feedback on the agent’s demeanor or professionalism.
        - **AGENT_KNOWLEDGE**: Issues with the agent’s knowledge or expertise.
        - **TECHNOLOGY**: Feedback on the systems, tools, or technology used.
        - **REPEATED_FOLLOW_UP**: Feedback about ongoing communication or lack of updates.
        """)
    else: 
        st.title("Process Support Tickets")
        st.subheader("This app is using a MOCK/fake dataset, this is NOT based on any real data, it is completely MADE UP")
        if st.button("Process Sample Tickets"):
            process_start_time = time.time()
            tickets = st.session_state.tickets
            results = []
            progress_bar = st.progress(0)
            for i, ticket in enumerate(tickets):
                try:
                    progress = (i + 1) / len(tickets) # Because i starts at 0 this is why we need i + 1
                    progress_bar.progress(progress)
                    result = classify_ticket(ticket)
                    results.append(result)
                except Exception as e:
                    st.error(f"Error processing ticket {ticket['Row_ID']}: {str(e)}")
            
            if results:
                results_df = pd.DataFrame([
                    {
                        'Row_ID': r.Row_ID,
                        'Problem': tickets[i]['ProblemDescription'],
                        'Resolution': tickets[i]['Resolution'],
                        'Feedback': tickets[i]['CustomerFeedback'],
                        'Categories': ', '.join(r.categories), #because this returns as a list, so need to convert to string
                        'Sentiment': r.sentiment,
                        'Confidence': r.confidence,
                        'Justification': r.justification
                    }
                    for i, r in enumerate(results)
                ])
            
            process_end_time = time.time()
            process_duration = process_end_time - process_start_time
            print(f"\nTicket processing completed")
            print(f"Total execution time: {process_duration:.2f} seconds")
            print(f"Successfully processed {len(results)} out of {len(tickets)} tickets")
            print(f"Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.subheader("Classification Results")
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "classification_results.csv",
                "text/csv"
            )

if __name__ == "__main__":
    create_streamlit_app()