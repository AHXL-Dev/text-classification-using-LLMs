import asyncio
from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import pandas as pd
import instructor
import logging
import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
 


# Configure logging
logging.basicConfig(level=logging.INFO)


# Constants
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

API_KEY = os.getenv("API_KEY")

#MODEL_NAME = "mistral:latest"
MODEL_NAME = "mistralai/mistral-small-24b-instruct-2501:free"

LABELS = Literal['TIME_WAITING', 'POLICY', 'SERVICE_PROCESS', 
                'QUALITY_OF_RESOLUTION', 'SELF_HELP_RESOURCES','AGENT_MANNERS', 
                'AGENT_KNOWLEDGE', 'TECHNOLOGY', 'REPEATED_FOLLOW_UP']

# Category definitions
CATEGORY_DEFINITIONS = {
    'TIME_WAITING': 'Feedback that EXPLICITY mentions long call waiting times, call queue lengths, or delays in receiving responses. This includes complaints about waiting for responses or resolution timeframes.',
    
    'POLICY': 'Feedback related to company policies, rules, or standard procedures that affect service delivery. This includes cases where policies are unclear, seem unfair, or limit service options.',
    
    'SERVICE_PROCESS': 'Feedback related to how processes in services are delivered or tasks are completed. This includes difficult processes or complicated workflows.',
    
    'QUALITY_OF_RESOLUTION': 'Feedback related to the customer\'s problem not being resolved, or answered. This also includes where the customer indicates the resolution was generic or incomplete.',
    
    'SELF_HELP_RESOURCES': 'Feedback related to QRG, website links, documentation, user guides, manuals, or other self-service materials. This includes unclear instructions, missing information, or difficult-to-use resources.',
    
    'AGENT_MANNERS': 'Feedback that EXPLICITLY mentions the agent\'s poor behavior towards customers. This includes specific mentions of rudeness, lack of empathy, being abrupt, dismissive, or any other unprofessional conduct. Do NOT apply this category for general complaints about resolution quality or service process.',
    
    'AGENT_KNOWLEDGE': 'Feedback related to the agent\'s expertise or understanding. This includes incorrect information, inability to explain clearly, or lack of technical knowledge.',
    
    'TECHNOLOGY': 'Feedback related to systems, software, or technical infrastructure. This includes system errors, software bugs, or lack of ease of use with digital tools.',
    
    'REPEATED_FOLLOW_UP': 'Feedback that EXPLICITLY mentions the customer having to follow-up multiple times on a request.'
}

CATEGORY_PROMPT = "\n".join(
    [f"- {cat}: {desc}" for cat, desc in CATEGORY_DEFINITIONS.items()]
)



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

    
class TicketClassification(BaseModel):
    '''Classification model for support tickets'''
    
    class Category(BaseModel):
        Category: List[LABELS] = Field(..., description='''
            Analyze the CustomerFeedback and select one or more labels that apply to categorise the feedback. Use the ProblemDescription and Resolution to provide context.
            Choose categories that best match the customer's feedback about their experience.
        ''')
        justification: str = Field(..., description='Explain why you selected these categories, referencing specific aspects of the ticket.')
        sentiment: float = Field(ge=0, le=1, description='What is the degree of negative sentiment in the feedback (0-1)')
        confidence: float = Field(ge=0, le=1, description='How confident are you in this classification (0-1)')
        Row_ID: int

class SimplifiedBatchProcessor:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.client = instructor.patch(
            AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.secrets["openrouter_key"]["api_key"]
                #base_url="http://localhost:11434/v1",
                #api_key="ollama"
            ),
            mode=instructor.Mode.JSON
        )

    async def classify_ticket(self, ticket: dict) -> TicketClassification.Category:
        """Classify a single support ticket with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
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
                    f"2. YOU MUST USE THE {CATEGORY_PROMPT} to guide your classification. CONSIDER THESE CAREFULLY AND ADHERE TO THE DEFINITIONS WHEN CLASSIFYING.\n" 
                    f"3. Compare the initial request (ProblemDescription) with the resolution provided and USE THIS AS CONTEXT\n"
                    f"4. Provide classification that reflects one or more of the following categories that best describe the customer's feedback. If multiple categories apply, please list them all (separate categories with commas):\n\n"
                )
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    response_model=TicketClassification.Category,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to classify ticket after {max_retries} attempts. Error: {str(e)}")
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def process_batch(self, tickets: List[dict]) -> List[TicketClassification.Category]:
        """Process a batch of tickets concurrently"""
        tasks = [self.classify_ticket(ticket) for ticket in tickets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def process_all_tickets(self, tickets: List[dict]) -> pd.DataFrame:
        """Process all tickets in batches and return results as DataFrame"""
        all_results = []
        total_batches = (len(tickets) + self.batch_size - 1) // self.batch_size
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, batch_start in enumerate(range(0, len(tickets), self.batch_size)):
            batch = tickets[batch_start:batch_start + self.batch_size]
            status_text.text(f"Processing batch {i+1} of {total_batches}...")
            
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            print(all_results)
            
            # Update progress
            progress = (i + 1) / total_batches
            progress_bar.progress(progress)

        status_text.text("Processing complete!")
        
        results_df = pd.DataFrame([
            {
                'Row_ID': result.Row_ID,
                'Categories': ', '.join(result.Category),
                'Sentiment': result.sentiment,
                'Confidence': result.confidence,
                'Justification': result.justification
            }
            for result in all_results
        ])
        
        
        tickets_df = pd.DataFrame(tickets)
        
        
        final_df = pd.merge(
            tickets_df[['Row_ID', 'ProblemDescription', 'Resolution', 'CustomerFeedback']], 
            results_df,
            on='Row_ID',
            how='inner'
        )
        
        return final_df.sort_values('Row_ID')
def format_category_name(Category: str) -> str:
    """Convert a category name to the correct format (TEXT_LIKE_THIS)"""
    
    formatted = ''.join(c for c in Category if c.isalnum() or c.isspace())
    return formatted.upper().replace(' ', '_')

def create_streamlit_app():
    st.set_page_config(layout="wide")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction","Dataset Preview" ,"Classification Demo","Custom Category Search"])
    
    
    if 'processor' not in st.session_state:
        st.session_state.processor = SimplifiedBatchProcessor(batch_size=5)
    
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
        
        st.header("Custom Category Search Page")
        st.markdown("""
        This page allows you to utilise the GenAI model to search for specific categories of feedback using the **mock dataset**. This can be useful for finding feedback that aligns with your specific requirements or goals.
        """)
    elif page == "Dataset Preview":
        st.title("Sample Dataset Preview (first 5 rows)")
        st.subheader("This app is using a MOCK/fake dataset, this is NOT based on any real data, it is completely MADE UP")
        st.sidebar.header("Configuration")
        st.markdown("""
        This page shows the **first 5 rows** from the dataset being used
        """)
        
        # Create and display sample data
        
        df = pd.DataFrame(create_sample_data())
        
        st.dataframe(df.head(5))
    
    elif page == "Custom Category Search":
        st.title("Custom Category Search")
        
        issue_description = st.text_area(
            "What kind of feedback are you looking for?",
            placeholder="Example: System being slow and unresponsive"
        )

        example_phrases = st.text_area(
            "Optional: Add example phrases",
            placeholder="Example phrases that customers might use"
        )

        if st.button("Search Tickets") and issue_description:
            with st.spinner("Analyzing tickets..."):
                sample_data = create_sample_data()
                
                async def analyze_ticket(ticket: Dict[str, Any], search_criteria: str) -> Dict[str, Any]:
                    prompt = f"""
                    Analyze if this ticket matches the search criteria.

                    SEARCH CRITERIA: {search_criteria}
                    EXAMPLE PHRASES: {example_phrases if example_phrases else 'None provided'}

                    TICKET INFORMATION:
                    Problem: {ticket['ProblemDescription']}
                    Resolution: {ticket['Resolution']}
                    Feedback: {ticket['CustomerFeedback']}

                    Consider:
                    1. Direct matches with search criteria
                    2. Similar issues or related problems
                    3. Context from problem description and resolution
                    4. Example phrases provided (if any)

                    Provide your analysis in this format:
                    MATCH: [YES/NO]
                    CONFIDENCE: [0-100]
                    REASON: [Brief explanation of why this matches or doesn't match]
                    KEY_THEMES: [Main topics/issues identified]
                    """

                    response = await st.session_state.processor.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing support tickets and identifying patterns in customer feedback."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    
                    return {
                        'Row_ID': ticket['Row_ID'],
                        'Problem': ticket['ProblemDescription'],
                        'Resolution': ticket['Resolution'],
                        'Feedback': ticket['CustomerFeedback'],
                        'Analysis': response.choices[0].message.content
                    }

                # Process tickets
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tasks = [analyze_ticket(ticket, issue_description) for ticket in sample_data]
                    results = loop.run_until_complete(asyncio.gather(*tasks))
                finally:
                    loop.close()

                # Create DataFrame and filter for matches
                df = pd.DataFrame(results)
                
                # Create tabs for results and analysis
                results_tab, analysis_tab = st.tabs(["Results", "Analysis"])
                
                with results_tab:
                    # Display full results table
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="search_results.csv",
                        mime="text/csv"
                    )
                
                with analysis_tab:
                    st.markdown("### Key Findings")
                    
                    # Count matches
                    matches = sum(1 for analysis in df['Analysis'] if 'MATCH: YES' in analysis.upper())
                    
                    # Display summary statistics
                    st.metric("Total Tickets Analyzed", len(df))
                    st.metric("Matching Tickets Found", matches)
                    
                    if matches > 0:
                        # Display matching tickets with their analysis
                        st.markdown("### Matching Tickets")
                        for _, row in df.iterrows():
                            if 'MATCH: YES' in row['Analysis'].upper():
                                with st.expander(f"Ticket {row['Row_ID']}"):
                                    st.write("**Problem:**", row['Problem'])
                                    st.write("**Feedback:**", row['Feedback'])
                                    st.write("**Analysis:**", row['Analysis'])
                                    
                        # Add visualization if you have enough matches
                        if matches >= 3:
                            st.markdown("### Confidence Distribution")
                            # Extract confidence scores
                            confidences = []
                            for analysis in df['Analysis']:
                                if 'MATCH: YES' in analysis.upper():
                                    for line in analysis.split('\n'):
                                        if 'CONFIDENCE:' in line.upper():
                                            try:
                                                confidence = int(line.split(':')[1].strip().split()[0])
                                                confidences.append(confidence)
                                            except ValueError:
                                                continue
                            
                            if confidences:
                                confidence_df = pd.DataFrame({
                                    'Confidence': confidences
                                })
                                st.bar_chart(confidence_df['Confidence'])
                    else:
                        st.info("No matching tickets found for your search criteria. Try adjusting your search terms or example phrases.")
            
    else:
        st.title("Support Ticket Classification System")
        st.subheader("This app is using a MOCK/fake dataset, this is NOT based on any real data, it is completely MADE UP")
        st.sidebar.header("Configuration")
        batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=10, value=5)
        st.session_state.processor.batch_size = batch_size
        
        if st.button("Process Sample Tickets"):
            try:
                sample_data = create_sample_data()
                results_tab, analysis_tab = st.tabs(["Results", "Analysis"])
                
                with st.spinner("Processing tickets..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results_df = loop.run_until_complete(
                            st.session_state.processor.process_all_tickets(sample_data)
                        )
                    finally:
                        loop.close()
                    
                    with results_tab:
                        st.dataframe(results_df, use_container_width=True)
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="classification_results.csv",
                            mime="text/csv"
                        )
                    
                    with analysis_tab:
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = pd.DataFrame(results_df['Sentiment'].value_counts())
                        st.bar_chart(sentiment_counts)
                        
                        st.subheader("Top Categories")
                        categories = [cat.strip() for cats in results_df['Categories'].str.split(',') for cat in cats]
                        category_counts = pd.DataFrame(pd.Series(categories).value_counts())
                        st.bar_chart(category_counts)
                        
                        st.metric("Average Confidence Score", f"{results_df['Confidence'].mean():.2f}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error in Streamlit app: {str(e)}", exc_info=True)

if __name__ == "__main__":
    create_streamlit_app()