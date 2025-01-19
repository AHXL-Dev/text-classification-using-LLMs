import asyncio
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import pandas as pd
import instructor
import logging
import streamlit as st



# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.1:8b"

LABELS = Literal['TIME_WAITING', 'POLICY', 'PROCESS', 
                'QUALITY_OF_RESOLUTION', 'SELF_HELP_RESOURCES', 'AGENT_BEHAVIOUR', 
                'AGENT_KNOWLEDGE', 'TECHNOLOGY', 'FOLLOW_UP']
CUSTOMER_SENTIMENT = Literal['ANGRY', 'FRUSTRATED', 'NEUTRAL', 'SATISFIED']

# Category definitions
CATEGORY_DEFINITIONS = {
    'TIME_WAITING': 'Feedback related to long response times, queue lengths, or delays in service. This includes complaints about waiting for responses or resolution timeframes.',
    
    'POLICY': 'Feedback related to company policies, rules, or standard procedures that affect service delivery. This includes cases where policies are unclear, seem unfair, or limit service options.',
    
    'PROCESS': 'Feedback related to how services are delivered or tasks are completed. This includes unclear procedures, complicated workflows, or inefficient steps in completing a task.',
    
    'QUALITY_OF_RESOLUTION': 'Feedback related to the effectiveness or completeness of the solution provided. This includes generic, incomplete, or ineffective solutions that don\'t actually resolve the customer\'s problem.',
    
    'SELF_HELP_RESOURCES': 'Feedback related to documentation, guides, manuals, or other self-service materials. This includes unclear instructions, missing information, or difficult-to-use resources.',
    
    'AGENT_BEHAVIOUR': 'Feedback related to how agents interact with customers. This includes lack of empathy, being rude, abusive or abrupt.',
    
    'AGENT_KNOWLEDGE': 'Feedback related to the agent\'s expertise or understanding. This includes incorrect information, inability to explain clearly, or lack of technical knowledge.',
    
    'TECHNOLOGY': 'Feedback related to systems, software, or technical infrastructure. This includes system errors, software bugs, or lack of ease of use with digital tools.',
    
    'FOLLOW_UP': 'Feedback related to ongoing communication or updates on existing requests. This includes lack of status updates, unclear next steps, or poor handling of ongoing cases.'
}


def create_sample_data():
        sample_data = [
            {
                "Row_ID": 1,
                "ProblemDescription": "I want to know how much my bonus is for next year.",
                "Resolution": "Hi, see this QRG on how to calculate.",
                "CustomerFeedback": "Didn’t call me, didn’t answer my question, sent me a generic guide."
            },
            {
                "Row_ID": 2,
                "ProblemDescription": "Why is my pay lower than usual this month?",
                "Resolution": "Your pay this month reflects updated tax rates.",
                "CustomerFeedback": "No clear explanation, I still don’t understand why my pay changed."
            },
            {
                "Row_ID": 3,
                "ProblemDescription": "When will I receive the backdated payment for my promotion?",
                "Resolution": "This usually takes 4-6 weeks.",
                "CustomerFeedback": "No specific timeline given for my case, felt like a generic answer."
            },
            {
                "Row_ID": 4,
                "ProblemDescription": "I need help accessing my payslips.",
                "Resolution": "You can download them directly from the portal.",
                "CustomerFeedback": "Payslips aren’t available on the portal, advice didn’t help."
            },
            {
                "Row_ID": 5,
                "ProblemDescription": "How do I reset my password for the invoice system?",
                "Resolution": "Please follow the steps in the attached guide.",
                "CustomerFeedback": "Guide is not clear, and I still can’t reset my password."
            },
            {
                "Row_ID": 6,
                "ProblemDescription": "Why hasn’t my overtime been paid yet?",
                "Resolution": "Overtime payments are processed at the end of each month.",
                "CustomerFeedback": "Didn’t explain why mine hasn’t been paid, very unhelpful."
            },
            {
                "Row_ID": 7,
                "ProblemDescription": "Can you tell me how to claim travel expenses?",
                "Resolution": "Refer to the travel policy document linked here.",
                "CustomerFeedback": "Policy document is too complicated, no real help provided."
            },
            {
                "Row_ID": 8,
                "ProblemDescription": "I need help with raising a purchase order for my department.",
                "Resolution": "Here’s the user guide for raising purchase orders.",
                "CustomerFeedback": "Guide didn’t cover approval steps, incomplete information."
            },
            {
                "Row_ID": 9,
                "ProblemDescription": "Why hasn’t my new hire’s account been created?",
                "Resolution": "Account creation takes up to 5 business days.",
                "CustomerFeedback": "Agent didn’t check the specific status of my request."
            },
            {
                "Row_ID": 10,
                "ProblemDescription": "I need help with processing invoices for my team.",
                "Resolution": "Here’s a link to the process flow for invoice approvals.",
                "CustomerFeedback": "Process flow was unclear, needed more guidance."
            },
            {
                "Row_ID": 11,
                "ProblemDescription": "Can I get an update on my previous query about delayed pay?",
                "Resolution": "No update available yet, we’ll get back to you soon.",
                "CustomerFeedback": "No real resolution, just delayed responses."
            },
            {
                "Row_ID": 12,
                "ProblemDescription": "How do I submit an expense claim for a conference I attended?",
                "Resolution": "Refer to the expense claim submission guide.",
                "CustomerFeedback": "Guide was not helpful, couldn’t find the form I needed."
            },
            {
                "Row_ID": 13,
                "ProblemDescription": "Why was my claim for relocation expenses denied?",
                "Resolution": "Please review the relocation policy linked here.",
                "CustomerFeedback": "Didn’t explain why my specific claim was denied."
            },
            {
                "Row_ID": 14,
                "ProblemDescription": "Can you tell me why my shift allowance wasn’t included in my pay?",
                "Resolution": "Shift allowances are included based on the payroll cycle.",
                "CustomerFeedback": "No explanation of why my specific allowance was missing."
            },
            {
                "Row_ID": 15,
                "ProblemDescription": "How do I get a copy of my tax forms from last year?",
                "Resolution": "You can download tax forms from the portal.",
                "CustomerFeedback": "Forms weren’t available on the portal, bad advice."
            },
            {
                "Row_ID": 16,
                "ProblemDescription": "Why did my expense reimbursement only cover half of what I claimed?",
                "Resolution": "Reimbursement is subject to policy limits.",
                "CustomerFeedback": "Didn’t explain which limits applied or how they were calculated."
            },
            {
                "Row_ID": 17,
                "ProblemDescription": "How can I get help with the new payroll software?",
                "Resolution": "Attend a scheduled training session.",
                "CustomerFeedback": "No immediate help provided, had to wait for a scheduled session."
            },
            {
                "Row_ID": 18,
                "ProblemDescription": "Why was my annual leave request denied?",
                "Resolution": "Leave requests are subject to approval based on workload.",
                "CustomerFeedback": "Agent didn’t check my specific request, just gave a generic answer."
            },
            {
                "Row_ID": 19,
                "ProblemDescription": "I need assistance with submitting invoices for approval.",
                "Resolution": "Check the attached user guide on invoice submission.",
                "CustomerFeedback": "Guide wasn’t helpful, still unclear about the process."
            },
            {
                "Row_ID": 20,
                "ProblemDescription": "Why was my pension contribution higher this month?",
                "Resolution": "Pension contributions vary based on salary.",
                "CustomerFeedback": "Didn’t explain why mine changed or how it was calculated."
            },
            {
                "Row_ID": 21,
                "ProblemDescription": "How do I update my bank account for salary payments?",
                "Resolution": "Please submit a request via the HR portal.",
                "CustomerFeedback": "No instructions on how to do it, wasted my time."
            },
            {
                "Row_ID": 22,
                "ProblemDescription": "I need help understanding my payslip deductions.",
                "Resolution": "Check the payslip guide attached.",
                "CustomerFeedback": "Didn’t explain the specific deductions I asked about."
            },
            {
                "Row_ID": 23,
                "ProblemDescription": "Why is the processing time for purchase orders so long?",
                "Resolution": "Processing times depend on approver availability.",
                "CustomerFeedback": "Didn’t offer any solution to speed up the process."
            },
            {
                "Row_ID": 24,
                "ProblemDescription": "Can I get assistance with the online leave application system?",
                "Resolution": "Follow the steps in the attached manual.",
                "CustomerFeedback": "Manual was confusing, still unable to apply for leave."
            },
            {
                "Row_ID": 25,
                "ProblemDescription": "How do I register for the new benefits program?",
                "Resolution": "Check the benefits registration page on the portal.",
                "CustomerFeedback": "Page was not easy to find, took too long to navigate."
            },
            {
                "Row_ID": 26,
                "ProblemDescription": "Can I change my payslip delivery method to email?",
                "Resolution": "Payslips are only available on the portal.",
                "CustomerFeedback": "Didn’t offer any alternatives, poor customer service."
            },
            {
                "Row_ID": 27,
                "ProblemDescription": "Why does it take so long to get support on payroll queries?",
                "Resolution": "Support is provided based on queue order.",
                "CustomerFeedback": "Queue is too long, unacceptable wait time."
            },
            {
                "Row_ID": 28,
                "ProblemDescription": "How can I escalate my query about missing pay?",
                "Resolution": "Escalation is only possible after 5 business days.",
                "CustomerFeedback": "Didn’t provide immediate escalation, felt ignored."
            },
            {
                "Row_ID": 29,
                "ProblemDescription": "Why do I need to resubmit my claim for dependent care reimbursement?",
                "Resolution": "Your previous submission was incomplete.",
                "CustomerFeedback": "Didn’t explain what was missing, wasted my time."
            },
            {
                "Row_ID": 30,
                "ProblemDescription": "I need clarification on the policy for weekend overtime.",
                "Resolution": "Refer to the attached policy document.",
                "CustomerFeedback": "Policy document was too vague, didn’t clarify anything."
            }
        ]
        
        return sample_data
    
class TicketClassification(BaseModel):
    '''Classification model for support tickets'''
    
    class Category(BaseModel):
        category: List[LABELS] = Field(..., description='''
            Analyze the ProblemDescription, Resolution and CustomerFeedback and select one or more labels that apply to categorise the feedback.
            Consider these category definitions carefully:
            
            'TIME_WAITING': 'Feedback related to long response times, queue lengths, or delays in service. This includes complaints about waiting for responses or resolution timeframes.',
            'POLICY': 'Feedback related to company policies, rules, or standard procedures that affect service delivery. This includes cases where policies are unclear, seem unfair, or limit service options.',
            'PROCESS': 'Feedback related to how services are delivered or tasks are completed. This includes unclear procedures, complicated workflows, or inefficient steps in completing a task.',
            'QUALITY_OF_RESOLUTION': 'Feedback related to the effectiveness or completeness of the solution provided. This includes generic, incomplete, or ineffective solutions that don\'t actually resolve the customer\'s problem.',
            'SELF_HELP_RESOURCES': 'Feedback related to documentation, guides, manuals, or other self-service materials. This includes unclear instructions, missing information, or difficult-to-use resources.',
            'AGENT_BEHAVIOUR': 'Feedback related to how agents interact with customers. This includes lack of empathy, being rude, abusive or abrupt.',
            'AGENT_KNOWLEDGE': 'Feedback related to the agent\'s expertise or understanding. This includes incorrect information, inability to explain clearly, or lack of technical knowledge.',
            'TECHNOLOGY': 'Feedback related to systems, software, or technical infrastructure. This includes system errors, software bugs, or lack of ease of use with digital tools.',
            'FOLLOW_UP': 'Feedback related to ongoing communication or updates on existing requests. This includes lack of status updates, unclear next steps, or poor handling of ongoing cases.'
                
            Choose categories that best match the customer's feedback about their experience.
        ''')
        justification: str = Field(..., description='Explain why you selected these categories, referencing specific aspects of the ticket.')
        sentiment: CUSTOMER_SENTIMENT = Field(..., description='''
            Determine the customer's emotional state based on their feedback:
            ANGRY: Shows strong negative emotions, frustration, or dissatisfaction
            FRUSTRATED: Shows mild to moderate negative emotions or disappointment
            NEUTRAL: Shows neither strong positive nor negative emotions
            SATISFIED: Shows positive emotions or satisfaction
        ''')
        confidence: float = Field(ge=0, le=1, description='How confident are you in this classification (0-1)')
        Row_ID: int

    @classmethod
    def get_category_definition(cls, category: LABELS) -> str:
        """Get the detailed definition for a category"""
        return CATEGORY_DEFINITIONS.get(category, "No definition available")

class SimplifiedBatchProcessor:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.client = instructor.patch(
            AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
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
                    f"Analyze this support ticket with attention to the specific roles and context of each field:\n\n"
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
                    f"1. Focus primarily on the CustomerFeedback as it is the primary driver of the classification\n"
                    f"2. Compare the initial request (ProblemDescription) with the resolution provided\n"
                    f"3. Evaluate if the agent's resolution (Resolution) appropriately addressed the customer's needs\n"
                    f"4. Consider the gap between what was requested and what was delivered\n"
                    f"5. Assess the customer's emotional response in their feedback\n\n"
                    f"6. Provide classification that reflects one or more of the following categories that best describe the customer's feedback. If multiple categories apply, please list them all (separate categories with commas):\n\n"
                    f"Provide classifications that reflects:\n"
                    f"- How well the customer's needs were met\n"
                    f"- Any gaps in service delivery\n"
                    f"- The customer's emotional response\n"
                    f"- The effectiveness of the resolution"
                )
                
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    response_model=TicketClassification.Category,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
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
                'Categories': ', '.join(result.category),
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
def format_category_name(category: str) -> str:
    """Convert a category name to the correct format (TEXT_LIKE_THIS)"""
    
    formatted = ''.join(c for c in category if c.isalnum() or c.isspace())
    return formatted.upper().replace(' ', '_')

def create_streamlit_app():
    st.set_page_config(layout="wide")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "Classification Demo", "Custom Category Search"])
    
    
    if 'processor' not in st.session_state:
        st.session_state.processor = SimplifiedBatchProcessor(batch_size=5)
    
    if page == "Introduction":
        st.title("Text Classification using Large Language Models – Demonstration")
        st.markdown("*Created by Jonko*")
        
        st.header("Key Points:")
        st.markdown("""
        * This app is a demonstration of how Generative AI can be used to classify customer feedback based on mock customer support tickets. The AI model analyzes various parts of the ticket (Problem Description, Resolution, and Customer Feedback) to categorize the feedback and determine customer sentiment.**
        * The app runs on a local LLM (Large Language Model), which is downloaded and hosted locally, meaning **no data is sent to an external model provider**. The model being used is Meta's Llama 3.18B, a lightweight model with 8 billion parameters.**
        * The demonstration is hosted on Streamlit Hub.
        * This is purely for demonstration, and there are further ways to enhance the program or use a stronger model.
        """)
        
        st.header("Classification Page")
        st.markdown("""
        This page showcases how the AI classifies customer support tickets using the **mock dataset**. The system automatically categorizes each ticket based on the customer's feedback, problem description, and agent's resolution.

        The categories used in this classification are designed to reflect different aspects of customer service and feedback.  
        This is a **fake dataset** with **X number of rows** of mock tickets.

        ### Categories Used in Classification:
        - **TIME_WAITING**: Feedback about long wait times or delays.
        - **POLICY**: Issues with company policies or procedures.
        - **PROCESS**: Feedback on how services or tasks are delivered.
        - **QUALITY_OF_RESOLUTION**: Effectiveness of the resolution provided.
        - **SELF_HELP_RESOURCES**: Issues with self-help materials like guides or manuals.
        - **AGENT_BEHAVIOUR**: Feedback on the agent’s demeanor or professionalism.
        - **AGENT_KNOWLEDGE**: Issues with the agent’s knowledge or expertise.
        - **TECHNOLOGY**: Feedback on the systems, tools, or technology used.
        - **FOLLOW_UP**: Feedback about ongoing communication or lack of updates.
        """)
        
        st.header("Custom Category Search Page")
        st.markdown("""
        This page allows you to utilise the GenAI model to search for specific categories of feedback using the **mock dataset**. This can be useful for finding feedback that aligns with your specific requirements or goals.
        """)
    
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
                
                async def analyze_ticket(ticket, search_criteria):
                    prompt = f"""
                    Analyze this support ticket for feedback {ticket['CustomerFeedback']} matching the following criteria,:
                    SEARCH CRITERIA: {search_criteria}
                    EXAMPLE PHRASES: {example_phrases if example_phrases else 'None provided'}
                    
                    Focus on the {ticket['CustomerFeedback']} but use the following fields in the analysis to find feedback matching the {search_criteria} 
                    TICKET:
                    Resolution: {ticket['Resolution']}
                    Feedback: {ticket['CustomerFeedback']}
                    
                    ** RETURN 'MATCH' IF THERE IS A MATCH OTHERWISE RETURN 'NO MATCH'. Provide Jusitifcation on your selection **
                    """
                    
                    response = await st.session_state.processor.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    
                    return {
                        'Row_ID': ticket['Row_ID'],
                        'Problem': ticket['ProblemDescription'],
                        'Resolution': ticket['Resolution'],
                        'Feedback': ticket['CustomerFeedback'],
                        'Analysis': response.choices[0].message.content
                    }

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tasks = [analyze_ticket(ticket, issue_description) for ticket in sample_data]
                    results = loop.run_until_complete(asyncio.gather(*tasks))
                finally:
                    loop.close()
                
                df = pd.DataFrame(results)
                
                results_tab, analysis_tab = st.tabs(["Results", "Analysis"])
                
                with results_tab:
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="search_results.csv",
                        mime="text/csv"
                    )
                
                with analysis_tab:
                    st.markdown("### Key Findings")
                    st.info("Analysis overview will be shown here")
            
    else:
        st.title("Support Ticket Classification System")
        
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