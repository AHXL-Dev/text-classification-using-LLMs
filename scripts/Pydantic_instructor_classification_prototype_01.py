import streamlit as st
import asyncio
from typing import List, Type, Literal
from pydantic import BaseModel, Field, computed_field, field_validator
from openai import OpenAI,AsyncOpenAI
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import instructor
import os
import chromadb


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "../data/BA_AirlineReviews.csv"
MODEL_NAME = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


async def get_async_client(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama") -> OpenAI:
    """
    Return an OpenAI client configured with instructor to return themes and explanations,
    in accordance with the Pydantic models. Async is used to allow batch processing (processing simulatenously)
    """
    return instructor.patch(
        AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        ),
        mode=instructor.Mode.JSON,
    )
    
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


LABELS = Literal['CALL_WAIT_TIMES','CUSTOMER_EXPECTATIONS','POLICY','PROCESS','QUALITY_OF_RESOLUTION','SELF_HELP_RESOURCES','AGENT_BEHAVIOUR','AGENT_KNOWLEDGE','TECHNOLOGY','FOLLOW_UP']
CUSTOMER_SENTIMENT = Literal['ANGRY','FRUSTRATED','NEUTRAL','SATISFIED']
class TicketClassification(BaseModel):
    '''
    A few-shot example of multi-label classification:
    Examples:

    "I want to know how much my bonus is for next year. Resolution: Hi, see this QRG on how to calculate. CustomerFeedback: Didn’t call me, didn’t answer my question, sent me a generic guide.": POLICY, SELF_HELP_RESOURCES
    "Why is my pay lower than usual this month? Resolution: Your pay this month reflects updated tax rates. CustomerFeedback: No clear explanation, I still don’t understand why my pay changed.": QUALITY_OF_RESOLUTION, AGENT_KNOWLEDGE
    "When will I receive the backdated payment for my promotion? Resolution: This usually takes 4-6 weeks. CustomerFeedback: No specific timeline given for my case, felt like a generic answer.": FOLLOW_UP, PROCESS
    "I need help accessing my payslips. Resolution: You can download them directly from the portal. CustomerFeedback: Payslips aren’t available on the portal, advice didn’t help.": SELF_HELP_RESOURCES, QUALITY_OF_RESOLUTION
    "How do I reset my password for the invoice system? Resolution: Please follow the steps in the attached guide. CustomerFeedback: Guide is not clear, and I still can’t reset my password.": TECHNOLOGY, PROCESS
    "Why hasn’t my overtime been paid yet? Resolution: Overtime payments are processed at the end of each month. CustomerFeedback: Didn’t explain why mine hasn’t been paid, very unhelpful.": AGENT_KNOWLEDGE, FOLLOW_UP
    "Can you tell me how to claim travel expenses? Resolution: Refer to the travel policy document linked here. CustomerFeedback: Policy document is too complicated, no real help provided.": POLICY, SELF_HELP_RESOURCES
    "I need help with raising a purchase order for my department. Resolution: Here’s the user guide for raising purchase orders. CustomerFeedback: Guide didn’t cover approval steps, incomplete information.": PROCESS, SELF_HELP_RESOURCES
    "Why hasn’t my new hire’s account been created? Resolution: Account creation takes up to 5 business days. CustomerFeedback: Agent didn’t check the specific status of my request.": AGENT_BEHAVIOUR, FOLLOW_UP
    "How can I escalate my query about missing pay? Resolution: Escalation is only possible after 5 business days. CustomerFeedback: Didn’t provide immediate escalation, felt ignored.": FOLLOW_UP, PROCESS'
    '''
    
    
    class Category(BaseModel):
        category: List[LABELS] = Field(...,description = 'Analyse the ProblemDescription , Resolution and then analyse the CustomerFeedback and only select labels that apply to categorise the feedback')
        justification: str
        sentiment: CUSTOMER_SENTIMENT
        confidence: float = Field(ge=0, le=1)
        Row_ID: int


class BatchProcessor:
    """
    Handles batch processing of tickets with retry logic and error handling
    """
    def __init__(self, batch_size: int = 5, max_retries: int = 3):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def create_prompt(ticket: dict) -> str:
        """
        Creates a formatted prompt for the LLM
        """
        return (f"Please classify this support ticket's CustomerFeedback\n"
                "*ALWAYS* Analyse the ProblemDescription,Resolution and the CustomerFeedback together:\n"
                f"ProblemDescription: {ticket['ProblemDescription']}\n"
                f"Resolution: {ticket['Resolution']}\n"
                f"CustomerFeedback: {ticket['CustomerFeedback']}")

    async def process_single_ticket(
        self,
        ticket: dict,
        client: AsyncOpenAI
    ) -> TicketClassification.Category:
        """
        Process a single ticket with retry logic
        """
        for attempt in range(self.max_retries):
            try:
                prompt = self.create_prompt(ticket)
                response = await client.chat.completions.create(
                    model="llama3.1:8b",  # Or your preferred model
                    response_model=TicketClassification.Category,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to process ticket {ticket['Row_ID']}: {str(e)}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def process_batch(
        self,
        batch: List[dict], # dictionary of tickets
        client: AsyncOpenAI
    ) -> List[TicketClassification.Category]:
        """
        Process a batch of tickets concurrently. Must use *tasks to get each item from the list and pass into the asyncio.gather method.
        """
        tasks = [self.process_single_ticket(ticket, client) for ticket in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_all_tickets(
        self,
        tickets: List[dict]
    ) -> pd.DataFrame:
        """
        Process all tickets in all batches and return results as a DataFrame
        """
        # Create batches
        batches = [tickets[i:i + self.batch_size] 
                  for i in range(0, len(tickets), self.batch_size)]
        
        # Initialize client
        client = await get_async_client()
        
        # Process all batches
        all_results = []
        for batch in batches:
            try:
                batch_results = await self.process_batch(batch, client)
                all_results.extend([r for r in batch_results if not isinstance(r, Exception)])
            except Exception as e:
                self.logger.error(f"Batch processing failed: {str(e)}")
                continue

        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'Row_ID': result.Row_ID,
                'Categories': result.category,
                'Sentiment': result.sentiment,
                'Confidence': result.confidence,
                'Justification': result.justification
            }
            for result in all_results
        ])

        return results_df

# Streamlit app components
def create_streamlit_app():
    st.title("Support Ticket Classification System")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = BatchProcessor(batch_size=5, max_retries=3)

    if st.button("Process Sample Tickets"):
        sample_data = create_sample_data()
        
        with st.spinner("Processing tickets..."):
            # Run async processing
            results_df = asyncio.run(
                st.session_state.processor.process_all_tickets(sample_data)
            )
            
            # Display results
            st.write("Classification Results:")
            st.dataframe(results_df)
            
            # Show summary statistics
            st.write("Summary Statistics:")
            sentiment_counts = results_df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

if __name__ == "__main__":
    create_streamlit_app()

