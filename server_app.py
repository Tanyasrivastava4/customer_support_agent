from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json

app = FastAPI(title="Salad AI Server with LLM")

# -----------------------
# Load KB from JSON
# -----------------------
with open("kb.json", "r") as f:
    KB = json.load(f)

# -----------------------
# Pydantic Model
# -----------------------
class TextInput(BaseModel):
    text: str

# -----------------------
# Load Hugging Face LLM for text generation
# -----------------------
MODEL_NAME = "google/flan-t5-small"  # Lightweight CPU-friendly
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# -----------------------
# Email Classification using LLM
# -----------------------
@app.post("/classify")
def classify_email(data: TextInput):
    prompt = f"Classify this email into one of the categories: billing, technical, account, refund, shipping, general.\nEmail: {data.text}"
    result = generator(prompt, max_length=20)[0]['generated_text']
    category = result.strip().lower()
    if category not in KB.keys():
        category = "general"
    return {"category": category}

# -----------------------
# RAG: Retrieve Answer from KB using LLM
# -----------------------
@app.post("/rag")
def get_rag_answer(data: TextInput):
    # First classify email
    category = classify_email(data).get("category")
    # Generate better response using LLM + KB
    kb_answer = KB.get(category, "Sorry, we do not have an answer for this query yet.")
    prompt = f"Generate a professional customer support response based on the following answer:\n{kb_answer}\nEmail: {data.text}"
    response = generator(prompt, max_length=200)[0]['generated_text']
    return {"answer": response}










# this is improved code pasted below for server and local code
# we can use this if we donot get the output from above codes or codes written in local vscode


"""
#this is server_app.py
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from jira import JIRA
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import json
import os

# ----------------------------
# Load environment variables
# ----------------------------
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Customer Support Server with LLM")

# ----------------------------
# Load KB
# ----------------------------
with open("kb.json", "r") as f:
    KB = json.load(f)

# ----------------------------
# Pydantic Models
# ----------------------------
class TextInput(BaseModel):
    text: str

class Ticket(BaseModel):
    user_id: str
    issue_type: str
    urgency: str

    # Validation
    @staticmethod
    def validate_issue_type(v):
        allowed = ["billing", "technical", "account", "refund", "feedback"]
        if v.lower() not in allowed:
            raise ValueError(f"Issue type must be one of {allowed}")
        return v

    @staticmethod
    def validate_urgency(v):
        if v.lower() not in ["low", "medium", "high"]:
            raise ValueError("Invalid urgency")
        return v

# ----------------------------
# Load LLM Model
# ----------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ----------------------------
# Initialize Jira client
# ----------------------------
jira_client = JIRA(server=JIRA_BASE_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))

# ----------------------------
# Helper functions
# ----------------------------

# 1️⃣ Classify email using LLM
def classify_email_llm(text):
    prompt = f"Classify this email into one of the categories: billing, technical, account, refund, shipping, general.\nEmail: {text}"
    result = generator(prompt, max_length=20)[0]['generated_text']
    category = result.strip().lower()
    if category not in KB.keys():
        category = "general"
    return category

# 2️⃣ RAG response from KB + LLM
def get_rag_response(email_text, category):
    kb_answer = KB.get(category, "Our support team will contact you shortly.")
    prompt = f"Generate a professional customer support response based on the following answer:\n{kb_answer}\nEmail: {email_text}"
    response = generator(prompt, max_length=200)[0]['generated_text']
    return response

# 3️⃣ Send email using SendGrid
def send_email(to_email, subject, body):
    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )
    sg = SendGridAPIClient(SENDGRID_API_KEY)
    sg.send(message)

# 4️⃣ Create Jira ticket
def create_jira_ticket(ticket, summary, description, issue_type_name="Support"):
    issue_dict = {
        "project": {"key": JIRA_PROJECT_KEY},
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type_name},
        "priority": {"name": ticket.urgency.capitalize()},
    }
    issue = jira_client.create_issue(fields=issue_dict)
    issue = jira_client.issue(issue.key)
    return issue.key, issue.fields.status.name, issue.fields.issuetype.name

# 5️⃣ Handle ticket workflow
def handle_ticket(ticket_data, email_text):
    # Validate Pydantic model
    ticket = Ticket(**ticket_data)

    # Step 1: classify
    category = classify_email_llm(email_text)

    # Step 2: generate response
    response = get_rag_response(email_text, category)

    jira_key = jira_status = jira_issue_type = None

    # Step 3: Escalation & Jira if high urgency
    if ticket.urgency.lower() == "high":
        response += "\n\nThis issue has been escalated to high priority."
        # Send escalation email
        send_email(
            to_email=ESCALATION_EMAIL,
            subject=f"Escalation: High Priority {category} issue",
            body=f"High priority ticket from {ticket.user_id}.\n\nDetails:\n{response}"
        )
        # Create Jira ticket
        summary = f"{category.capitalize()} issue from {ticket.user_id}"
        jira_key, jira_status, jira_issue_type = create_jira_ticket(ticket, summary, response)

    # Step 4: send response email to user
    send_email(
        to_email=ticket.user_id,
        subject=f"Support Response: {category}",
        body=response
    )

    return {
        "category": category,
        "response": response,
        "jira_key": jira_key,
        "jira_status": jira_status,
        "jira_issue_type": jira_issue_type
    }

# ----------------------------
# API Endpoints
# ----------------------------

@app.post("/process_ticket")
def process_ticket(data: dict):
    try:
        ticket_data = {
            "user_id": data.get("user_id"),
            "issue_type": data.get("issue_type", "general"),
            "urgency": data.get("urgency", "low")
        }
        email_text = data.get("email_body", "")
        result = handle_ticket(ticket_data, email_text)
        return result
    except ValidationError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": str(e)}

"""

"""
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import requests  # to call server API

app = FastAPI()
templates = Jinja2Templates(directory="local_vscode/templates")

# Keep history locally for demo
ticket_history = []

# ----------------------------
# Home page
# ----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "ticket_history": ticket_history, 
            "error": None, 
            "success": None
        }
    )

# ----------------------------
# Process email form
# ----------------------------
@app.post("/process_email/")
def process_email(
    request: Request,
    user_id: str = Form(...),
    email_body: str = Form(...),
    urgency: str = Form(...)
):
    try:
        # 1️⃣ Prepare payload to send to server
        payload = {
            "user_id": user_id,
            "email_body": email_body,
            "urgency": urgency
        }

        # 2️⃣ Call server API
        SERVER_URL = "http://YOUR_SERVER_URL/process_ticket"  # replace with Salad server URL
        response = requests.post(SERVER_URL, json=payload)
        result = response.json()

        if "error" in result:
            # Server returned an error
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "ticket_history": ticket_history,
                    "error": result["error"],
                    "success": None
                }
            )

        # 3️⃣ Append ticket history for display
        ticket_history.append({
            "user_id": user_id,
            "issue_type": result.get("category"),
            "jira_issue_type": result.get("jira_issue_type") or "N/A",
            "urgency": urgency,
            "response": result.get("response"),
            "jira_key": result.get("jira_key") or "N/A",
            "jira_status": result.get("jira_status") or "N/A"
        })

        # 4️⃣ Show success message
        success_message = f"Ticket processed successfully! Issue type: {result.get('category')}"
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ticket_history": ticket_history,
                "error": None,
                "success": success_message
            }
        )

    except Exception as e:
        # Catch-all error
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "ticket_history": ticket_history,
                "error": f"Internal error: {str(e)}",
                "success": None
            }
        )
"""
