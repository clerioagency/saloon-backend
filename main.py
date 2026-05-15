from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os

from pymongo import MongoClient

app = FastAPI()

SERVER_SESSION_ID = str(uuid.uuid4())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
OWNER_WHATSAPP_NUMBER = os.getenv("OWNER_WHATSAPP_NUMBER")

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI  = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI)
db           = mongo_client["studio5"]
leads_col    = db["leads"]

# ── Lead helpers ──────────────────────────────────────────────────────────────

def load_leads() -> list:
    return list(leads_col.find({}, {"_id": 0}))

def store_lead(name: str, phone: str, requirement: str, slot: str = "", booking_date: str = "") -> dict:
    now = datetime.now()
    if not booking_date:
        booking_date = now.strftime("%Y-%m-%d")
    try:
        bd = datetime.strptime(booking_date, "%Y-%m-%d")
        display_booking = bd.strftime("%d %b %Y")
    except:
        display_booking = booking_date
    lead = {
        "id":              str(uuid.uuid4()),
        "name":            name,
        "phone":           phone,
        "requirement":     requirement,
        "slot":            slot,
        "booking_date":    booking_date,          # date of appointment
        "display_booking": display_booking,       # e.g. "16 May 2026"
        "status":          "New",
        "timestamp":       now.isoformat(),
        "date":            now.strftime("%Y-%m-%d"),  # date lead was created
        "time":            now.strftime("%I:%M %p"),
        "display_date":    now.strftime("%d %b %Y"),
    }
    leads_col.insert_one({**lead})
    return lead

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a warm, enthusiastic booking assistant for Studio 5, a premium salon. Your goal: convert visitors into booked clients.

BEHAVIOR:
- Greet warmly, answer questions, suggest 2-3 relevant services based on the chat
- Keep replies SHORT (1-3 lines), use emojis naturally 💇‍♀️✨
- Never repeat a question already answered — use full conversation history to personalise

BOOKING FORM RULE (CRITICAL):
- The website has a built-in booking form that appears automatically in the chat
- When a user shows interest in booking, respond ONLY with something like:
  "Awesome! Just fill in the quick form below and our team will call you to confirm 🎉"
- NEVER ask for name or phone number in chat — the form collects that
- NEVER say "can you share your phone number" or "what's your name"
- The form also shows available time slots for today — booked slots are marked and disabled

LEAD CAPTURE:
- The moment user shows ANY interest in a service or booking, say:
  "Awesome! Just fill in the quick form below and our team will call you to confirm 🎉"

SERVICES & PRICING:
Haircut Women Rs.399 · Men Rs.199 · Hair Colour Rs.1,499+ · Highlights/Balayage Rs.2,499+
Keratin Rs.3,999+ · Hair Spa Rs.799 · Facial Rs.999 · Threading Rs.49
Waxing Arms Rs.299 · Legs Rs.399 · Manicure Rs.499 · Pedicure Rs.599 · Bridal Rs.9,999+

TIMINGS: 10 AM – 8 PM daily · LOCATION: City Centre Mall, Ground Floor
"""

# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class StatusUpdate(BaseModel):
    status: str

class LoginData(BaseModel):
    username: str
    password: str

# ── Utilities ─────────────────────────────────────────────────────────────────

def detect_phone(text: str):
    match = re.search(r'\b\d{10}\b', text)
    return match.group(0) if match else None

def detect_slot(text: str) -> str:
    """Extract time slot like '01:00 PM' or '1pm' from message."""
    # Match format from form: HH:MM AM/PM
    match = re.search(r'\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Match loose format: 1pm, 2:30pm etc
    match = re.search(r'\bat\s+(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""

def extract_name(history: list, groq_api_key: str) -> str:
    if not history:
        return "Customer"
    transcript = "\n".join(f"User: {m.content}" for m in history if m.role == "user")
    prompt = (
        "From the following chat messages, extract the customer's first name if they mentioned it anywhere.\n"
        "Reply with ONLY the name (1-3 words, no punctuation). "
        "If no name is mentioned, reply with exactly: Customer\n\n"
        f"{transcript}"
    )
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0}
        )
        raw = res.json()["choices"][0]["message"]["content"].strip().title()
        if raw and len(raw) <= 40 and raw.lower() not in ("customer", "unknown", "none", "n/a"):
            return raw
    except Exception as e:
        print("Name extraction error:", e)

    patterns = [
        r"my name is\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"i(?:'?m| am)\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"this is\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"call me\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"name(?:'?s)?\s*[:\-]?\s*([A-Za-z]+(?:\s[A-Za-z]+)?)",
    ]
    INVALID = {"a", "the", "going", "planning", "interested", "here", "not", "just", "ok", "yes"}
    for m in history:
        if m.role != "user":
            continue
        for pattern in patterns:
            match = re.search(pattern, m.content, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().title()
                if candidate.lower() not in INVALID:
                    return candidate
    return "Customer"

def extract_requirement(history: list, current_msg: str, groq_api_key: str) -> str:
    all_msgs = list(history)

    class _Msg:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    all_msgs.append(_Msg("user", current_msg))
    transcript = "\n".join(f"{'Customer' if m.role == 'user' else 'Bot'}: {m.content}" for m in all_msgs)
    prompt = (
        "Read this salon chat and extract what service the customer wants (2-6 words max).\n"
        "Examples: 'Haircut', 'Bridal Package', 'Hair Colour', 'Facial'. No names or phone numbers.\n"
        "If nothing specific, reply: General enquiry\n\n"
        f"{transcript}"
    )
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0}
        )
        raw = res.json()["choices"][0]["message"]["content"].strip()
        if raw and len(raw) <= 60:
            return raw
    except Exception as e:
        print("Requirement extraction error:", e)

    SERVICE_KEYWORDS = [
        "haircut", "hair cut", "colour", "color", "highlights", "balayage",
        "keratin", "spa", "facial", "threading", "waxing", "bridal",
        "manicure", "pedicure", "appointment", "booking", "treatment"
    ]
    for m in all_msgs:
        if m.role != "user":
            continue
        txt = m.content.strip().lower()
        if re.fullmatch(r'[\d\s\+\-]+', txt):
            continue
        for kw in SERVICE_KEYWORDS:
            if kw in txt:
                return kw.title()
    return "General enquiry"

def send_whatsapp(name: str, phone: str, requirement: str, slot: str = "", booking_date: str = "") -> dict:
    missing = [k for k, v in {
        "TWILIO_ACCOUNT_SID":     TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN":      TWILIO_AUTH_TOKEN,
        "TWILIO_WHATSAPP_NUMBER": TWILIO_WHATSAPP_NUMBER,
        "OWNER_WHATSAPP_NUMBER":  OWNER_WHATSAPP_NUMBER,
    }.items() if not v]

    if missing:
        print(f"❌ Missing Twilio env vars: {missing}")
        return {"status": None, "error": f"Missing env vars: {missing}"}

    from_num = TWILIO_WHATSAPP_NUMBER if TWILIO_WHATSAPP_NUMBER.startswith("whatsapp:") else f"whatsapp:{TWILIO_WHATSAPP_NUMBER}"
    to_num   = OWNER_WHATSAPP_NUMBER  if OWNER_WHATSAPP_NUMBER.startswith("whatsapp:")  else f"whatsapp:{OWNER_WHATSAPP_NUMBER}"

    slot_line = f"\n⏰ Slot: {slot}" if slot else ""
    body = f"New Booking Lead 💇‍♀️\n\nName: {name}\nPhone: {phone}\nService: {requirement}{slot_line}\n📅 Date: {booking_date}\n\n— Studio 5"

    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        res = requests.post(
            url,
            data={"From": from_num, "To": to_num, "Body": body},
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        result = res.json()
        print("── Twilio ────────────────────────────")
        print(f"  HTTP : {res.status_code} | SID: {result.get('sid','—')} | Error: {result.get('message','none')}")
        return {"status": res.status_code, "sid": result.get("sid"), "error": result.get("message")}
    except Exception as e:
        print(f"❌ WhatsApp exception: {e}")
        return {"status": None, "error": str(e)}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/login")
def login(data: LoginData):
    if data.username == "admin" and data.password == "1234":
        return {"success": True}
    return {"success": False}

@app.get("/ping")
def ping():
    return {"session_id": SERVER_SESSION_ID}

@app.post("/chat")
def chat(req: ChatRequest):
    user_msg = req.message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (req.history or []):
        role = m.role if m.role in ("user", "assistant") else "user"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_msg})

    ai_reply = "Something went wrong. Try again!"
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 200, "temperature": 0.75}
        )
        data = response.json()
        if "choices" in data:
            ai_reply = data["choices"][0]["message"]["content"].strip()
        else:
            print("Groq error:", data)
    except Exception as e:
        print("Groq exception:", e)
        ai_reply = "AI error. Try again later."

    phone = detect_phone(user_msg)
    if phone:
        print("=== LEAD DETECTED ===")

        # Fast parse from synthetic message: "My name is X and my number is Y. I want to book: Z at T."
        # No LLM calls needed — form always sends this exact format
        name_match = re.search(r'My name is ([A-Za-z ]+?) and my number', user_msg, re.IGNORECASE)
        req_match  = re.search(r'I want to book:\s*([^.]+?)(?:\s+at\s+|\.)', user_msg, re.IGNORECASE)
        name        = name_match.group(1).strip().title() if name_match else "Customer"
        requirement = req_match.group(1).strip() if req_match else "General Enquiry"
        slot        = detect_slot(user_msg)
        # Extract booking date from synthetic message: "on YYYY-MM-DD"
        date_match    = re.search(r'on (\d{4}-\d{2}-\d{2})', user_msg)
        booking_date  = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')

        print(f"  Name: {name} | Req: {requirement} | Slot: {slot} | Date: {booking_date} | Phone: {phone}")

        store_lead(name, phone, requirement, slot, booking_date)
        wa_result = send_whatsapp(name, phone, requirement, slot, booking_date)
        print("WhatsApp result:", wa_result)

        greeting  = f", {name}" if name != "Customer" else ""
        slot_line = f" at {slot}" if slot else ""
        ai_reply  = f"Perfect{greeting}! ✨ Your {requirement} appointment{slot_line} is confirmed. Our Studio 5 team will call you shortly! 💇‍♀️"

    return {"reply": ai_reply}

@app.get("/test-whatsapp")
def test_whatsapp():
    result = send_whatsapp("Test Client", "9999999999", "Test ping from Studio 5", "11:00 AM")
    return {"twilio_result": result, "from": TWILIO_WHATSAPP_NUMBER, "to": OWNER_WHATSAPP_NUMBER}

# ── Leads API ─────────────────────────────────────────────────────────────────

@app.get("/leads")
def get_leads():
    leads = load_leads()
    grouped = {}
    for lead in sorted(leads, key=lambda x: x["timestamp"], reverse=True):
        d = lead["display_date"]
        if d not in grouped:
            grouped[d] = []
        grouped[d].append(lead)
    return {"leads": leads, "grouped": grouped, "total": len(leads)}

@app.patch("/leads/{lead_id}/status")
def update_status(lead_id: str, body: StatusUpdate):
    result = leads_col.update_one({"id": lead_id}, {"$set": {"status": body.status}})
    if result.matched_count:
        lead = leads_col.find_one({"id": lead_id}, {"_id": 0})
        return {"ok": True, "lead": lead}
    return {"ok": False, "error": "Lead not found"}

@app.delete("/leads/{lead_id}")
def delete_lead(lead_id: str):
    leads_col.delete_one({"id": lead_id})
    return {"ok": True}

# ── Serve HTML ────────────────────────────────────────────────────────────────

# FRONTEND_OWNER_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend-owner")
# INDEX_HTML_PATH     = os.path.join(FRONTEND_OWNER_DIR, "index.html")
# DASHBOARD_HTML_PATH = os.path.join(FRONTEND_OWNER_DIR, "dashboard.html")
INDEX_HTML_PATH     = os.getenv("INDEX_HTML_PATH",     os.path.join(os.path.dirname(__file__), "index.html"))
DASHBOARD_HTML_PATH = os.getenv("DASHBOARD_HTML_PATH", os.path.join(os.path.dirname(__file__), "dashboard.html"))

@app.get("/dashboard")
def serve_dashboard():
    if not os.path.exists(DASHBOARD_HTML_PATH):
        return {"error": f"dashboard.html not found at: {DASHBOARD_HTML_PATH}"}
    return FileResponse(DASHBOARD_HTML_PATH)

@app.get("/")
def serve_index():
    if not os.path.exists(INDEX_HTML_PATH):
        return {"error": f"index.html not found at: {INDEX_HTML_PATH}"}
    return FileResponse(INDEX_HTML_PATH)