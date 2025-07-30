# app.py backend updates

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime
from pathlib import Path
from main import generate_IPC_bot_response, generate_gpt_default, generate_diff_change_prob
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-study-xi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
CLASS_FILE = os.path.join(DATA_DIR, "classifications.json")
EVAL_FILE = os.path.join(DATA_DIR, "evaluations.json")
USER_FILE = os.path.join(DATA_DIR, "users.json")
LOG_DIR = os.path.join(DATA_DIR, "chatlogs")
SUMMARY_FILE = os.path.join(DATA_DIR, "evaluation_summary.json")


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class Message(BaseModel):
    user: str
    history: list
    bot: str
    llm_icm: Optional[list[int]] = None
    patient: str

class ClassificationEntry(BaseModel):
    sentence: str
    dominance: int
    friendliness: int
    classificator: str

class Evaluation(BaseModel):
    realism: int
    appropriateness: int
    consistency: int
    feedback: str
    log_filename: str
    ipc_d_guess: Optional[int] = None
    ipc_f_guess: Optional[int] = None

class EvaluationSummary(BaseModel):
    user_id: str
    ranking: str
    reason: str
    diversity: int

class UserInfo(BaseModel):
    age: str
    field: str
    id: str

@app.get("/sentences")
async def get_sentences():
    if not os.path.exists(CLASS_FILE):
        return []

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/register-user")
async def register_user(info: UserInfo):
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(USER_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append(info.dict())

    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "saved"}


@app.post("/chat")
def chat(msg: Message):
    bot = msg.bot

    if bot == "gpt_default":
        return {"response": generate_gpt_default(msg.user, msg.history)}
    
    elif bot == "icm_agent_0.5":
        response, new_llm_icm, patient = generate_IPC_bot_response(msg.user, msg.history, msg.llm_icm, msg.patient)
        return {"response": response, "llm_icm": new_llm_icm, "patient": patient}

    elif bot == "neutral_agent_0.8":
        response, new_llm_icm, patient = generate_IPC_bot_response(msg.user, msg.history, msg.llm_icm, msg.patient)
        return {"response": response, "llm_icm": new_llm_icm, "patient": patient}

    else:
        return {"response": f"[Error] Unknown bot type: {bot}"}

@app.post("/save-dialogue")
def save_dialogue(payload: dict):
    dialogue = payload.get("dialogue")
    bot = payload.get("bot")
    index = payload.get("index")
    user_id = payload.get("userid")

    filename = f"chatlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.json"
    path = os.path.join(LOG_DIR, filename)
    print("Empfangener user_id:", user_id)
    log = {
        "dialogue": dialogue,
        "bot": bot,
        "userid": user_id,
        "evaluation": []
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return {"status": "saved", "filename": filename}

@app.post("/classify")
async def save_classification(entry: ClassificationEntry):
    if not os.path.exists(CLASS_FILE):
        return {"error": "Sentence data not found"}, 404

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Match sentence and append classification
    matched = False
    for item in data:
        if item.get("sentence") == entry.sentence:
            item.setdefault("classifications", []).append({
                "dominance": entry.dominance,
                "friendliness": entry.friendliness,
                "classificator": entry.classificator
            })
            matched = True
            break

    if not matched:
        return {"error": "Sentence not found"}, 404

    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "success"}

@app.post("/evaluate")
def evaluate(entry: Evaluation):
    timestamp = datetime.utcnow().isoformat()
    entry_dict = entry.dict()
    log_filename = entry_dict.pop("log_filename")
    entry_dict["evaluator"] = "human"
    entry_dict["timestamp"] = timestamp

    path = os.path.join(LOG_DIR, log_filename)
    if os.path.exists(path):
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("evaluation", []).append(entry_dict)
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
        return {"status": "evaluated"}

    return {"error": "log file not found"}


@app.post("/evaluate-summary")
async def save_evaluation_summary(summary: EvaluationSummary):
    try:
        if not os.path.exists(SUMMARY_FILE):
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summaries = json.load(f)

        summaries.append(summary.dict())

        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

        return {"status": "success"}

    except Exception as e:
        return {"error": str(e)}



@app.get("/download/{kind}")
def download(kind: str):
    if kind == "classification":
        path = CLASS_FILE
    elif kind == "evaluation":
        path = EVAL_FILE
    elif kind == "users":
        path = USER_FILE
    else:
        return {"error": "invalid kind"}

    with open(path, "r", encoding="utf-8") as f:
        return {"data": json.load(f)}

