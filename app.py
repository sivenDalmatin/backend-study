from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
from main import generate_IPC_bot_response, generate_gpt_default, generate_llama_ipc
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
EVAL_FILE = os.path.join(DATA_DIR, "evaluations.json")  # legacy
USER_FILE = os.path.join(DATA_DIR, "users.json")
LOG_DIR = os.path.join(DATA_DIR, "chatlogs")
EVAL_DIR = os.path.join(DATA_DIR, "evaluations")  # new
SUMMARY_FILE = os.path.join(DATA_DIR, "evaluation_summary.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

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

def safe_append_and_backup(json_path_local, filename_in_repo, new_entry, unique_key=None):
    try:
        repo_url = os.environ["GITHUB_REPO_URL"]
        token = os.environ["GITHUB_BACKUP_TOKEN"]
        tmp_dir = "/tmp/github_backup"

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        clone_url = repo_url.replace("https://", f"https://x-access-token:{token}@")
        subprocess.run(["git", "clone", clone_url, tmp_dir], check=True)

        subprocess.run(["git", "-C", tmp_dir, "config", "user.name", "Backup Bot"], check=True)
        subprocess.run(["git", "-C", tmp_dir, "config", "user.email", "backup@localhost"], check=True)

        repo_file_path = os.path.join(tmp_dir, filename_in_repo)
        os.makedirs(os.path.dirname(repo_file_path), exist_ok=True)

        if os.path.exists(repo_file_path):
            try:
                with open(repo_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[Warnung] {filename_in_repo} ist leer oder beschädigt. Initialisiere neu.")
                data = []
        else:
            data = []

        if unique_key:
            exists = any(entry.get(unique_key) == new_entry.get(unique_key) for entry in data)
        else:
            exists = False

        if not exists:
            data.append(new_entry)

            with open(repo_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            subprocess.run(["git", "-C", tmp_dir, "add", filename_in_repo], check=True)
            subprocess.run(["git", "-C", tmp_dir, "commit", "-m", f"Update {filename_in_repo}"], check=True)
            subprocess.run(["git", "-C", tmp_dir, "push"], check=True)

            print(f"[Backup] {filename_in_repo} erfolgreich aktualisiert.")
        else:
            print(f"[Backup] Kein Update notwendig für {filename_in_repo}, Eintrag existiert bereits.")

    except Exception as e:
        print("[Backup-Fehler]", e)

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

    try:
        safe_append_and_backup(USER_FILE, "users.json", info.dict(), unique_key="id")
    except Exception as e:
        print("[Backup-Fehler in register-user]", e)

    return {"status": "saved"}

@app.post("/chat")
def chat(msg: Message):
    bot = msg.bot

    if bot == "gpt_default":
        return {"response": generate_gpt_default(msg.user, msg.history)}
    
    elif bot in ["icm_agent_0.5", "neutral_agent_0.8"]:
        response, new_llm_icm, patient = generate_IPC_bot_response(msg.user, msg.history, msg.llm_icm, msg.patient)
        return {"response": response, "llm_icm": new_llm_icm, "patient": patient}
    
    elif bot == 'llama_agent':
        response, new_llm_icm, patient = generate_llama_ipc(msg.user, msg.history, msg.llm_icm, msg.patient)

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
        "evaluation": [],
        "filename": filename
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    try:
        safe_append_and_backup(path, f"chatlogs/{filename}", log, unique_key="filename")
    except Exception as e:
        print("[Backup-Fehler in save-dialogue]", e)

    return {"status": "saved", "filename": filename}

@app.post("/evaluate")
def evaluate(entry: Evaluation):
    timestamp = datetime.utcnow().isoformat()
    entry_dict = entry.dict()
    log_filename = entry.log_filename
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

        # ⬇️ Evaluation separat speichern
        eval_filename = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
        eval_path = os.path.join(EVAL_DIR, eval_filename)

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(entry_dict, f, indent=2, ensure_ascii=False)

        try:
            safe_append_and_backup(eval_path, f"evaluations/{eval_filename}", entry_dict, unique_key="timestamp")
        except Exception as e:
            print("[Backup-Fehler bei Evaluation-Datei]", e)

        return {"status": "evaluated"}

    return {"error": "log file not found"}

@app.post("/classify")
async def save_classification(entry: ClassificationEntry):
    if not os.path.exists(CLASS_FILE):
        return {"error": "Sentence data not found"}, 404

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    matched = False
    for item in data:
        if item.get("sentence") == entry.sentence:
            item.setdefault("classifications", []).append({
                "dominance": entry.dominance,
                "friendliness": entry.friendliness,
                "classificator": entry.classificator
            })
            matched = True

            try:
                safe_append_and_backup(CLASS_FILE, "classifications.json", item, unique_key="sentence")
            except Exception as e:
                print("[Backup-Fehler in classify]", e)
            break

    if not matched:
        return {"error": "Sentence not found"}, 404

    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "success"}

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

        try:
            safe_append_and_backup(SUMMARY_FILE, "evaluation_summary.json", summary.dict(), unique_key=None)
        except Exception as e:
            print("[Backup-Fehler in evaluate-summary]", e)

        return {"status": "success"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/download/{kind}")
def download(kind: str):
    if kind == "classification":
        path = CLASS_FILE
        with open(path, "r", encoding="utf-8") as f:
            return {"data": json.load(f)}

    elif kind == "evaluation":
        evals = []
        for fname in os.listdir(EVAL_DIR):
            if fname.endswith(".json"):
                with open(os.path.join(EVAL_DIR, fname), "r", encoding="utf-8") as f:
                    evals.append(json.load(f))
        return {"data": evals}

    elif kind == "users":
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return {"data": json.load(f)}

    return {"error": "invalid kind"}