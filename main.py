# import openai # for chatgpt
from openai import OpenAI # for uniGPT


import re
import json
import numpy as np
import random
from dotenv import load_dotenv

#für abspeichen
import os
from datetime import datetime
import uuid

from state_dist import change_prob

# ======= Logging vorbereiten ========
#conversation_history = []
conversation_log = []

def save_conversation_log(log_path, conversation_log):
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Fehler beim Speichern der Konversation:", e)



# ======= Konfig und Variablen =========

load_dotenv()  # lädt automatisch aus `.env`

# --- Key for UNiGPT + url change ---
api_key = os.getenv("UNI_API_KEY")
base_url = os.getenv("BASE_URL")

ipc_descriptions = {
    (0, 0): "Extrem zurückhaltend, sanft, vertrauensvoll – wirkt unsicher, fügt sich ruhig, sucht Nähe.",
    (0, 1): "Etwas sicherer, aber freundlich zurückhaltend – dankbar, positiv gestimmt.",
    (0, 2): "Kooperativ-freundlich-neutral – fühlt sich wohl, zeigt Offenheit.",
    (0, 3): "Freundlich-dominant – weist weich-respektvoll auf eigene Bedürfnisse hin.",
    (0, 4): "Sehr dominant, aber warm: energisch überzeugt, freundlich fordernd.",

    (1, 0): "Sanft und etwas schüchtern freundlich – höflich, dankbar, wenig Initiative.",
    (1, 1): "Kooperativ, höflich, freundlich, aber folgt eher.",
    (1, 2): "Freundlich-zugänglich, emotiver Ton, spricht offen über Symptome.",
    (1, 3): "Direkt-freundlich – klar im Ausdruck, aber positiv und kooperativ.",
    (1, 4): "Direkt-freundlich-konfrontativ: klare Forderungen mit charmantem Ton.",

    (2, 0): "Passiv, sachlich und kooperativ – antwortet knapp, vermeidet Emotion.",
    (2, 1): "Sachlich und etwas weniger zurückhaltend – teils fragend, teils zustimmend.",
    (2, 2): "Rational-ausgewogen, klar und sachlich – weder warm noch kühl.",
    (2, 3): "Dominant-neutral – bestimmt, sachlich, stellt Anforderungen.",
    (2, 4): "Assertiv-neutral: deutliche Ansagen, fokussiert, wenig Emotion.",

    (3, 0): "Reserviert-passiv, skeptisch – hält Distanz, vertraut kaum.",
    (3, 1): "Freundlich-zurückhaltend, aber skeptisch – erwartet Klarheit.",
    (3, 2): "Skeptisch-neutral – fragt gezielt, hält Distanz, hinterfragt.",
    (3, 3): "Dominant-vorsichtig – kritisch-sachlich, eigener Standpunkt ist deutlich.",
    (3, 4): "Dominant-vorsichtig: kritisch, konfrontativ aber ohne persönliche Angriffe.",

    (4, 0): "Sehr passiv, aber kühl und bissig – zurückweisend ohne eigene Stimme.",
    (4, 1): "Reserviert freundlich-submissiv, aber kühl – eher distanziert.",
    (4, 2): "Kalt-neutral – knapp, sachlich, wenig emotional, minimal kooperativ.",
    (4, 3): "Dominant-kühl – fordert klar und schnippisch, wenig Empathie.",
    (4, 4): "Sehr dominant-kalt: fordernd, aggressiv-sarkastisch, wenig kooperativ."
    }


patient_profiles = {
    "Max H.": {
        "Name": "Max H.",
        "Alter": 42,
        "Geschlecht": "Männlich",
        "Beruf": "Bürokaufmann",
        "Familienstand": "Verheiratet, zwei Kinder",
        "Hauptsymptomatik": "Chronische Müdigkeit seit mehreren Monaten, unabhängig von Schlafdauer",
        "Weitere Symptome": [
            "Konzentrationsstörungen",
            "Gereiztheit",
            "Häufige Spannungskopfschmerzen",
            "Einschlafprobleme"
        ],
        "Psychischer Zustand": [
            "Gefühl von innerer Leere und Antriebslosigkeit",
            "Leicht reizbar, schnell überfordert",
            "Gelegentlich niedergeschlagen, depressive Verstimmungen"
        ],
        "Soziale Umstände": "Hohe berufliche Belastung (40+ Stunden/Woche), kaum Freizeit, wenig soziale Kontakte außerhalb der Familie",
        "Familiäres Umfeld": "Unterstützendes, aber ebenso belastetes Umfeld (Ehefrau mit Teilzeitjob, Kinder 6 und 9 Jahre alt)",
        "Lebensstil": "Bewegungsmangel, ungesunde Ernährung, kein Sport",
        "Suchtverhalten": "4–5 Tassen Kaffee täglich, gelegentlich Alkohol am Wochenende",
        "Charaktereigenschaften": [
            "Pflichtbewusst und zuverlässig",
            "Hohe Selbstansprüche, perfektionistisch",
            "Schwierigkeiten, Hilfe anzunehmen"
        ],
        "Schlafverhalten": "Einschlafdauer über 45 Minuten, häufiges Aufwachen, unausgeruht am Morgen",
        "Vorerkrankungen": "Keine körperlichen Grunderkrankungen, einmalige Burnout-Phase vor 5 Jahren"
    },
    "Anna S.": {
        "Name": "Anna S.",
        "Alter": 51,
        "Geschlecht": "Weiblich",
        "Beruf": "Sekretärin",
        "Familienstand": "Ledig",
        "Hauptsymptomatik": "Schwere depressive Verstimmung, Angst, Suizidgedanken (prämenstruell verstärkt)",
        "Weitere Symptome": [
            "Energieverlust", "Konzentrationsstörungen", "soziale Isolierung"
        ],
        "Psychischer Zustand": [
            "Langjähriges MDD mit Angstzuständen", "tendenzielle Hoffnungslosigkeit"
        ],
        "Soziale Umstände": "Lebt alleine, eingeschränktes soziales Netzwerk",
        "Charaktereigenschaften": ["Niedriges Selbstwertgefühl", "Vermeidungsverhalten"],
        "Vorerkrankungen": "Hypothyreose, Vitamin-D-Mangel" 
    },
    "Luca M.": {
        "Name": "Luca M.",
        "Alter": 29,
        "Geschlecht": "Männlich",
        "Beruf": "Journalist",
        "Familienstand": "Single",
        "Hauptsymptomatik": "Post-COVID Fatigue: anhaltende Erschöpfung, Atemnot, Brain fog",
        "Weitere Symptome": [
            "Kopf- und Muskelschmerzen", "Gedächtnisprobleme", "Arbeitsunfähigkeit"
        ],
        "Psychischer Zustand": ["Ängstliche Sorge um Gesundheit", "frustrierte Stimmung"],
        "Soziale Umstände": "Wiederholte Krankheitsausfälle, Isolation",
        "Charaktereigenschaften": ["Perfektionistisch", "frustriert, sauer"],
        "Vorerkrankungen": "Infektion vor 6 Monaten – Long-COVID nach multizentrischer Kohortenanalyse" 
    },
    "Maria T.": {
        "Name": "Maria T.",
        "Alter": 68,
        "Geschlecht": "Weiblich",
        "Beruf": "Im Ruhestand",
        "Familienstand": "Verwitwet",
        "Hauptsymptomatik": "Chronische Ängstlichkeit und Schlafstörungen",
        "Weitere Symptome": ["Herzrasen (situationsabhängig)", "Muskelverspannungen", "Unruhe"],
        "Psychischer Zustand": ["Generalisierten Angststörung (GAD)", "Überängstlich"],
        "Soziale Umstände": "Allein lebend, wenige soziale Kontakte",
        "Charaktereigenschaften": ["Vorsichtig", "kontrollbedürftig"],
        "Vorerkrankungen": "Hypertonie, keine psychiatrische Vorgeschichte"
    },
    "Jasmin K.": {
        "Name": "Jasmin K.",
        "Alter": 15,
        "Geschlecht": "Weiblich",
        "Beruf": "Schülerin",
        "Familienstand": "lebt mit Eltern",
        "Hauptsymptomatik": "ADHS: Aufmerksamkeitsschwierigkeiten, Hyperaktivität",
        "Weitere Symptome": ["Impulsivität", "frustrationsbedingt starke Gereiztheit"],
        "Psychischer Zustand": ["Schnell überfordert", "niedrige Frustrationstoleranz"],
        "Soziale Umstände": "Mobbingerfahrungen in der Schule",
        "Charaktereigenschaften": ["Energiegeladen", "spontan", "rebellisch"],
        "Vorerkrankungen": "ADHS seit Kindheit diagnostiziert"
    }
}



def inital_personality(steckbrief):

    ipc = user_classification(steckbrief)
    #ipc_semantic = ipc_descriptions[(ipc[0], ipc[1])]

    #client = OpenAI(api_key = api_key, base_url = base_url)
    #completion = client.chat.completions.create(
    #    messages = [{"role": "developer", "content": "Bitte fasse den Steckbrief kurz zusammen, damit er als Anleitung für eine KI dient. Inkludiere den Charakter"}, {"role": "user", "content": f"{steckbrief} mit persönlichkeit: {ipc_semantic}"}],
    #    model = "Llama-3.3-70B",)
    
    #starting = completion.choices[0].message.content
    return steckbrief, ipc


def choose_patient():
    random_profile_name = random.choice(list(patient_profiles.keys()))
    random_profile = patient_profiles[random_profile_name]

    # In JSON umwandeln, falls du es als Eingabe an eine API übermitteln willst
    input_for_llm = json.dumps(random_profile, ensure_ascii=False, indent=2)
    return inital_personality(input_for_llm)




def build_instruct_ipc(friendliness, dominance, steckbrief, i = 0):
    desc = ipc_descriptions[(friendliness, dominance)]

    rup = f"""
        Aufgrund des Verlaufs des gespräch bist du nun eher {desc}. Behalte deine Rolle als Patient bei, wie sie dir am Anfang gegeben wurde. Antworte passend zu deinem Charakter. 
    """
    if i ==0:
        return f"""
        Du sollst einen Patienten mit einer vorgefertigten persönlichkeit basierend auf einem Profil rollenspielen. Du musst immer in diesem Charakter bleiben und dem Doktor natürlich antworten.

        Profil:
        {steckbrief}

    Dein Inter-Persönlicher Stil ist so beschrieben: {desc}.

        Ab jetzt sollte jede Antwort dem hier folgen:
        - Bleibe der Charakter
        - Passe dich an die gegebenen Dominanz und Freundlichkeitslevel an
        - Brich nie deine Rolle
        Antworte immer nur als der Patient. """
    
    elif i == 1:
        return f"""
            Stell dir vor, du bist eine Person mit einer einzigartigen Lebensgeschichte und bestimmten Persönlichkeitszügen. Du erinnerst dich an Erfahrungen, Gefühle und Denkweisen, die deine Sicht auf andere Menschen und Gespräche prägen.

            Hier ist dein persönliches Profil:
            {steckbrief}

            Dein zwischenmenschlicher Kommunikationsstil ist so beschrieben:
            {desc}

            Deine Aufgabe ist es, aus dieser inneren Perspektive heraus auf Gesprächsbeiträge zu reagieren – so, wie du es als diese Person wirklich tun würdest. Nicht als Schauspiel, sondern als echte Innenwelt.

            Dabei spielt es eine Rolle, wie du die Stimmung und Haltung deines Gegenübers einschätzt (z. B. ob er freundlich oder bestimmend wirkt). Diese Einschätzung beeinflusst, wie du antwortest – ob du dich zurückziehst, dich öffnest oder dich verteidigst.

            Sprich stets in der Stimme dieser Person – nicht als KI, nicht als Beobachter, sondern als sie selbst.
            Du denkst, fühlst und antwortest aus ihrer Sicht.
            """



    # ========== ICM state-chart Logik ==========
def change2 (user_state, current_llm):

    #print(current_llm)
    new_friendliness,friendliness_dist = change_prob(user_state[0], current_llm[0], strategy = "mimic")
    new_dominance, dominance_dist = change_prob(user_state[1], current_llm[1], strategy = "complement")

    return [new_friendliness, new_dominance], [friendliness_dist, dominance_dist]




def user_classification(prompt):
    ins_sep= """

                # Regeln:

                Du bekommst einen Ausschnitt aus einem Dialog. Klassifiziere den Ausschnitt im Interpersonal Circumplex Model. Jede Achse hat den Wert 0-4, wobei ein höherer Wert mehr Dominanz und mehr Kälte entspricht

                Dominanz Level:
                - 0: Sehr passiv — zögerlich, sehr submissiv.
                - 1: leicht submissiv — resertviert aber kooperativ.
                - 2: Neutral — balancierter Ton.
                - 3: leicht dominant — Meinungsklar aber Respektvoll.
                - 4: sehr dominant — Sehr direkt und konfrontativ.

                Freundlichkeits Level:
                - 0: Warm — empathisch, sehr freundlich, vertrauensvoll.
                - 1: Etwas freundlich - nett, emotional ausdrucksvoll.
                - 2: Neutral — emotional eher flach und faktenbasiert.
                - 3: Vorsichtig — skeptisch, Distanziert.
                - 4: Kalt — unverschämt, böse sarkastisch.d

                Antworte nur mit der Klassifikation in diesem Format:

                d:0-4, f:0-4

               Es sollten nur sieben Zeichen in der Antwort sein: Das d, doppelpunkt, wert für d, komma, f, doppelpunkt , wert für f
            """
    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = [{"role": "developer", "content": ins_sep}, {"role": "user", "content": prompt}],
        model = "Llama-3.3-70B",)

    user_icm = completion.choices[0].message.content
    #print (user_icm)
    match = re.search(r"d:(\d), f:(\d)", user_icm)
    if match:
        d_value = match.group(1)
        f_value = match.group(2)
        user_icm_state = [int(f_value), int(d_value)]
        #print("user icm state:", user_icm_state)
    else:
        # standard values
        print("Kein Treffer gefunden")
        user_icm_state = [2, 2]

    return user_icm_state


# ========== Hauptfunktion ==========
def chat_IPC_Bot(prompt, changeability, model, conversation_history, llm_icm_state, patient_intro = "", j = 0):

    global conversation_log
    # global conversation_history

    print(conversation_history)

    if  len(conversation_history) <=1:
        patient_intro, initial_ipc_state = choose_patient()
        llm_icm_state = initial_ipc_state


    # Aktuelle Eingabe hinzufügen
    #conversation_history.append({"role": "user", "content": prompt})

    user_icm_state = user_classification(prompt)


    new_llm_state, prob_dist = change2(user_icm_state, llm_icm_state)

    new_llm_state[0] = int(new_llm_state[0])
    new_llm_state[1] = int(new_llm_state[1])


    #print("new llm icm state:", new_llm_state)

    #print (patient_intro)
    ins = build_instruct_ipc(new_llm_state[0],new_llm_state[1], patient_intro, i = j)

    #print([{"role": "developer", "content": ins}] + conversation_history)

    #if len(conversation_history) <= 1:
        # ==== 2. Add to conversation and set IPC state ====
    #    ins = f"verhalte dich passend zu dieser Patientenbeschreibung: {patient_intro}, während du charakterlich eher {ins} bist"
    # --- UniGPT request ---
    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = [{"role": "developer", "content": ins}] + conversation_history,
        model = model,)
    # --- End of API calls ---
    msg = completion.choices[0].message.content

    # Antwort dem Verlauf hinzufügen
    conversation_history.append({"role": "assistant", "content": msg})

    #Kürze Verlauf, wenn zu lang (4 Dialogrunden = 8 Nachrichten)
    MAX_TURNS = 10
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]

    # === log speichern ===
    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "bot": "IPC_Framework",
        "prompt": prompt,
        "response": msg,
        "changeability": changeability,
        "IPC": {
            "states": {
                "user": {
                    "friendliness": int(user_icm_state[0]),
                    "dominance": int(user_icm_state[1])
                },
                "chatbot": {
                    "friendliness": int(new_llm_state[0]),
                    "dominance": int(new_llm_state[1])
                },
                "probability_dists": {
                    "friendliness": prob_dist[0],
                    "dominance": prob_dist[1]
                }
            }
        }
    })


    log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
    log_path = os.path.join(log_dir, log_filename)

    save_conversation_log(log_path, conversation_log)

    return msg, [user_icm_state[0], user_icm_state[1]], new_llm_state, conversation_history, patient_intro



def chat_standard_bot(prompt, conversation_history, model):

    global conversation_log
    
    instruct = """
        Du spielst die Rolle eines Patienten in einer ärtztlichen Sprechstunde. Du bist hier für eine Einschätzung deiner Symptome und sollst so menschlich wie möglich reden. Persönlichkeit, Stimmung und Hintergrundinformationen sollen in jeder conversation wechseln. Manchmal bist du ruhig, manchmal ängstlich, sauer freundlich, sarkastisch oder ähnliches.

        Bleibe bei deinem Charakter. Du bist keine KI.

        Wenn der Doktor antwortet antworte passend zu deinem Szenario und Charakter. Zeige Symptome, frag Fragen oder zähle den Doktor sogar an, abhängig von deiner Persönlichkeit.

        Antworte nur textlich, beschreibe keine Bewegungen und ähnlichens

    """



    #conversation_history.append({"role": "user", "content": prompt})

    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages =  [{"role": "developer", "content": instruct}] + conversation_history,
        model = model,)

    answer = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    
    MAX_TURNS = 10
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]

    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "bot": "gpt_default",
        "prompt": prompt,
        "response": answer
    })

    log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/no_ipc"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
    log_path = os.path.join(log_dir, log_filename)

    save_conversation_log(log_path, conversation_log)
    return answer, conversation_history





def generate_IPC_bot_response(user_input, history, llm_icm_state = [2,2], patient_intro = ""):
    """
    Public interface to use your chatbot externally.
    `history` is a list of {"role": "user"|"assistant", "content": "..."}
    """
    patient = patient_intro
    history = history.copy()
    llm_icm_state = llm_icm_state.copy()

    # Default or fixed changeability; can be made dynamic later
    changeability = 0.7

    response, user_icm_state, new_llm_icm, history, patient_intro = chat_IPC_Bot(user_input, changeability, "gemma-3", history, llm_icm_state, patient_intro=patient, j = 0)

    return response, new_llm_icm, patient_intro

def generate_gpt_default(user_input, history):
    """
    Public interface to use your chatbot externally.
    `history` is a list of {"role": "user"|"assistant", "content": "..."}
    """
    history = history.copy()
    response, blub = chat_standard_bot(user_input, history, "gemma-3")
    return response


def generate_diff_change_prob(user_input, history, llm_icm_state = [2,2], patient_intro = ""):
    """
    Public interface to use your chatbot externally.
    `history` is a list of {"role": "user"|"assistant", "content": "..."}
    """
    patient = patient_intro
    history = history.copy()
    llm_icm_state = llm_icm_state.copy()

    # Default or fixed changeability; can be made dynamic later
    changeability = 0.5

    response, user_icm_state, new_llm_icm, history, patient_intro = chat_IPC_Bot(user_input, changeability, "gemma-3", history, llm_icm_state, patient_intro=patient, j = 1)

    return response, new_llm_icm, patient_intro


# ========== Main Loop ==========
if __name__ == "__main__":

    changeability = random.uniform(0.3, 0.9)
    print (changeability)

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Exiting...")
            break

        if user_input.lower().startswith("change "):
            try:
                new_val = float(user_input.split()[1])
                if 0.0 <= new_val <= 1.0:
                    changeability = new_val
                    print(f"Changeability wurde auf {changeability:.2f} gesetzt.")
                else:
                    print("Bitte einen Wert zwischen 0.0 und 1.0 eingeben.")
            except (IndexError, ValueError):
                print("Verwendung: 'change 0.5' — eine Zahl zwischen 0.0 und 1.0.")
            continue

        resp, icm = chat_IPC_Bot(user_input, changeability)
        #resp = chat_standard_bot(user_input)
        print("Chatbot:", resp)