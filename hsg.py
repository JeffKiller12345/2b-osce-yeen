#!/usr/bin/env python3
"""
generate_stations.py
--------------------
Builds a bank of homogeneous OSCE history-taking stations using the Gemini API.

How it works
  1. Parses presentations.md (the merged presentation -> diagnosis list with #common tags).
  2. Picks a presentation + underlying diagnosis (weighted towards #common, never repeating a pair
     already in the bank).
  3. Picks 2-3 examiner question types (summarise findings / differential / next steps / management).
  4. Asks Gemini for a structured JSON station (patient details, patient script, examiner Q&A,
     16 history marks + 3 question marks).
  5. Validates the JSON (retries if it breaks the rules).
  6. Assembles the final candidate / patient / examiner text using fixed boilerplate, so every
     station has exactly the same structure, and saves it.

Outputs
  station_bank.csv
  stations_txt/        one readable .txt file per station for proofreading

Setup
  pip install google-genai pydantic
  export GEMINI_API_KEY="your-key"        (Windows PowerShell: $env:GEMINI_API_KEY="your-key")

Usage
  python generate_stations.py --dry-run                 # show allocation + prompt, no API call
  python generate_stations.py -n 5                      # generate 5 stations
  python generate_stations.py -n 50 --model gemini-2.5-flash --seed 1
"""

from __future__ import annotations

import csv
import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# CONFIG - edit these to taste
# --------------------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.8-flash"   # change to whichever Gemini model you want to use
TEMPERATURE = 1.0                  # higher = more varied stations
MAX_ATTEMPTS = 3                   # retries per station if validation fails

N_HISTORY_MARKS = 16               # history-taking marks on the mark sheet
N_QUESTION_MARKS = 3               # marks that come from the examiner's questions
# (16 + 3 = 19 marks, excluding initial approach and patient marks)

COMMON_RATIO = 0.8                 # chance a station uses a #common diagnosis
PRESENTATION_COMMON_WEIGHT = 2.0   # #common presentations are picked this many times more often

# Presentations to skip entirely (case-insensitive substring match), e.g. ["Hot (fever", "Diabetic complication"]
EXCLUDE_PRESENTATIONS: list[str] = []

CSV_COLUMNS = [
    "id", "created_at", "model",
    "specialty", "presentation", "presentation_common",
    "diagnosis", "diagnosis_common", "question_types",
    "patient_name", "patient_age", "patient_sex", "patient_dob",
    "candidate_info", "patient_info", "examiner_info",
    "setting", "presenting_complaint", "opening_statement", "patient_script",
    "patient_follow_up_question", "examiner_questions", "history_marks", "question_marks",
]

# Examiner question types: (key, description shown to the AI)
QUESTION_TYPES = [
    ("summarise_findings", "Briefly summarise your findings"),
    ("differential_diagnosis", "Provide a (differential) diagnosis"),
    ("next_steps", "Suggest appropriate next steps (e.g. important investigations)"),
    ("management_plan", "Describe an appropriate management plan"),
]

# --------------------------------------------------------------------------------------
# Fixed boilerplate (identical in every station)
# --------------------------------------------------------------------------------------
PATIENT_OPENING_INTRO = "After introductions, in response to the student’s opening question or comment, please say:"
PATIENT_OPENING_OUTRO = "(Please ensure you use this wording every time, to ensure consistency)"
PATIENT_FOLLOW_UP_INTRO = "After the consultation has progressed a little please ask the following question"
PATIENT_OUTRO = (
    "When a student asks a question not covered in the information provided, please give a vague or "
    "non-committal reply such as \"I’m not sure about that\", or \"I can’t really remember\"."
)

# --------------------------------------------------------------------------------------
# Style example given to the model
# --------------------------------------------------------------------------------------
EXAMPLE_STATION = """\
Setting: the Emergency Department
Presenting complaint (candidate-facing): chest pain
Patient: Samantha Parkins, 46, female
Opening statement: I have pain in my chest and I’m quite worried about it.

Patient script:
You are an office manager and are usually fit and well. A week ago, you badly sprained your ankle while playing football and you struggled to mobilise for a few days. Your ankle is still a bit swollen and tender, but not red or hot.
Yesterday, you developed a sharp chest pain on the right side of your chest, specifically in the upper chest area. The pain occurs every time you take a breath in and is worse when you take a deep breath. The pain began suddenly and hasn’t fully gone away since it started. On a scale of 1 to 10 you would rate it as 7. The pain does not radiate to other areas such as the arm or back. Movement also makes the pain worse, but it isn’t sore to touch your ribs. You felt a little breathless when you walked into the waiting room from the car park, but you haven’t exerted yourself since you hurt your ankle. You haven’t had a cough or fever. You haven’t felt sick, sweaty or dizzy. You’ve not had any recent travel or flights and you didn’t injure your chest when you injured your ankle.
You have a history of hypertension, which is well-controlled with medication (lisinopril). You have no history of heart disease, diabetes, previous lung problems or previous blood clots. You also have no known allergies to medications or other substances.
You are a non-smoker and consume alcohol occasionally, primarily during social gatherings. You usually engage in regular exercise, but your recent ankle injury has limited your mobility.
Your mother had a deep vein thrombosis (DVT) after an operation.

Patient follow-up question: What do you think is the cause and what tests will I need?

Examiner questions:
1. What do you think is the main problem and differential diagnosis?
Answer: The differential diagnosis is broad but there are risk factors for pulmonary embolism. Other possible diagnoses would be a musculoskeletal injury, chest infection, pneumothorax.
2. If the patient was found to have a pulmonary embolism, how would you treat this condition?
Answer: Students should mention anticoagulation or name specific anticoagulants. They may also describe supportive treatment such as oxygen and analgesia. Students that mention thrombolysis should state that this would be used for high risk PE.

History marks (16):
Asks about Site of pain
Asks about Onset and timing: when it started, sudden or gradual, constant or intermittent
Asks about Character and aggravating or relieving factors
Asks about Radiation and severity
Asks about Breathlessness: current or on exertion
Asks about Respiratory and infective symptoms: at least two of cough, haemoptysis, fever
Asks about Cardiac or autonomic symptoms: at least two of nausea, sweating, dizziness, palpitations, syncope
Asks about Recent injury: asks about the ankle injury and about chest trauma
Asks about Leg symptoms: calf pain or swelling
Asks about Personal VTE and other risk factors: previous DVT or PE, and at least one of malignancy, oestrogen or hormonal therapy, or pregnancy where relevant
Asks about immobility: long-haul travel or flights, recent operation
Asks about Family history: of clots or other relevant conditions
Asks about Past medical history
Asks about Drug history and allergies: both
Asks about Social history: at least two of smoking, alcohol, occupation, exercise
Asks about Ideas, concerns and expectations

Question marks (3) - each linked to an examiner question:
(Question 1) Correctly identifies PE as primary differential
(Question 1) Names another differential eg. musculoskeletal injury, chest infection, pneumothorax
(Question 2) Mentions anticoagulation for treatment
"""

SYSTEM_INSTRUCTION = (
    "You are an experienced UK medical school OSCE examiner and station writer. You write realistic, "
    "clinically accurate history-taking stations for third-year medical students, using UK spelling, "
    "UK clinical practice (NICE/BNF-style) and UK drug names. You follow the requested structure exactly."
)


# --------------------------------------------------------------------------------------
# Structured output schema (what Gemini must return)
# --------------------------------------------------------------------------------------
class ExaminerQuestion(BaseModel):
    question_type: str = Field(description="Must be exactly the question type key requested for this question number.")
    question: str = Field(description="The question the examiner asks the candidate.")
    answer: str = Field(description="Concise model answer / guidance for the examiner.")


class QuestionMark(BaseModel):
    linked_question: int = Field(description="The examiner question number (1-based) this mark belongs to.")
    mark: str = Field(description="One single, separately-markable point, e.g. 'Mentions anticoagulation for treatment'.")


class Station(BaseModel):
    setting: str = Field(description="Phrase completing 'You are in ___.' e.g. 'the Emergency Department', 'a GP surgery'.")
    presenting_complaint: str = Field(description="Short, non-diagnostic phrase for the candidate instructions, e.g. 'chest pain'.")
    patient_name: str = Field(description="Realistic full name.")
    patient_age: int = Field(description="Age in years.")
    patient_sex: str = Field(description="'male' or 'female'.")
    opening_statement: str = Field(description="The patient's first line in lay language. Must not name the diagnosis.")
    patient_script: str = Field(description="Free-text paragraphs in the second person ('You ...') containing everything the patient knows.")
    patient_follow_up_question: str = Field(description="Question the patient asks later in the consultation.")
    examiner_questions: list[ExaminerQuestion]
    history_marks: list[str] = Field(description="History-taking mark sheet items, one mark each, each starting 'Asks about'.")
    question_marks: list[QuestionMark]


# --------------------------------------------------------------------------------------
# Parsing the presentation list
# --------------------------------------------------------------------------------------
@dataclass
class Diagnosis:
    name: str
    common: bool


@dataclass
class Presentation:
    specialty: str
    name: str
    common: bool
    diagnoses: list[Diagnosis]


def split_tag(text: str) -> tuple[str, bool]:
    """Remove '#common' from a string and report whether it was there."""
    common = "#common" in text
    text = re.sub(r"\s*#common\s*", " ", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, common


def split_top_level_commas(text: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def load_presentations(path: Path) -> list[Presentation]:
    specialty = None
    presentations: list[Presentation] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            specialty = line[3:].strip()
            continue
        m = re.match(r"^-\s+\*\*(.+?):\*\*\s*(.+)$", line)
        if not m or not specialty:
            continue
        name, p_common = split_tag(m.group(1).strip())
        diagnoses = []
        for item in split_top_level_commas(m.group(2).replace("*", "")):
            dname, d_common = split_tag(item)
            if dname:
                diagnoses.append(Diagnosis(dname, d_common))
        if not diagnoses:
            continue
        if any(x.lower() in name.lower() for x in EXCLUDE_PRESENTATIONS):
            continue
        presentations.append(Presentation(specialty, name, p_common, diagnoses))
    if not presentations:
        sys.exit(f"No presentations could be parsed from {path}. Check the file format.")
    return presentations


# --------------------------------------------------------------------------------------
# Allocation (which presentation / diagnosis / question types to use)
# --------------------------------------------------------------------------------------
def pick_allocation(presentations, used, rng, common_ratio):
    pairs, weights = [], []
    for p in presentations:
        avail = [d for d in p.diagnoses if (p.specialty, p.name, d.name) not in used]
        if not avail:
            continue
        common = [d for d in avail if d.common]
        other = [d for d in avail if not d.common]
        ratio = common_ratio if (common and other) else (1.0 if common else 0.0)
        pw = PRESENTATION_COMMON_WEIGHT if p.common else 1.0
        for d in common:
            pairs.append((p, d))
            weights.append(pw * ratio / len(common))
        for d in other:
            pairs.append((p, d))
            weights.append(pw * (1 - ratio) / len(other))
    if not pairs or sum(weights) <= 0:
        return None
    return rng.choices(pairs, weights=weights, k=1)[0]


def plan_questions(rng):
    """Choose 2 or 3 distinct question types, kept in a logical order."""
    n = rng.choices([2, 3], weights=[0.6, 0.4], k=1)[0]
    idx = sorted(rng.sample(range(len(QUESTION_TYPES)), n))
    return [QUESTION_TYPES[i] for i in idx]


def random_dob(age: int, today: date, rng) -> date:
    """A random date of birth that is consistent with the given age today."""
    def years_ago(d: date, years: int) -> date:
        try:
            return d.replace(year=d.year - years)
        except ValueError:  # 29 Feb
            return d.replace(year=d.year - years, day=28)

    latest = years_ago(today, age)
    earliest = years_ago(today, age + 1) + timedelta(days=1)
    return earliest + timedelta(days=rng.randint(0, (latest - earliest).days))


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
def build_prompt(p: Presentation, d: Diagnosis, q_plan, today: date, problems=None) -> str:
    q_lines = "\n".join(f"  - Question {i}: question_type = \"{key}\" ({desc})" for i, (key, desc) in enumerate(q_plan, 1))
    prompt = f"""Write ONE history-taking OSCE station for final-year UK medical students.

ALLOCATION (fixed - do not change)
- Specialty: {p.specialty}
- Presenting problem: {p.name}
- Underlying diagnosis the simulated patient actually has: {d.name} ({"a common" if d.common else "a less common"} condition)
- Today's date: {today:%d/%m/%Y}

The presenting problem and diagnosis are shorthand labels from a list. Interpret them sensibly (e.g. "stress" under
"Urinary incontinence" means stress urinary incontinence; "GI cancer" means choose one specific, realistic GI cancer).
Choose a realistic, specific way a patient would present with this problem.

PATIENT RULES
- Choose a realistic setting, name, age and sex that fit the diagnosis. Vary ethnicity and background naturally; avoid stereotypes.
- The opening statement is the patient's first line, in lay language. Keep it short and similar in style to
  "I have pain in my chest and I’m quite worried about it." It must NOT name or hint at the diagnosis.
- The presenting complaint for the candidate is a short, non-diagnostic phrase (e.g. "chest pain", "a swollen leg", "tiredness").
- The patient script is free text in the second person ("You are ...", "You have ..."), in short paragraphs, containing:
  occupation and usual health; a full account of the presenting complaint with the key discriminating positives AND the
  relevant negatives; associated symptoms; relevant risk factors; past medical history; drug history and allergies;
  family history; social history (smoking, alcohol, occupation, exercise as relevant); and the patient's ideas, concerns
  and expectations.
- The script is all the information the patient has. Never name the diagnosis in the script, and keep the patient in lay terms
  (they may know a prior diagnosis they have been told about, e.g. "asthma").
- Make the history clinically accurate and internally consistent. It should be possible, but not trivial, to reach the diagnosis.
  Include one or two plausible red herrings where realistic.
- The patient follow-up question is asked part-way through the consultation. Keep it close to
  "What do you think is the cause and what tests will I need?" (adapt only if the case needs it).
- Do NOT include the date of birth, the opening instructions or the closing instructions - these are added automatically.

MARK SHEET RULES (history marks)
- Write exactly {N_HISTORY_MARKS} history marks, one mark each, in the order a good history would be taken.
- Each mark MUST start with "Asks about" (as in the example). Related points may be bundled into one mark
  using wording like "at least two of ..." or "both".
- Always include marks for: past medical history; drug history and allergies; family history; social history; and
  ideas, concerns and expectations. The remaining {N_HISTORY_MARKS - 5} marks are specific to this presentation and
  diagnosis (characterising the main symptom, associated symptoms, discriminating features for the differential,
  risk factors, red flags).
- Every mark must be answerable from the patient script. No mark may reward something the script does not contain.

EXAMINER QUESTIONS RULES
- Write {len(q_plan)} examiner questions, in this order and of these types:
{q_lines}
- Write each question as the examiner would say it to the candidate (e.g. "What do you think is the main problem and
  differential diagnosis?", "If the patient was found to have X, how would you treat this condition?").
- Each answer is concise guidance for the examiner on what a good answer includes (as in the example).
- Write exactly {N_QUESTION_MARKS} question marks in total across ALL the questions. Every question must have at least
  one mark. Each mark is a single separately-markable point and is linked to its question by number. Write each as a
  short statement such as "Correctly identifies X as primary differential" or "Mentions anticoagulation for treatment".

STYLE EXAMPLE (a different case - copy the style and level of detail, NOT the content)
-----
{EXAMPLE_STATION}-----
"""
    if problems:
        prompt += "\nYour previous attempt was rejected for these reasons. Fix them:\n" + "\n".join(f"- {x}" for x in problems) + "\n"
    return prompt


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate_station(st: Station, q_plan, d: Diagnosis) -> list[str]:
    probs = []

    if len(st.history_marks) != N_HISTORY_MARKS:
        probs.append(f"history_marks has {len(st.history_marks)} items; it must have exactly {N_HISTORY_MARKS}.")
    if len({m.strip().lower() for m in st.history_marks}) != len(st.history_marks):
        probs.append("history_marks contains duplicates.")
    for m in st.history_marks:
        if not m.strip().lower().startswith("asks about"):
            probs.append(f"History mark does not start with 'Asks about': {m!r}")

    if len(st.examiner_questions) != len(q_plan):
        probs.append(f"There must be exactly {len(q_plan)} examiner questions (got {len(st.examiner_questions)}).")
    else:
        for i, (q, (key, _)) in enumerate(zip(st.examiner_questions, q_plan), 1):
            if q.question_type != key:
                probs.append(f"Question {i} must have question_type '{key}' (got '{q.question_type}').")

    if len(st.question_marks) != N_QUESTION_MARKS:
        probs.append(f"question_marks has {len(st.question_marks)} items; it must have exactly {N_QUESTION_MARKS}.")
    covered = {qm.linked_question for qm in st.question_marks}
    for i in range(1, len(q_plan) + 1):
        if i not in covered:
            probs.append(f"Examiner question {i} has no linked mark.")
    if any(qm.linked_question not in range(1, len(q_plan) + 1) for qm in st.question_marks):
        probs.append("A question mark is linked to a question number that does not exist.")

    if not 16 <= st.patient_age <= 100:
        probs.append(f"patient_age {st.patient_age} is outside 16-100.")
    if st.patient_sex.strip().lower() not in {"male", "female"}:
        probs.append("patient_sex must be 'male' or 'female'.")

    label = re.sub(r"\(.*?\)", "", d.name).strip().lower()
    if len(label) >= 8:
        for field_name, text in (("opening_statement", st.opening_statement), ("presenting_complaint", st.presenting_complaint)):
            if label in text.lower():
                probs.append(f"{field_name} gives away the diagnosis ('{label}').")
    return probs


# --------------------------------------------------------------------------------------
# Assembling the final text (fixed structure)
# --------------------------------------------------------------------------------------
def assemble_candidate(st: Station) -> str:
    return f"You are in {st.setting.strip()}.\nTake a focused history from this patient with {st.presenting_complaint.strip()}."


def assemble_patient(st: Station, dob: date) -> str:
    return (
        f"You are in {st.setting.strip()}.\n"
        f"Name: {st.patient_name.strip()}\n"
        f"Age: {st.patient_age} DOB: {dob:%d/%m/%Y}\n"
        f"{PATIENT_OPENING_INTRO}\n"
        f"\"{st.opening_statement.strip()}\"\n"
        f"{PATIENT_OPENING_OUTRO}\n"
        f"{st.patient_script.strip()}\n"
        f"{PATIENT_FOLLOW_UP_INTRO}\n"
        f"“{st.patient_follow_up_question.strip()}”\n"
        f"{PATIENT_OUTRO}"
    )


def assemble_examiner(st: Station) -> str:
    lines = ["Questions"]
    for i, q in enumerate(st.examiner_questions, 1):
        lines.append(f"{i}. {q.question.strip()}")
        lines.append(f"Answer: {q.answer.strip()}")
    lines.append("Mark Sheet")
    lines.extend(m.strip() for m in st.history_marks)
    lines.append("Examiner’s Questions")
    for n, qm in enumerate(sorted(st.question_marks, key=lambda x: x.linked_question), 1):
        lines.append(f"{n}- {qm.mark.strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Gemini call
# --------------------------------------------------------------------------------------
def make_client():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Missing package. Run: pip install google-genai pydantic")
    try:
        return genai.Client(http_options=types.HttpOptions(timeout=180_000))
    except Exception as e:
        sys.exit(f"Could not create Gemini client: {e}\nSet your key with: export GEMINI_API_KEY=\"your-key\"")


def generate_station(client, model, p, d, q_plan, today):
    from google.genai import types

    problems = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(p, d, q_plan, today, problems)
        try:
            print(f"    attempt {attempt}: calling {model}...", end="", flush=True)
            t0 = time.time()
            chunks = []
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=Station,
                    temperature=TEMPERATURE,
                ),
            ):
                if chunk.text:
                    chunks.append(chunk.text)
                    print(".", end="", flush=True)
            print(f" done in {time.time() - t0:.0f}s", flush=True)

            text = "".join(chunks)
            if not text:
                raise ValueError("Empty response (possibly blocked by safety filters)")
            station = Station.model_validate_json(text)
        except Exception as e:
            print(f"\n    attempt {attempt}: API/parse error: {e}")
            problems = None
            time.sleep(2 ** attempt)
            continue

        problems = validate_station(station, q_plan, d)
        if not problems:
            return station
        print(f"    attempt {attempt}: validation failed:")
        for x in problems:
            print(f"      - {x}")
    return None


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def load_used(bank_path: Path):
    used, count = set(), 0
    if bank_path.exists() and bank_path.stat().st_size > 0:
        with bank_path.open(newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                used.add((rec["specialty"], rec["presentation"], rec["diagnosis"]))
                count += 1
    return used, count


def save_station(bank_path: Path, txt_dir: Path, number: int, p, d, q_plan, model, st: Station, dob: date):
    candidate = assemble_candidate(st)
    patient = assemble_patient(st, dob)
    examiner = assemble_examiner(st)
    station_id = f"hx-{number:04d}"
    parts = st.model_dump()

    def js(x):  # lists -> JSON text, which imports into jsonb columns
        return json.dumps(x, ensure_ascii=False)

    row = {
        "id": station_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "specialty": p.specialty,
        "presentation": p.name,
        "presentation_common": str(p.common).lower(),
        "diagnosis": d.name,
        "diagnosis_common": str(d.common).lower(),
        "question_types": js([k for k, _ in q_plan]),
        "patient_name": st.patient_name,
        "patient_age": st.patient_age,
        "patient_sex": st.patient_sex.strip().lower(),
        "patient_dob": dob.isoformat(),          # YYYY-MM-DD so Postgres reads it as a date
        "candidate_info": candidate,
        "patient_info": patient,
        "examiner_info": examiner,
        "setting": st.setting,
        "presenting_complaint": st.presenting_complaint,
        "opening_statement": st.opening_statement,
        "patient_script": st.patient_script,
        "patient_follow_up_question": st.patient_follow_up_question,
        "examiner_questions": js(parts["examiner_questions"]),
        "history_marks": js(parts["history_marks"]),
        "question_marks": js(parts["question_marks"]),
    }

    write_header = not bank_path.exists() or bank_path.stat().st_size == 0
    with bank_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    txt_dir.mkdir(parents=True, exist_ok=True)
    fname = txt_dir / f"{station_id}_{slugify(p.name)}_{slugify(d.name)}.txt"
    fname.write_text(
        f"{station_id} | {p.specialty} | {p.name} -> {d.name}\n\n"
        f"=== CANDIDATE INFORMATION ===\n{candidate}\n\n"
        f"=== PATIENT INFORMATION ===\n{patient}\n\n"
        f"=== EXAMINER INFORMATION ===\n{examiner}\n",
        encoding="utf-8",
    )
    return station_id


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate OSCE history stations with Gemini.")
    ap.add_argument("-n", "--count", type=int, default=1, help="number of stations to generate")
    ap.add_argument("--list", default="presentations.md", help="presentation/diagnosis list (markdown)")
    ap.add_argument("--out", default="station_bank.csv", help="CSV station bank (appended to)")
    ap.add_argument("--txt-dir", default="stations_txt", help="folder for readable .txt copies")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--common-ratio", type=float, default=COMMON_RATIO, help="probability of a #common diagnosis")
    ap.add_argument("--seed", type=int, default=None, help="random seed (for reproducible allocation)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to pause between API calls")
    ap.add_argument("--dry-run", action="store_true", help="show allocation and prompt only; no API call")
    args = ap.parse_args()

    list_path, bank_path, txt_dir = Path(args.list), Path(args.out), Path(args.txt_dir)
    if not list_path.exists():
        sys.exit(f"Cannot find {list_path}. Save the merged presentation list as that file (or pass --list).")

    presentations = load_presentations(list_path)
    n_pairs = sum(len(p.diagnoses) for p in presentations)
    print(f"Loaded {len(presentations)} presentations / {n_pairs} presentation-diagnosis pairs.")

    rng = random.Random(args.seed)
    used, count = load_used(bank_path)
    today = date.today()
    client = None if args.dry_run else make_client()

    made = 0
    for _ in range(args.count):
        alloc = pick_allocation(presentations, used, rng, args.common_ratio)
        if alloc is None:
            print("Every presentation/diagnosis pair is already in the bank. Nothing left to generate.")
            break
        p, d = alloc
        q_plan = plan_questions(rng)
        print(f"\n[{made + 1}/{args.count}] {p.specialty} | {p.name} -> {d.name} "
              f"({'common' if d.common else 'less common'}) | questions: {[k for k, _ in q_plan]}")

        if args.dry_run:
            print("-" * 80)
            print(build_prompt(p, d, q_plan, today))
            made += 1
            continue

        st = generate_station(client, args.model, p, d, q_plan, today)
        if st is None:
            print("    FAILED after all attempts - skipping (pair not marked as used).")
            continue

        dob = random_dob(st.patient_age, today, rng)
        count += 1
        station_id = save_station(bank_path, txt_dir, count, p, d, q_plan, args.model, st, dob)
        used.add((p.specialty, p.name, d.name))
        made += 1
        print(f"    saved {station_id}")
        time.sleep(args.delay)

    print(f"\nDone. {made} station(s) {'previewed' if args.dry_run else 'saved'}.")


if __name__ == "__main__":
    main()