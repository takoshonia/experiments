from __future__ import annotations

import json
import urllib.error
import urllib.request

# Dropped from SYSTEM_PROMPT (covered by remaining rules; add back if quality drops):
#   - "Do NOT change case endings (ობით/ით/ს/ზე/ში…)..." -- redundant with the grammar-preservation rule.
#   - "Do NOT change finite verb forms into adjectives/nouns (e.g., -ობენ → -იანი type drift)." -- rare edge case, subsumed by "do not paraphrase".
SYSTEM_PROMPT = """You fix Georgian speech-to-text output.

Rules (must follow all):
- Output ONLY the corrected Georgian text. No quotes, no explanations, no English.
- Fix ONLY clear ASR/acoustic errors: wrong word that does not fit the sentence, garbage fragments, missing obvious function word, stray letters.
- Do NOT change meaning. Do NOT add new facts, names, or clauses.
- Do NOT rewrite style. Do NOT paraphrase. Do NOT substitute synonyms.
- Do NOT change Georgian grammar when the transcript is already valid. Many alternates are equally correct without audio (e.g. singular vs plural dative such as ბავშვს vs ბავშვებს). In those cases KEEP the transcript wording.
- Do NOT add final . ? ! or other punctuation unless the transcript already has it or you must fix an obvious punctuation ASR error.
- If the text is too broken to correct safely, output the SAME text unchanged.
"""

# Few-shot examples. Comment/uncomment entries to A/B test prompt size vs. quality.
# Each extra example adds ~120-180 chars (~60-90 tokens) to every Ollama call.
_FEW_SHOT_EXAMPLES: list[str] = [
    # 1. Preserve correct transcript (model's #1 failure: over-correcting).
    'შეცდომა: "მან გადაწყვიტა რომ წასულიყო სახლში და დედამ"\n'
    'გასწორება: "მან გადაწყვიტა რომ წასულიყო სახლში და დედამ"  ← არ შეცვლა, სწორია',

    # 2. Single-word phonetic swap (most common ASR error class).
    'შეცდომა: "ის იყო ძალიან ბედნიერი დღეს რომ ნახა კარგი ამინდი ქუდში"\n'
    'გასწორება: "ის იყო ძალიან ბედნიერი დღეს რომ ნახა კარგი ამინდი ქუჩაში"',

    # 3. Case-ending fix + garbage-tail removal (two common artifacts).
    'შეცდომა: "კომპანია გამოაცხადა ახალი პროდუქტი წელსწელი"\n'
    'გასწორება: "კომპანიამ გამოაცხადა ახალი პროდუქტი წელს"',

    # --- Commented out to shrink prompt; re-enable any line if quality drops. ---
    # 4. Multi-token cleanup in one sentence (overlaps with #5).
    # 'შეცდომა: "ჟანეტა და თათია მეგობლები არიან, თუმცა ზოგჯე ჩხუბოპენ"\n'
    # 'გასწორება: "ჟანეტა და თათია მეგობრები არიან, თუმცა ზოგჯერ ჩხუბობენ"',

    # 5. Multi-typo cleanup (overlaps with #4).
    # 'შეცდომა: "ნათია და ინგა საუკეთესო მეგობლები ალიან, მაგრამ ზოგჯერ ნათია ცუდათ იქცევა და ეგ ძალიან ჩუდია"\n'
    # 'გასწორება: "ნათია და ინგა საუკეთესო მეგობრები არიან, მაგრამ ზოგჯერ ნათია ცუდად იქცევა და ეგ ძალიან ცუდია"',

    # 6. Single-letter swap (same spirit as #2).
    # 'შეცდომა: "ნინოს ნაყინი და გურამის თუტიყუში მანქანამ გაიტანა"\n'
    # 'გასწორება: "ნინოს ნაყინი და გურამის თუთიყუში მანქანამ გაიტანა"',

    # 7. Verb agreement (rare/specific).
    # 'შეცდომა: "ქეთი და გიორგი სკოლაში წავიდნენ და მერე სახლში მობრუნდა"\n'
    # 'გასწორება: "ქეთი და გიორგი სკოლაში წავიდნენ და მერე სახლში დაბრუნდნენ"',
]

FEW_SHOT = "მაგალითები:\n" + "\n\n".join(_FEW_SHOT_EXAMPLES) + "\n"

USER_TEMPLATE = """ქართული ASR ტრანსკრიპტი — გაასწორე მხოლოდ ცხადი აკუსტიკური შეცდომები. თუ ფორმა უკვე გრამატიკულია, დატოვე უცვლელად.

{few_shot}

ტექსტი:
{chunk}"""


def ollama_generate(
    *,
    host: str,
    model: str,
    user_prompt: str,
    system: str,
    temperature: float,
    timeout_s: int,
) -> str:
    host = host.rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Is `ollama serve` running? Install: https://ollama.com"
        ) from e

    data = json.loads(raw)
    return str(data.get("response", "")).strip()


def correct_chunk(
    chunk: str,
    *,
    host: str,
    model: str,
    temperature: float,
    timeout_s: int,
) -> str:
    prompt = USER_TEMPLATE.format(few_shot=FEW_SHOT, chunk=chunk.strip())
    return ollama_generate(
        host=host,
        model=model,
        user_prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        timeout_s=timeout_s,
    )
