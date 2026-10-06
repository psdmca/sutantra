#!/usr/bin/env python3
"""
Tamil Verse AI Agent (தமிழ் செய்யுள் ஆய்வு AI முகவர்)

Reads any Tamil literature file, understands context and verses,
and passes each selected verse to an AI model to convert in-place into:
  1. மூலப் பாடல் (Original verse in code block)
  2. சொற்பிரிப்பு & சந்திப் பிரித்த பாடம் (Word-splitting with '+' markers)
  3. அருஞ்சொற்பொருள் (Archaic & complex words glossary)
  4. எளிய உரை (Simple modern Tamil paraphrase applying 1+2+3)

Usage:
  python3 scripts/tamil_verse_agent.py <filepath> [--start N] [--end M] [--api-key KEY] [--model MODEL]
"""

import os
import re
import sys
import json
import argparse
import urllib.request
import urllib.error

def call_gemini_api(prompt, api_key, model="gemini-2.5-flash"):
    """Calls Gemini REST API using Python standard library without external dependencies."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini API Error {e.code}: {err_body}")
    except Exception as e:
        raise RuntimeError(f"Network error calling Gemini API: {e}")

def build_prompt(book_context, verse_header, verse_lines):
    """Builds the scholarly prompt for the AI model."""
    return f"""நீங்கள் ஒரு சிறந்த செவ்வியல் தமிழ் இலக்கிய அறிஞர் (Classical Tamil Literature Scholar).
கீழே கொடுக்கப்பட்டுள்ள நூல் பின்புலத்தையும் செய்யுளையும் கவனமாக வாசித்து, நயங்களை முழுமையாக உணர்ந்து, பின்வரும் 3 பிரிவுகளுடன் செவ்வியல் முறையில் உரை விளக்கம் செய்து தருக.

[நூல் பின்புலம்]:
{book_context}

[செய்யுள் தலைப்பு / துறை]:
{verse_header}

[செய்யுள் வரிகள்]:
{verse_lines}

கட்டாயமாகப் பின்வரும் 3 பிரிவுகளை மட்டுமே இந்த வடிவத்தில் தருக (வேறு முன்னுரை / பின்னுரை தேவையில்லை):

#### 1. பாடல் (சொற்பிரிப்பு & சந்திப் பிரித்த பாடம்)
```tamil
(செய்யுளைப் பிழையின்றி எளிதில் பொருள் விளங்கும்படி பின்வரும் குறியீடுகளைப் பயன்படுத்திச் சீரமைக்கப்பட்ட முழுமையான பாடல் வரிகள்:
  - '+' : சொற்புணர்ச்சி / கூட்டுச் சொல் பிரிப்பு (எ.கா: தண்+தாது, தீம்+தேன், நீடு+தொறும்)
  - '-' : இடைச்சொல் / விகுதி / உருபுப் பிரிப்பு (எ.கா: செல்வர்-கொல், தன்ன-கொல்)
  - '~' : செய்யுள் அளபெடை நீட்சி (எ.கா: அசை~இ, சிறா~அர், தரூ~உம், வெரூ~உம்)
  - '“...”' : செய்யுளில் வரும் நேரடிக் கூற்று / மேற்கோள்
  - '?' / '!' : வினா மற்றும் உணர்ச்சி முடிபுக் குறிகள்
  - ',' / ';' / '—' : சொற்றொடர் அமைப்பு மற்றும் வாசிப்பு இடைநிறுத்தக் குறிகள்)
```

#### 2. அருஞ்சொற்பொருள் (Archaic & Complex Words Glossary)
- **(அருஞ்சொல்)** : (பொருள் / விளக்கம்)
(செய்யுளில் அமைந்துள்ள அனைத்துக் கடினச் சொற்கள், சங்கச் சொற்களுக்கான தமிழ் விளக்கம்)

#### 3. எளிய உரை (Simple Modern Tamil Paraphrase)
> (1, 2 ஆகியவற்றை அடிப்படையாகக் கொண்டு, திணை மற்றும் துறைப் பின்னணியுடன் கூடிய தெளிவான, நயமான நவீனத் தமிழ்ப் பொழிப்புரை)
"""

def extract_book_context(content):
    """Extracts book title, category, author, period from YAML frontmatter or top lines."""
    context_lines = []
    fm_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        context_lines.append(fm_match.group(1).strip())
    else:
        # Grab first 15 lines as context
        context_lines.append("\n".join(content.split("\n")[:15]))
    return "\n".join(context_lines)

def parse_verses(content):
    """
    Parses verse blocks from the file.
    Matches verses starting with:
      <number>. <thinai/details>
    """
    # Regex to capture verse header and lines up to the next verse or section
    pattern = r"(?:^|\n)((\d+)\.\s+([^\n]+))\n\n((?:(?!\n\d+\.|\n###|\Z).)+)"
    verses = []
    for m in re.finditer(pattern, content, re.DOTALL):
        full_match = m.group(0)
        header = m.group(1).strip()
        num = int(m.group(2))
        body = m.group(4).strip()
        start_pos = m.start()
        end_pos = m.end()
        
        # Check if already converted
        already_converted = ("#### 1. பாடல்" in body) or ("#### 1. மூலப் பாடல்" in body)
        
        verses.append({
            "number": num,
            "header": header,
            "body": body,
            "start": start_pos,
            "end": end_pos,
            "full_match": full_match,
            "already_converted": already_converted
        })
    return verses

def process_file(filepath, start_idx=None, end_idx=None, api_key=None, model="gemini-2.5-flash"):
    """Reads file, extracts verses, passes to AI model, and updates in-place."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
        
    print(f"📖 Reading context from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    book_context = extract_book_context(content)
    verses = parse_verses(content)
    
    if not verses:
        print("No numbered verses found matching pattern (e.g., '1. குறிஞ்சி - ...') in file.", file=sys.stderr)
        return
        
    total_found = len(verses)
    min_num = verses[0]["number"]
    max_num = verses[-1]["number"]
    print(f"Found {total_found} verses (range: {min_num} to {max_num}).")
    
    # Filter verses
    selected_verses = []
    for v in verses:
        num = v["number"]
        if start_idx is not None and num < start_idx:
            continue
        if end_idx is not None and num > end_idx:
            continue
        selected_verses.append(v)
        
    print(f"Targeting {len(selected_verses)} verse(s) for conversion (start={start_idx}, end={end_idx})...")
    
    # Resolve API key
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_api_key:
        print("\n⚠️ Note: No GEMINI_API_KEY provided in environment or --api-key argument.", file=sys.stderr)
        print("To run with an external AI model, set GEMINI_API_KEY=... or pass --api-key <KEY>.\n", file=sys.stderr)
        return

    # Process each verse and replace in content
    # Process in reverse order so character offsets in content don't shift
    modified_content = content
    for v in sorted(selected_verses, key=lambda x: x["start"], reverse=True):
        num = v["number"]
        header = v["header"]
        
        # If already formatted, extract original lines from the code block if possible
        if v["already_converted"]:
            print(f"Verse {num} is already formatted. Re-processing...")
            cb_match = re.search(r"```tamil\n(.*?)\n```", v["body"], re.DOTALL)
            verse_lines = cb_match.group(1).strip() if cb_match else v["body"]
        else:
            verse_lines = v["body"]
            
        print(f"🤖 Processing Verse {num}: {header}...")
        prompt = build_prompt(book_context, header, verse_lines)
        
        ai_response = call_gemini_api(prompt, resolved_api_key, model=model)
        
        # Construct the replacement block
        replacement_block = f"\n{header}\n\n{ai_response.strip()}\n"
        
        # Replace the verse slice in modified_content
        modified_content = modified_content[:v["start"]] + replacement_block + modified_content[v["end"]:]
        print(f"  ✓ Verse {num} converted successfully.")
        
    # Write back to current file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(modified_content)
        
    print(f"\n✅ Completed! Current file '{filepath}' updated with converted verses.")

def main():
    parser = argparse.ArgumentParser(
        description="Tamil Verse AI Agent: Converts classical verses into structured 4-part literary format in-place."
    )
    parser.add_argument("filepath", help="Path to the Tamil literature markdown file to convert")
    parser.add_argument("--start", type=int, default=1, help="Start verse index (inclusive, default: 1)")
    parser.add_argument("--end", type=int, default=None, help="End verse index (inclusive, default: None)")
    parser.add_argument("--api-key", help="Gemini API key (or set via GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="AI model to use (default: gemini-2.5-flash)")
    
    args = parser.parse_args()
    process_file(args.filepath, start_idx=args.start, end_idx=args.end, api_key=args.api_key, model=args.model)

if __name__ == "__main__":
    main()
