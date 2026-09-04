import os
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from pypdf import PdfReader
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


if not FEATHERLESS_API_KEY:
    raise RuntimeError(
        "FEATHERLESS_API_KEY is missing from .env"
    )

if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY is missing from .env"
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="VerifyAI",
    description="AI Hallucination and Information Verification System"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FEATHERLESS AI
# ============================================================

ai_client = OpenAI(
    api_key=FEATHERLESS_API_KEY,
    base_url="https://api.featherless.ai/v1",
    timeout=60.0
)

MODEL = "Qwen/Qwen3-8B"


# ============================================================
# TAVILY
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# REQUEST MODEL
# ============================================================

class VerifyRequest(BaseModel):
    text: str


class CodeVerifyRequest(BaseModel):
    code: str
    language: str


class GoogleLoginRequest(BaseModel):
    credential: str


# ============================================================
# HOME
# ============================================================

@app.post("/auth/google")
def google_login(request: GoogleLoginRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID is missing from .env")

    try:
        info = id_token.verify_oauth2_token(
            request.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        return {
            "authenticated": True,
            "user": {
                "id": info.get("sub"),
                "name": info.get("name", "Google User"),
                "email": info.get("email", ""),
                "picture": info.get("picture", "")
            }
        }
    except Exception as error:
        print("GOOGLE LOGIN ERROR:", repr(error))
        raise HTTPException(status_code=401, detail="Invalid Google credential")


@app.get("/config")
def public_config():
    return {"google_client_id": GOOGLE_CLIENT_ID or ""}


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "VerifyAI backend is running"
    }


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(text):

    prompt = f"""
You are the factual claim extraction engine of VerifyAI.

Read the following AI-generated text.

Extract every independently verifiable factual claim.

Rules:

- Extract factual statements only.
- Ignore opinions.
- Ignore greetings.
- Ignore questions.
- Keep separate facts as separate claims.
- Do not explain anything.
- Do not use JSON.
- Do not use markdown.

Return one factual claim per line.

Example:

The capital of Australia is Sydney.
The Earth revolves around the Sun.

TEXT:

{text}
"""

    try:

        response = ai_client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You accurately extract factual claims. "
                        "Return only the claims."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,
            max_tokens=500,

            # IMPORTANT:
            # Qwen3 thinking is disabled so the model
            # returns the actual answer directly.
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

        message = response.choices[0].message

        ai_output = message.content or ""

        # Fallback for providers that expose reasoning separately
        if not ai_output:
            ai_output = getattr(
                message,
                "reasoning_content",
                ""
            ) or ""

        print(
            "\n========== FEATHERLESS CLAIM RESPONSE =========="
        )

        print(ai_output)

        print(
            "================================================\n"
        )

    except Exception as error:

        print(
            "\n========== CLAIM EXTRACTION AI ERROR =========="
        )

        print(repr(error))

        print(
            "================================================\n"
        )

        ai_output = ""


    # ========================================================
    # PARSE AI OUTPUT
    # ========================================================

    claims = []

    for line in ai_output.splitlines():

        line = line.strip()

        if not line:
            continue


        # Remove CLAIM:
        line = re.sub(
            r"^CLAIM\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE
        )


        # Remove numbering
        line = re.sub(
            r"^\d+[\.\)\:\-]\s*",
            "",
            line
        )


        # Remove bullets
        line = re.sub(
            r"^[-*•]\s*",
            "",
            line
        )


        line = line.strip()


        # Ignore headings
        if line.lower() in [
            "claims",
            "factual claims",
            "extracted claims",
            "factual claims:"
        ]:
            continue


        if len(line) >= 10:
            claims.append(line)


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_claims = []

    for claim in claims:

        if claim not in unique_claims:
            unique_claims.append(claim)


    # ========================================================
    # LOCAL FALLBACK
    # ========================================================

    if not unique_claims:

        print(
            "AI extraction returned no usable claims."
        )

        print(
            "Using local sentence extraction fallback..."
        )


        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip()
        )


        for sentence in sentences:

            sentence = sentence.strip()


            # Remove numbering
            sentence = re.sub(
                r"^\d+[\.\)\:\-]\s*",
                "",
                sentence
            )


            # Remove bullets
            sentence = re.sub(
                r"^[-*•]\s*",
                "",
                sentence
            )


            sentence = sentence.strip()


            if len(sentence) >= 10:

                if sentence.endswith("?"):
                    continue

                unique_claims.append(sentence)


    # ========================================================
    # FINAL FALLBACK
    # ========================================================

    if not unique_claims and text.strip():

        unique_claims.append(
            text.strip()
        )


    # Maximum 5 claims
    unique_claims = unique_claims[:5]


    print(
        "\n========== FINAL CLAIMS =========="
    )


    for index, claim in enumerate(
        unique_claims,
        1
    ):

        print(
            index,
            ":",
            claim
        )


    print(
        "==================================\n"
    )


    return unique_claims


# ============================================================
# SEARCH WEB EVIDENCE USING TAVILY
# ============================================================

def search_evidence(claim):

    try:

        print(
            "\n---------- TAVILY SEARCH ----------"
        )

        print(
            "Searching:",
            claim
        )


        result = tavily_client.search(
            query=claim,
            search_depth="basic",
            max_results=3
        )


        evidence = []


        for item in result.get(
            "results",
            []
        ):

            evidence.append({

                "title": item.get(
                    "title",
                    ""
                ),

                "url": item.get(
                    "url",
                    ""
                ),

                "content": item.get(
                    "content",
                    ""
                )
            })


        print(
            "Tavily sources found:",
            len(evidence)
        )


        print(
            "-----------------------------------\n"
        )


        return evidence


    except Exception as error:

        print(
            "\n========== TAVILY ERROR =========="
        )

        print(
            repr(error)
        )

        print(
            "==================================\n"
        )


        return []


# ============================================================
# VERIFY CLAIM USING FEATHERLESS
# ============================================================

def verify_claim(claim, evidence):
    """Verify one factual claim against Tavily evidence using Featherless."""

    evidence_parts = []
    for index, source in enumerate(evidence, 1):
        content = (source.get("content") or "").strip()[:2200]
        if content:
            evidence_parts.append(
                f"SOURCE {index}\n"
                f"TITLE: {source.get('title', '')}\n"
                f"URL: {source.get('url', '')}\n"
                f"CONTENT: {content}"
            )

    evidence_text = "\n--------------------\n".join(evidence_parts)
    if not evidence_text:
        evidence_text = "NO WEB EVIDENCE FOUND."

    prompt = f"""You are VerifyAI, a strict factual verification engine.

Verify ONE CLAIM using ONLY the supplied WEB EVIDENCE.

CLAIM:
{claim}

WEB EVIDENCE:
{evidence_text}

Choose exactly one status:
Supported
Partially Supported
Contradicted
Cannot Verify

Rules:
- Supported: the evidence clearly supports the claim.
- Partially Supported: only part of the claim is supported.
- Contradicted: the evidence clearly shows the claim is false.
- Cannot Verify: the supplied evidence is insufficient.
- Do not use outside knowledge.
- Confidence must be an integer from 0 to 100.
- Explanation must be one short sentence.
- Return ONLY the three lines below.
- Do not use JSON or markdown.

Required format:
STATUS: Supported
CONFIDENCE: 95
EXPLANATION: The evidence clearly supports the claim.
"""

    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a factual verification classifier. Follow the exact output format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        message = response.choices[0].message
        output = (message.content or "").strip()

        print("\n========== FEATHERLESS VERIFICATION ==========")
        print("Claim:", claim)
        print("AI output:", output)
        print("Finish reason:", getattr(response.choices[0], "finish_reason", "unknown"))
        print("===============================================\n")

        if not output:
            output = (getattr(message, "reasoning_content", "") or "").strip()

        if not output:
            raise RuntimeError("Featherless returned an empty verification response.")

        result = parse_verification(output)

        # Retry once if the model did not follow the requested format.
        if result["status"] == "Cannot Verify" and result["confidence"] == 0:
            retry = ai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only STATUS, CONFIDENCE, and EXPLANATION.",
                    },
                    {
                        "role": "user",
                        "content": f"""Classify this claim using ONLY this evidence.

CLAIM: {claim}

EVIDENCE:
{evidence_text}

Return exactly:
STATUS: Supported
CONFIDENCE: 95
EXPLANATION: short reason

Allowed STATUS values: Supported, Partially Supported, Contradicted, Cannot Verify.""",
                    },
                ],
                temperature=0.0,
                max_tokens=100,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            retry_output = (retry.choices[0].message.content or "").strip()
            print("VERIFICATION RETRY OUTPUT:", retry_output)
            result = parse_verification(retry_output)

        print("PARSED RESULT:", result)
        return result

    except Exception as error:
        print("\n========== VERIFICATION AI ERROR ==========")
        print(repr(error))
        print("Claim:", claim)
        print("============================================\n")

        try:
            recovery = ai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Classify the claim using only the evidence. Return exactly STATUS, CONFIDENCE, EXPLANATION."},
                    {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE: {evidence_text}\n\nAllowed status: Supported, Partially Supported, Contradicted, Cannot Verify."},
                ],
                temperature=0.0, max_tokens=100,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            recovery_output = (recovery.choices[0].message.content or "").strip()
            print("RECOVERY OUTPUT:", recovery_output)
            recovered = parse_verification(recovery_output)
            if not (recovered["status"] == "Cannot Verify" and recovered["confidence"] == 0):
                return recovered
        except Exception as recovery_error:
            print("RECOVERY ERROR:", repr(recovery_error))

        # Safe fallback only when the evidence directly contains most claim terms.
        claim_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", claim) if len(w) >= 4]
        evidence_lower = evidence_text.lower()
        matches = sum(1 for w in claim_words if w in evidence_lower)
        ratio = matches / max(len(claim_words), 1)
        if ratio >= 0.55:
            return {
                "status": "Supported",
                "confidence": 70,
                "explanation": "The available web evidence contains strong matching information for the claim."
            }

        return {
            "status": "Cannot Verify",
            "confidence": 0,
            "explanation": "The AI verification service returned an error and the available evidence was insufficient for a safe fallback.",
        }


# ============================================================
# PARSE VERIFICATION
# ============================================================

def parse_verification(text):

    result = {

        "status": "Cannot Verify",

        "confidence": 0,

        "explanation": (
            "The claim could not be verified."
        )
    }


    if not text:
        return result


    # ========================================================
    # STATUS
    # ========================================================

    status_match = re.search(

        r"STATUS\s*:\s*"
        r"(Supported|Partially Supported|"
        r"Contradicted|Cannot Verify)",

        text,

        re.IGNORECASE
    )


    if status_match:

        status = (
            status_match
            .group(1)
            .strip()
            .lower()
        )


        if status == "supported":

            result["status"] = "Supported"


        elif status == "partially supported":

            result["status"] = (
                "Partially Supported"
            )


        elif status == "contradicted":

            result["status"] = "Contradicted"


        else:

            result["status"] = "Cannot Verify"


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_match = re.search(

        r"CONFIDENCE\s*:\s*(\d{1,3})",

        text,

        re.IGNORECASE
    )


    if confidence_match:

        confidence = int(
            confidence_match.group(1)
        )


        result["confidence"] = max(

            0,

            min(
                100,
                confidence
            )
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation_match = re.search(

        r"EXPLANATION\s*:\s*(.+)",

        text,

        re.IGNORECASE | re.DOTALL
    )


    if explanation_match:

        explanation = (
            explanation_match
            .group(1)
            .strip()
        )


        if explanation:

            result["explanation"] = (
                explanation
            )


    return result


# ============================================================
# CALCULATE SCORE
# ============================================================

def calculate_score(claims):

    if not claims:
        return 0


    score_values = {

        "Supported": 100,

        "Partially Supported": 60,

        "Contradicted": 0,

        "Cannot Verify": 40
    }


    total = 0


    for claim in claims:

        total += score_values.get(

            claim["status"],

            40
        )


    return round(

        total / len(claims)
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(
    verified_claims
):

    supported = sum(

        1

        for item in verified_claims

        if item["status"] == "Supported"
    )


    partial = sum(

        1

        for item in verified_claims

        if item["status"]
        == "Partially Supported"
    )


    contradicted = sum(

        1

        for item in verified_claims

        if item["status"]
        == "Contradicted"
    )


    cannot_verify = sum(

        1

        for item in verified_claims

        if item["status"]
        == "Cannot Verify"
    )


    return (

        f"Analyzed {len(verified_claims)} "
        f"factual claim(s). "

        f"{supported} supported, "

        f"{partial} partially supported, "

        f"{contradicted} contradicted, "

        f"and {cannot_verify} "
        f"could not be verified."
    )




# ============================================================
# AI CODE VERIFICATION
# ============================================================

def verify_code(code, language):
    prompt = f"""You are VerifyAI's code verification engine.

Analyze the following {language} code. Check for syntax errors, logic/algorithm errors, likely runtime bugs, incorrect API/language usage, obvious security problems, and edge cases that would make the code unreliable. Do not invent problems.

Return ONLY this format:
SCORE: 0-100
STATUS: Correct | Issues Found | Cannot Analyze
SUMMARY: one short sentence
ISSUE: one issue, or NONE
ISSUE: one issue, or NONE
CORRECTED_CODE:
<corrected code>

If the code is already correct, use STATUS: Correct, ISSUE: NONE, and return the original code under CORRECTED_CODE.

LANGUAGE: {language}
CODE:
{code}
"""
    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict code review and correction engine. Follow the requested format exactly."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        message = response.choices[0].message
        output = message.content or getattr(message, "reasoning_content", "") or ""
        print("\n========== FEATHERLESS CODE REVIEW ==========")
        print(output)
        print("=============================================\n")
        return parse_code_review(output, code)
    except Exception as error:
        print("CODE VERIFICATION ERROR:", repr(error))
        return {
            "score": 0,
            "status": "Cannot Analyze",
            "summary": "The AI code verification service returned an error.",
            "issues": ["Unable to analyze the code because the AI service returned an error."],
            "corrected_code": code
        }


def parse_code_review(text, original_code):
    result = {
        "score": 0,
        "status": "Cannot Analyze",
        "summary": "The code could not be analyzed.",
        "issues": [],
        "corrected_code": original_code
    }
    if not text:
        return result

    m = re.search(r"SCORE\s*:\s*(\d{1,3})", text, re.I)
    if m:
        result["score"] = max(0, min(100, int(m.group(1))))

    m = re.search(r"STATUS\s*:\s*(Correct|Issues Found|Cannot Analyze)", text, re.I)
    if m:
        value = m.group(1).lower()
        result["status"] = {"correct":"Correct", "issues found":"Issues Found", "cannot analyze":"Cannot Analyze"}[value]

    m = re.search(r"SUMMARY\s*:\s*(.+)", text, re.I)
    if m:
        result["summary"] = m.group(1).strip()

    issues = re.findall(r"ISSUE\s*:\s*(.+)", text, re.I)
    result["issues"] = [x.strip() for x in issues if x.strip() and x.strip().upper() != "NONE"]

    m = re.search(r"CORRECTED_CODE\s*:\s*(.*)", text, re.I | re.S)
    if m and m.group(1).strip():
        corrected = m.group(1).strip()
        # Remove a surrounding markdown code fence if the model added one.
        corrected = re.sub(r"^```[a-zA-Z0-9+#.-]*\s*", "", corrected)
        corrected = re.sub(r"\s*```$", "", corrected)
        result["corrected_code"] = corrected.strip()

    if result["status"] == "Correct" and not result["issues"]:
        result["score"] = max(result["score"], 90)
    return result


@app.post("/verify-code")
def verify_code_endpoint(request: CodeVerifyRequest):
    code = request.code.strip()
    language = request.language.strip()
    if not code:
        return {"error": "Please paste some code."}
    if not language:
        return {"error": "Please select a programming language."}
    if len(code) > 12000:
        return {"error": "Please keep the code under 12,000 characters."}
    return verify_code(code, language)


# ============================================================
# VERIFY API
# ============================================================

@app.post("/verify")
def verify(
    request: VerifyRequest
):

    text = request.text.strip()


    if not text:

        return {

            "error":
            "Please paste AI-generated text."
        }


    try:

        print("\n")

        print(
            "#####################################################"
        )

        print(
            "#                 VERIFYAI START                   #"
        )

        print(
            "#####################################################"
        )


        # ====================================================
        # STEP 1 — CLAIM EXTRACTION
        # ====================================================

        print(
            "\n[1] Extracting factual claims..."
        )


        claims = extract_claims(
            text
        )


        print(
            f"Claims detected: {len(claims)}"
        )


        if not claims:

            return {

                "overall_score": 0,

                "summary":
                "No factual claims were detected.",

                "claims": []
            }


        verified_claims = []


        # ====================================================
        # STEP 2 — SEARCH + VERIFY
        # ====================================================

        for number, claim in enumerate(

            claims,

            start=1
        ):

            print(

                f"\n[2] Searching web evidence "
                f"for claim {number}..."
            )


            # ------------------------------------------------
            # TAVILY
            # ------------------------------------------------

            evidence = search_evidence(
                claim
            )


            print(
                f"Sources found: {len(evidence)}"
            )


            # ------------------------------------------------
            # FEATHERLESS
            # ------------------------------------------------

            print(

                f"[3] Verifying claim {number} "
                f"with Featherless..."
            )


            verification = verify_claim(

                claim,

                evidence
            )


            verified_claims.append({

                "claim":
                claim,

                "status":
                verification["status"],

                "confidence":
                verification["confidence"],

                "explanation":
                verification["explanation"],

                "evidence":
                evidence
            })


        # ====================================================
        # STEP 3 — SCORE
        # ====================================================

        overall_score = calculate_score(

            verified_claims
        )


        # ====================================================
        # STEP 4 — SUMMARY
        # ====================================================

        summary = generate_summary(

            verified_claims
        )


        print(

            "\n#####################################################"
        )

        print(

            "#                  VERIFYAI DONE                  #"
        )

        print(

            "#####################################################"
        )


        print(

            f"Overall Reliability Score: "
            f"{overall_score}/100"
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {

            "overall_score":
            overall_score,

            "summary":
            summary,

            "claims":
            verified_claims
        }


    except Exception as error:

        print(

            "\n#####################################################"
        )

        print(

            "#                  VERIFYAI ERROR                 #"
        )

        print(

            "#####################################################"
        )


        print(
            repr(error)
        )


        return {

            "error":
            str(error)
        }
# ============================================================
# MULTI-MODE VERIFICATION
# ============================================================

class ContentVerifyRequest(BaseModel):
    text: str
    mode: str = "information"

class URLVerifyRequest(BaseModel):
    url: str
    mode: str = "url"


def verify_text_content(text, mode="information"):
    """Run the existing evidence-grounded verification pipeline for a mode."""
    text = (text or "").strip()
    if not text:
        return {"error": "No content was provided."}
    if len(text) > 30000:
        text = text[:30000]

    claims = extract_claims(text)
    if not claims:
        return {"overall_score": 0, "summary": "No factual claims were detected.", "claims": [], "mode": mode}

    verified_claims = []
    for claim in claims:
        evidence = search_evidence(claim)
        verification = verify_claim(claim, evidence)
        verified_claims.append({
            "claim": claim,
            "status": verification["status"],
            "confidence": verification["confidence"],
            "explanation": verification["explanation"],
            "evidence": evidence
        })

    return {
        "overall_score": calculate_score(verified_claims),
        "summary": generate_summary(verified_claims),
        "claims": verified_claims,
        "mode": mode
    }


@app.post("/verify-content")
def verify_content(request: ContentVerifyRequest):
    return verify_text_content(request.text, request.mode)


@app.post("/verify-news-live")
def verify_news_live(request: dict):
    query = str(request.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a news topic, headline, or claim.")
    try:
        results = tavily_client.search(query, max_results=5, search_depth="advanced")
        evidence = []
        for item in results.get("results", []):
            evidence.append({
                "title": item.get("title", "Web Source"),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })
        claims = extract_claims(query)
        if not claims:
            claims = [query]
        verified = []
        for claim in claims:
            v = verify_claim(claim, evidence)
            verified.append({"claim": claim, "status": v["status"], "confidence": v["confidence"], "explanation": v["explanation"], "evidence": evidence})
        return {"overall_score": calculate_score(verified), "summary": generate_summary(verified), "claims": verified, "mode": "news", "live_search": True, "search_query": query}
    except HTTPException:
        raise
    except Exception as error:
        print("LIVE NEWS ERROR:", repr(error))
        raise HTTPException(status_code=500, detail="Live news verification failed. Check Tavily/API connectivity.")


@app.post("/verify-pdf")
async def verify_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF must be smaller than 15 MB.")
    try:
        import io
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages[:30]:
            extracted = page.extract_text(extraction_mode="layout") or ""
            pages.append(extracted)
        text = "\n\n".join(pages).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No readable text was found in this PDF.")

        # Guard against badly decoded PDF text. Keep readable Unicode text and
        # discard control characters that can confuse claim extraction.
        import unicodedata, re
        cleaned_lines = []
        for line in text.splitlines():
            line = "".join(ch for ch in line if unicodedata.category(ch)[0] != "C" or ch in "\t")
            line = re.sub(r"[\uFFFD]+", "", line).strip()
            if line:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No readable text was found in this PDF.")

        # If the PDF contains explicit Claim N lines (like reports/notes),
        # preserve those lines so the verifier receives the actual document claims.
        claim_lines = [line for line in cleaned_lines if re.match(r"^(?:claim|statement|fact)\s*\d*\s*[:.-]", line, re.I)]
        verification_text = "\n".join(claim_lines) if claim_lines else text
        result = verify_text_content(verification_text, "document")
        result["filename"] = filename
        result["pages_analyzed"] = min(len(reader.pages), 30)
        return result
    except HTTPException:
        raise
    except Exception as error:
        print("PDF VERIFICATION ERROR:", repr(error))
        raise HTTPException(status_code=400, detail="Could not read this PDF.")


@app.post("/verify-url")
def verify_url(request: URLVerifyRequest):
    raw = (request.url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Please enter a valid http:// or https:// URL.")
    try:
        import requests
        response = requests.get(raw, timeout=15, headers={"User-Agent": "VerifyAI/1.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else raw
        text = soup.get_text(" ", strip=True)
        if not text:
            raise HTTPException(status_code=400, detail="No readable text was found on this webpage.")
        result = verify_text_content(text[:30000], request.mode)
        result["source_url"] = raw
        result["page_title"] = title
        return result
    except HTTPException:
        raise
    except Exception as error:
        print("URL VERIFICATION ERROR:", repr(error))
        raise HTTPException(status_code=400, detail="Could not fetch that webpage.")
