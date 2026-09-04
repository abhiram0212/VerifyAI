# VerifyAI — Hackathon Final

**Don't just trust AI. Verify it.**

## Features
- Google sign-in with backend token verification
- Information/factual claim verification
- AI-generated code verification
- PDF/document verification
- Live news verification using web search
- Statistics/data claim verification
- Research text verification
- URL/webpage verification
- Evidence-backed claim status and confidence
- Reliability score + evidence breakdown
- Verification progress indicator
- Local history + analytics
- Demo content button
- Print/save verification report
- Responsive ChatGPT-style dashboard

## Run locally

### 1. Backend
Open PowerShell in `verifyai/backend`:

```powershell
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### 2. Environment
Create `verifyai/.env` (keep your existing keys if already present):

```env
FEATHERLESS_API_KEY=YOUR_EXISTING_KEY
TAVILY_API_KEY=YOUR_EXISTING_KEY
GOOGLE_CLIENT_ID=YOUR_EXISTING_GOOGLE_CLIENT_ID
```

**Never commit `.env` or API keys to GitHub.**

### 3. Frontend
Open `verifyai/frontend/index.html` with VS Code Live Server.
Use the same origin configured in Google Cloud, for example:
`http://127.0.0.1:5500`

## Fast demo
1. Sign in with Google.
2. Click **Load Demo**.
3. Click **Verify Content**.
4. Show the 67/100-style reliability breakdown and evidence.
5. Switch to **News** and enter a current headline/topic.
6. Switch to **Code** and paste AI-generated code.
7. Show **PDF**, **URL**, and **History** briefly.

## Important
The backend reads `GOOGLE_CLIENT_ID` from `.env`, so the frontend no longer needs the client ID hardcoded.
