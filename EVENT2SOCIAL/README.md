# EVENT2SOCIAL

EVENT2SOCIAL is an AI-powered event content creation tool. It turns confirmed event details and uploaded materials into editable social media drafts and downloadable visual posts. It extracts text, creates dense semantic embeddings using the configured provider, stores vectors and source metadata in SQLite, and retrieves passages for grounded generation. Groq is the configured provider for text generation and media analysis; OpenAI remains supported for generation/search, and Gemini remains available as a separate RAG embeddings backend. It does not invent a mock response when an API request fails.

## Run on Windows

From PowerShell in this project directory:

```powershell
Set-Location 'C:\Users\praha\OneDrive\Documents\ChatGPT\EVENT2SOCIAL'
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Activate the project environment first so the Streamlit command uses EVENT2SOCIAL's .venv (Python 3.12), rather than a globally installed Streamlit. The server port and HTTPS certificate settings are read from .streamlit/config.toml.

## Local login and network use

On first open, use **Create Account / Sign Up**. Passwords require at least 12 characters and are stored as salted PBKDF2-HMAC-SHA256 hashes. Accounts have separate event details, uploads, embeddings, campaigns, drafts, and approvals. The existing unassigned workspace remains with the original local account. Signing out and back in resumes the same workspace.

The login and app traffic use HTTPS in this workspace. Streamlit is configured to listen on the LAN interface (`0.0.0.0`) and use the local certificate in `.streamlit/certs/`. On this computer, open `https://10.176.123.48:8509` (the LAN IP can change when the network changes). The certificate is trusted in the current Windows user profile. For another device, copy only `client-trust.cer` and import it into that device's trusted certificates; never copy `server-key.pem`. If the LAN IP changes, regenerate the certificate with the new IP in its subject alternative names and restart Streamlit. Do not expose this development server to the public internet. Browser speech input still requires microphone permission; manual text entry remains available.

The current project environment uses Python 3.12. To set up a new environment:

```powershell
Set-Location 'C:\Users\praha\OneDrive\Documents\ChatGPT\EVENT2SOCIAL'
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Groq generation and media configuration

Set the following values in .streamlit/secrets.toml on the server, then restart Streamlit:

    EVENT2SOCIAL_PROVIDER = "groq"
    GROQ_API_KEY = "replace-with-your-Groq-key"
    GROQ_MODEL = "openai/gpt-oss-120b"
    GROQ_MEDIA_MODEL = "qwen/qwen3.8-27b"
    GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"

Text generation uses Groq chat completions with the OpenAI-compatible endpoint at https://api.groq.com/openai/v1. Image and scanned-page analysis use Groq vision. Video analysis samples at most three frames locally with FFmpeg and sends those frames plus optional audio transcription to Groq Whisper; the original video is not uploaded as a video file. Video audio is divided into ten-minute segments to stay within the transcription upload size limit.

Groq currently does not expose an embeddings endpoint. To keep the existing dense-vector RAG workflow, configure EVENT2SOCIAL_EMBEDDING_PROVIDER as gemini or openai and keep that provider credential in server secrets. With no separate setting, the application reuses the existing Gemini embeddings configuration when available. Existing stored vectors and document history remain unchanged. Campaign generation works through Groq even if no embedding credential or current RAG index is available.

Do not commit .streamlit/secrets.toml or place real API keys in source code. The app reads credentials from Streamlit secrets or protected environment variables and never displays the key.

## OpenAI configuration

To use OpenAI for text generation or embeddings, an administrator sets EVENT2SOCIAL_PROVIDER=openai or EVENT2SOCIAL_EMBEDDING_PROVIDER=openai and configures OPENAI_API_KEY (or service-specific keys). Existing OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, and service-specific key variables remain supported. Groq remains the selected provider for vision and video audio transcription when EVENT2SOCIAL_PROVIDER=groq.

You can also define the provider and keys in `.streamlit/secrets.toml` using the same names, for example:

```toml
EVENT2SOCIAL_PROVIDER = "groq"
GROQ_API_KEY = "your-key"
```

Do not commit API keys. Provider requests send document text, search queries, and the generation brief to the selected external service. If keys are missing or requests fail, the app displays a configuration or connection error and does not generate substitute content.

Changing the embedding provider, endpoint, model, or Gemini embedding input format invalidates that account's saved index. Re-prepare documents before semantic search; Campaign Creation can fall back to confirmed event details when no current index is available. Each account has its own active vector index, and the app does not mix embeddings from different models.

## Demonstration flow

1. Enter and save your own event details.
2. Upload TXT, text-based PDF, or DOCX references; save and index them.
3. Choose a social channel, post format, tone, audience, objective, and campaign angle.
4. Campaign Creation automatically retrieves relevant passages when a valid index exists. If not, it can still generate from confirmed event details.
5. Edit and save the draft or download it as a text file. Reopen saved drafts from Draft History.

Documents and campaigns are stored locally in `data/event2social.db`; uploaded documents are kept under `data/uploads/`. PDF text extraction uses pypdf; scanned pages need the optional local OCR dependencies described below. External social publishing, scheduling, and live analytics are not implemented.

## Media and creative support

Knowledge Base accepts TXT, PDF, DOCX, JPG, JPEG, PNG, WEBP, MP4, and MOV uploads. Originals are kept locally and each account has isolated file records. The browser previews supported images and videos. Event Profile, Media Analysis, and Document RAG are stored as separate sources. Campaign creation combines confirmed profile fields and media analysis directly, then adds retrieved document evidence when available; missing or empty document text never blocks generation.

Image OCR and scanned-PDF OCR use local Tesseract through pytesseract; on Windows install the Tesseract application with winget install --id UB-Mannheim.TesseractOCR -e. The app checks PATH and common Windows install locations; set TESSERACT_CMD if installed elsewhere. Scanned PDF OCR also needs pypdfium2 to render pages. FFmpeg and FFprobe are required for Groq video analysis and local previews. After explicit consent, Groq vision analyzes resized images and up to three sampled PDF pages; OCR text and visual descriptions remain separate records.

MP4/MOV analysis uses FFprobe to inspect duration and audio tracks. FFmpeg samples at most three timestamped frames locally and extracts ten-minute audio segments; Groq vision describes the frames and Groq Whisper transcribes available audio with approximate timestamps. Only the sampled frames and audio segments are sent after explicit consent. The source video remains local and unchanged. A failed provider request is reported and persisted per source. The existing local six-second FFmpeg preview and timestamp ranking remain available.

Photo selection ranks uploaded originals using resolution, sharpness, exposure, duplicate similarity, and available Groq scene descriptions against the campaign objective. Users can replace the recommendation. Instagram posts support square and 4:5 JPEGs; LinkedIn and Facebook creatives use square JPEGs. Composition is local with Pillow and uses the original image. Multi-photo layouts and generated reel video files are not implemented.

The microphone uses the browser Web Speech API when supported. EVENT2SOCIAL does not send audio itself, but a browser may use its own speech service; the component discloses this before recording. HTTPS and microphone permission are required for network access. Text entry remains available.

The OCR dependencies in requirements.txt include the Python bindings and PDF renderer; Tesseract itself is an operating-system application and must be installed separately. Groq handles text generation and selected media analysis. RAG embeddings use the separately configured Gemini or OpenAI embedding backend because Groq currently provides no embeddings endpoint. No image-generation service is used for event creatives.

If Prepare documents for search is disabled, a saved file and server-configured embedding credential are required. Campaign generation does not require a Knowledge Base index; it uses confirmed event details and completed media analysis when retrieval is unavailable. Image-only PDFs can be visually analyzed after rendering when pypdfium2 is installed. Groq operations require an administrator-configured GROQ_API_KEY.
