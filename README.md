# InsightIQ

Ask questions about long videos and PDFs and get answers that point you to the exact moment or page.

InsightIQ transcribes a YouTube video (or a local video file), or reads a PDF, and indexes it for semantic search. Gemini then answers your questions using only that content. Video answers link to the matching timestamp on YouTube; PDF answers list the source pages.

## Use cases

- **Lectures and courses:** "Where does the instructor explain backpropagation?" takes you to the timestamp instead of scrubbing through a 2-hour video.
- **Meetings and talks:** summarise the key decisions, or find what was said about a given topic.
- **Research papers, manuals, reports:** ask a PDF a question and get an answer with page references.
- **Study and revision:** get the key points of a video or document, then jump to the parts you need.

## How it works

| Step | Video | PDF |
|---|---|---|
| 1. Input | YouTube URL or local file (`yt-dlp`) | Upload a PDF |
| 2. Extract | Audio extraction (FFmpeg) | Text, tables and images (PyMuPDF) |
| 3. Understand | Transcription (WhisperX) | Image descriptions (Gemini) |
| 4. Index | Semantic chunking + embeddings (`BAAI/bge-small-en-v1.5`) → ChromaDB | Same |
| 5. Answer | MMR retrieval → Gemini, with timestamps | Hybrid BM25 + vector retrieval → Gemini, with page numbers |

The UI is a Streamlit dashboard with Video and PDF modes, a processing page, Q&A, and a chat history that keeps your 10 most recent queries.

## Requirements

- **Python 3.11 or newer** (developed on 3.13)
- **FFmpeg 4–7**, with shared libraries (needed by `torchcodec`; FFmpeg 8 is not supported)
- A **Gemini API key**, free from [Google AI Studio](https://aistudio.google.com/app/apikey)
- About 3 GB of disk space for dependencies. Everything runs on CPU; no GPU is needed.

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Nikhil28git/insight_IQ.git
cd insight_IQ

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install FFmpeg

**macOS (Homebrew):**

```bash
brew install ffmpeg@7
```

Homebrew doesn't link `ffmpeg@7` system-wide, so tell your venv where to find its libraries (run this once):

```bash
echo 'export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"' >> .venv/bin/activate
source .venv/bin/activate
```

**Ubuntu / Debian:**

```bash
sudo apt install ffmpeg
```

**Windows:**

```powershell
winget install Gyan.FFmpeg.Shared
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Linux without a GPU:** install the CPU build of PyTorch first to avoid downloading large CUDA packages:
> `pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu`

### 4. Add your API key

Create a file named `.env` in the project folder:

```env
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: choose a different Gemini model (default: gemini-3.8-flash)
# GEMINI_MODEL=gemini-3.1-flash-lite
```

`.env` is listed in `.gitignore`. Never commit it or share your key.

### 5. Build the search index

The search database (`chroma_db/`) isn't stored in the repository. Rebuild it from the included chunk files (this runs locally and makes no API calls):

```bash
python rebuild_index.py
```

You only need to do this once, or again if you delete `chroma_db/`.

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501 in your browser.

- **Process Video:** paste a YouTube URL (or a local file path), give it an ID, and start processing. A 10-minute video takes about 3 minutes on CPU.
- **Process PDF:** switch to PDF mode in the sidebar, upload a PDF and give it an ID.
- **Ask AI / Ask PDF:** pick an indexed item and ask a question.
- **Chat History:** your 10 most recent questions and answers, kept across restarts.

Processing can also be run from the command line:

```bash
python run_pipeline.py --source "https://youtu.be/VIDEO_ID" --video_id my_video --index_only
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar shows "Set GEMINI_API_KEY" | Check that `.env` exists in the project folder and contains your key, then restart Streamlit. |
| `429 RESOURCE_EXHAUSTED` | You hit the Gemini free-tier daily limit. Wait for it to reset, set a different `GEMINI_MODEL` in `.env`, or enable billing. |
| `404 NOT_FOUND` for a model | That model has been retired. Set `GEMINI_MODEL` in `.env` to a current one. |
| Answers ignore the video content / log says `Retrieved 0 chunks` | The search index is empty. Run `python rebuild_index.py`. |
| YouTube: "Sign in to confirm you're not a bot" | On the Process Video page, pick your browser under "YouTube bot fix" (you must be logged in to YouTube in that browser). |
| `Could not load libtorchcodec` | FFmpeg shared libraries weren't found. Install FFmpeg 4–7 (see step 2) and re-activate the venv. |

## Project structure

```
app.py                  Streamlit dashboard
ui/components.py        Reusable UI components (cards, badges, icons)
styles/theme.css        Dashboard styling
.streamlit/config.toml  Streamlit theme

run_pipeline.py         Video pipeline: audio → transcript → chunks → index
extract_audio.py        Phase 1: download and extract audio
transcribe.py           Phase 2: WhisperX transcription
chunk_transcript.py     Phase 3: semantic chunking
embed_and_store.py      Phase 4: store chunks in ChromaDB
rag_query.py            Video question answering

process_pdf.py          PDF pipeline: extract → chunk → embed → index
chunk_pdf.py, pdf_utils.py, image_utils.py, image_description.py
rag_query_pdf.py        PDF question answering

rebuild_index.py        Rebuild chroma_db/ from the saved chunks
videos/, pdfs/          Processed data (transcripts, chunks, embeddings)
```
