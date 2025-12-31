# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

**Always use `uv` to run commands; do not use pip directly.**

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The web interface is served at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Environment Setup

Requires `ANTHROPIC_API_KEY` in `.env` file at project root.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials.

### Component Flow

```
User Query → FastAPI (app.py) → RAGSystem → AIGenerator (Claude API)
                                    ↓
                            ToolManager executes CourseSearchTool
                                    ↓
                            VectorStore (ChromaDB) → Search Results
                                    ↓
                            Claude synthesizes final response
```

### Backend Modules (`backend/`)

- **`app.py`**: FastAPI server, mounts frontend static files, defines `/api/query` and `/api/courses` endpoints
- **`rag_system.py`**: Main orchestrator - coordinates document processing, vector search, and AI generation
- **`ai_generator.py`**: Handles Claude API calls with tool execution loop. Uses `claude-sonnet-4-20250514` model
- **`vector_store.py`**: ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks). Provides unified `search()` interface with course name resolution
- **`document_processor.py`**: Parses course documents (expected format: title, link, instructor on first lines, then `Lesson N:` markers). Chunks text with sentence-aware splitting
- **`search_tools.py`**: Implements Anthropic tool calling pattern. `CourseSearchTool` wraps vector store search; `ToolManager` handles tool registration and execution
- **`session_manager.py`**: In-memory conversation history per session
- **`config.py`**: Centralized settings (chunk size, embedding model, max results)
- **`models.py`**: Pydantic models for `Course`, `Lesson`, `CourseChunk`

### Frontend (`frontend/`)

Static HTML/CSS/JS served by FastAPI. Communicates with `/api/query` endpoint.

### Document Format

Course documents in `docs/` should follow:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
...
```

### Key Design Patterns

- **Tool-based RAG**: Claude decides when to search via tool calling rather than always retrieving
- **Semantic course matching**: Course names are resolved via vector similarity before content search
- **Chunk context injection**: First chunk of each lesson includes `"Lesson N content:"` prefix for better retrieval
