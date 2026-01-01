"""
Shared pytest fixtures for RAG chatbot tests.
"""
import pytest
from unittest.mock import Mock
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_search_results_success():
    """Fixture: Successful search results with content"""
    return SearchResults(
        documents=[
            "This is lesson 1 content about machine learning basics.",
            "This is lesson 2 content about neural networks."
        ],
        metadata=[
            {"course_title": "Introduction to ML", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to ML", "lesson_number": 2, "chunk_index": 0}
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def mock_search_results_empty():
    """Fixture: Empty search results (no matches)"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def mock_search_results_error():
    """Fixture: Search results with error"""
    return SearchResults.empty("Search error: ChromaDB connection failed")


@pytest.fixture
def mock_course():
    """Fixture: Sample course object"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="ML Basics", lesson_link="https://example.com/ml-1"),
            Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/ml-2")
        ]
    )


@pytest.fixture
def mock_course_chunks(mock_course):
    """Fixture: Sample course chunks"""
    return [
        CourseChunk(
            content="Lesson 1 content: Machine learning is a subset of AI.",
            course_title=mock_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Neural networks are inspired by biological neurons.",
            course_title=mock_course.title,
            lesson_number=2,
            chunk_index=1
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Fixture: Mocked VectorStore"""
    store = Mock()
    store.search = Mock()
    store.get_lesson_link = Mock(return_value="https://example.com/lesson-1")
    store._resolve_course_name = Mock(return_value="Introduction to ML")
    store.max_results = 5
    return store


@pytest.fixture
def mock_anthropic_response_text_only():
    """Fixture: Claude response without tool use"""
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Here is the answer to your question."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_response_with_tool_use():
    """Fixture: Claude response with tool_use"""
    response = Mock()
    response.stop_reason = "tool_use"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Let me search for that."

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning basics"}

    response.content = [text_block, tool_block]
    return response


@pytest.fixture
def mock_anthropic_final_response():
    """Fixture: Final Claude response after tool execution"""
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Based on the course content, machine learning is..."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_config():
    """Fixture: Mock configuration"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


# Sequential tool calling fixtures

@pytest.fixture
def mock_anthropic_response_with_outline_tool():
    """Fixture: Claude response requesting get_course_outline (first tool in sequence)"""
    response = Mock()
    response.stop_reason = "tool_use"
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "outline_1"
    tool_block.name = "get_course_outline"
    tool_block.input = {"course_name": "MCP"}
    response.content = [tool_block]
    return response


@pytest.fixture
def mock_anthropic_response_with_search_after_outline():
    """Fixture: Claude response after seeing outline, now searching (second tool in sequence)"""
    response = Mock()
    response.stop_reason = "tool_use"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Now let me search for more details about that topic."
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "search_1"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "tool integration patterns"}
    response.content = [text_block, tool_block]
    return response


@pytest.fixture
def mock_anthropic_response_two_tools_parallel():
    """Fixture: Claude response with two tools in one response (parallel execution)"""
    response = Mock()
    response.stop_reason = "tool_use"
    tool1 = Mock()
    tool1.type = "tool_use"
    tool1.id = "tool_1"
    tool1.name = "get_course_outline"
    tool1.input = {"course_name": "Course A"}
    tool2 = Mock()
    tool2.type = "tool_use"
    tool2.id = "tool_2"
    tool2.name = "get_course_outline"
    tool2.input = {"course_name": "Course B"}
    response.content = [tool1, tool2]
    return response


@pytest.fixture
def mock_anthropic_response_tool_use_only():
    """Fixture: Claude response with only tool_use, no text (for max iterations test)"""
    response = Mock()
    response.stop_reason = "tool_use"
    tool_block = Mock(spec=['type', 'id', 'name', 'input'])  # Explicitly limit attributes
    tool_block.type = "tool_use"
    tool_block.id = "tool_infinite"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "keep searching"}
    response.content = [tool_block]
    return response
