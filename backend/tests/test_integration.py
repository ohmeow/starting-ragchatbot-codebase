"""
Integration tests for RAG chatbot using real components.
These tests use actual ChromaDB and embeddings to catch issues mocks might miss.
"""
import pytest
from unittest.mock import Mock, patch
import tempfile
import shutil
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager, CourseOutlineTool
from models import Course, Lesson, CourseChunk


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real ChromaDB"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def real_vector_store(self, temp_chroma_path):
        """Create a real VectorStore with temporary storage"""
        return VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    @pytest.fixture
    def sample_course(self):
        """Create a sample course for testing"""
        return Course(
            title="Introduction to Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Test",
            lessons=[
                Lesson(lesson_number=1, title="ML Basics", lesson_link="https://example.com/ml-1"),
                Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/ml-2"),
            ]
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample course chunks"""
        return [
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Neural networks are computing systems inspired by biological neural networks in the brain.",
                course_title=sample_course.title,
                lesson_number=2,
                chunk_index=1
            ),
            CourseChunk(
                content="Deep learning uses multiple layers of neural networks to process complex patterns.",
                course_title=sample_course.title,
                lesson_number=2,
                chunk_index=2
            ),
        ]

    def test_vector_store_initialization(self, real_vector_store):
        """
        Test: VectorStore initializes successfully with real ChromaDB.
        Debug scenario: ChromaDB configuration or embedding model issues.
        """
        assert real_vector_store is not None
        assert real_vector_store.course_catalog is not None
        assert real_vector_store.course_content is not None

    def test_add_and_search_course_content(self, real_vector_store, sample_course, sample_chunks):
        """
        Test: Can add course content and search it.
        Debug scenario: Content not being indexed or retrieved correctly.
        """
        # Add course metadata and content
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_chunks)

        # Search for content
        results = real_vector_store.search(query="machine learning artificial intelligence")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert "machine learning" in results.documents[0].lower() or "artificial intelligence" in results.documents[0].lower()

    def test_search_with_course_filter(self, real_vector_store, sample_course, sample_chunks):
        """
        Test: Course name filter works with semantic matching.
        Debug scenario: Course filtering not working.
        """
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_chunks)

        # Search with course filter (partial name)
        results = real_vector_store.search(
            query="neural networks",
            course_name="Machine Learning"  # Partial match
        )

        assert not results.is_empty()
        # All results should be from the ML course
        for meta in results.metadata:
            assert meta["course_title"] == sample_course.title

    def test_search_with_lesson_filter(self, real_vector_store, sample_course, sample_chunks):
        """
        Test: Lesson number filter works correctly.
        Debug scenario: Lesson filtering returning wrong results.
        """
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_chunks)

        # Search only in lesson 2
        results = real_vector_store.search(
            query="neural networks",
            lesson_number=2
        )

        assert not results.is_empty()
        # All results should be from lesson 2
        for meta in results.metadata:
            assert meta["lesson_number"] == 2

    def test_search_nonexistent_course_returns_error(self, real_vector_store, sample_course, sample_chunks):
        """
        Test: Searching for non-existent course returns error due to distance threshold.
        Debug scenario: Missing course handling with semantic distance check.
        """
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_chunks)

        # Search for non-existent course - should fail distance threshold check
        results = real_vector_store.search(
            query="test query",
            course_name="Quantum Physics Advanced Theory"  # Very different from ML
        )

        # Should return error about no matching course (distance too high)
        assert results.error is not None and "No course found" in results.error

    def test_search_empty_database_returns_empty(self, real_vector_store):
        """
        Test: Searching empty database returns empty results.
        Debug scenario: Empty vector store behavior.
        """
        results = real_vector_store.search(query="anything")

        assert results.is_empty()

    def test_course_name_resolution(self, real_vector_store, sample_course):
        """
        Test: Course name resolution uses semantic matching.
        Debug scenario: Partial course names not matching.
        """
        real_vector_store.add_course_metadata(sample_course)

        # Test various partial matches
        resolved = real_vector_store._resolve_course_name("ML")
        assert resolved == sample_course.title or resolved is None  # Depends on embedding similarity

        resolved = real_vector_store._resolve_course_name("Machine Learning")
        assert resolved == sample_course.title

        resolved = real_vector_store._resolve_course_name("Introduction")
        assert resolved == sample_course.title


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with real VectorStore"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def populated_vector_store(self, temp_chroma_path):
        """Create a VectorStore with test data"""
        store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        course = Course(
            title="Deep Learning Fundamentals",
            course_link="https://example.com/dl",
            instructor="Dr. Neural",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to Deep Learning", lesson_link="https://example.com/dl-1"),
            ]
        )

        chunks = [
            CourseChunk(
                content="Deep learning is a branch of machine learning that uses neural networks with many layers.",
                course_title=course.title,
                lesson_number=1,
                chunk_index=0
            ),
        ]

        store.add_course_metadata(course)
        store.add_course_content(chunks)

        return store

    def test_course_search_tool_with_real_store(self, populated_vector_store):
        """
        Test: CourseSearchTool works with real VectorStore.
        Debug scenario: Tool integration with actual vector store.
        """
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="deep learning neural networks")

        assert "[Deep Learning Fundamentals" in result
        assert "neural networks" in result.lower() or "deep learning" in result.lower()

    def test_course_search_tool_tracks_sources(self, populated_vector_store):
        """
        Test: Sources are correctly tracked with real data.
        Debug scenario: Source tracking with actual search results.
        """
        tool = CourseSearchTool(populated_vector_store)

        tool.execute(query="deep learning")

        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["title"] is not None

    def test_tool_manager_with_real_tools(self, populated_vector_store):
        """
        Test: ToolManager correctly orchestrates real tools.
        Debug scenario: Tool registration and execution flow.
        """
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        outline_tool = CourseOutlineTool(populated_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Execute search
        result = manager.execute_tool("search_course_content", query="deep learning")
        assert "Deep Learning" in result

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) > 0


class TestRAGSystemIntegration:
    """Integration tests for full RAG system with real VectorStore, mocked API"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def integration_config(self, temp_chroma_path):
        """Create config for integration testing"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = temp_chroma_path
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.MAX_HISTORY = 2
        return config

    @patch('rag_system.AIGenerator')
    def test_rag_query_with_real_vector_store(self, mock_ai_class, integration_config):
        """
        Test: RAG query flow with real VectorStore, mocked API.
        Debug scenario: Full query flow with actual vector operations.
        """
        # Setup mock AI that uses tools
        mock_ai = Mock()

        # Create response that triggers tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Let me search."
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "machine learning"}
        tool_response.content = [text_block, tool_block]

        # Final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_text = Mock()
        final_text.type = "text"
        final_text.text = "Based on the course content, machine learning is a subset of AI."
        final_response.content = [final_text]

        # For this test, we'll just return the final response directly
        # since we're not testing the full tool execution flow here
        mock_ai.generate_response.return_value = "Machine learning is a subset of AI that learns from data."
        mock_ai_class.return_value = mock_ai

        from rag_system import RAGSystem
        rag = RAGSystem(integration_config)

        # Add test data
        course = Course(
            title="AI Fundamentals",
            course_link="https://example.com/ai",
            instructor="Dr. AI",
            lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/ai-1")]
        )
        chunks = [
            CourseChunk(
                content="Machine learning is a subset of AI that enables learning from data.",
                course_title=course.title,
                lesson_number=1,
                chunk_index=0
            )
        ]
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Execute query
        response, sources = rag.query("What is machine learning?")

        assert response is not None
        assert "machine learning" in response.lower() or "ai" in response.lower()

    @patch('rag_system.AIGenerator')
    def test_rag_analytics_with_real_vector_store(self, mock_ai_class, integration_config):
        """
        Test: Analytics work with real VectorStore.
        Debug scenario: Course counting and listing.
        """
        mock_ai_class.return_value = Mock()

        from rag_system import RAGSystem
        rag = RAGSystem(integration_config)

        # Add test courses
        for i in range(3):
            course = Course(
                title=f"Test Course {i}",
                course_link=f"https://example.com/course-{i}",
                instructor=f"Dr. Test {i}",
                lessons=[]
            )
            rag.vector_store.add_course_metadata(course)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3


class TestErrorScenarios:
    """Tests for error scenarios that might cause 'query failed'"""

    @pytest.fixture
    def temp_chroma_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_search_results_error_field_propagation(self):
        """
        Test: SearchResults error field is properly set and returned.
        Debug scenario: Error messages not reaching the user.
        """
        error_results = SearchResults.empty("Custom error message")

        assert error_results.error == "Custom error message"
        assert error_results.is_empty()

    def test_tool_execution_returns_error_string(self, temp_chroma_path):
        """
        Test: Tool returns error string when search fails.
        Debug scenario: Tool errors not formatted correctly.
        """
        store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        tool = CourseSearchTool(store)

        # Search empty store with course filter should return error
        result = tool.execute(query="test", course_name="Nonexistent")

        # Should get an informative message, not crash
        assert "No" in result or "not found" in result.lower() or "error" in result.lower()
