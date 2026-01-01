"""
Tests for CourseSearchTool - validates search execution and result handling.
These tests help debug "query failed" by isolating VectorStore integration issues.
"""
import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager, CourseOutlineTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_with_valid_query_returns_formatted_results(
        self, mock_vector_store, mock_search_results_success
    ):
        """
        Test: Valid query with results returns properly formatted content.
        Debug scenario: Verify basic search functionality works.
        """
        mock_vector_store.search.return_value = mock_search_results_success
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="machine learning basics")

        # Verify VectorStore.search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning basics",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains expected content
        assert "[Introduction to ML - Lesson 1]" in result
        assert result != ""

    def test_execute_with_empty_results_returns_no_content_message(
        self, mock_vector_store, mock_search_results_empty
    ):
        """
        Test: Query with no matches returns informative message.
        Debug scenario: User asks about non-existent course content.
        """
        mock_vector_store.search.return_value = mock_search_results_empty
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum computing")

        assert "No relevant content found" in result

    def test_execute_with_course_name_filter_passes_filter_to_search(
        self, mock_vector_store, mock_search_results_success
    ):
        """
        Test: Course name filter is correctly passed to VectorStore.
        Debug scenario: Filtered searches not returning expected results.
        """
        mock_vector_store.search.return_value = mock_search_results_success
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", course_name="Introduction to ML")

        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name="Introduction to ML",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filter_passes_filter_to_search(
        self, mock_vector_store, mock_search_results_success
    ):
        """
        Test: Lesson number filter is correctly passed to VectorStore.
        Debug scenario: Lesson-specific queries not working.
        """
        mock_vector_store.search.return_value = mock_search_results_success
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="neural networks", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name=None,
            lesson_number=2
        )

    def test_execute_with_both_filters_passes_both_to_search(
        self, mock_vector_store, mock_search_results_success
    ):
        """
        Test: Both course and lesson filters work together.
        Debug scenario: Combined filters not working as expected.
        """
        mock_vector_store.search.return_value = mock_search_results_success
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", course_name="ML Course", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="basics",
            course_name="ML Course",
            lesson_number=1
        )

    def test_execute_when_vector_store_returns_error_returns_error_message(
        self, mock_vector_store, mock_search_results_error
    ):
        """
        Test: VectorStore errors are properly propagated.
        Debug scenario: ChromaDB connection issues causing "query failed".
        """
        mock_vector_store.search.return_value = mock_search_results_error
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        # The error message should be returned directly
        assert "Search error" in result or "ChromaDB" in result

    def test_execute_tracks_sources_for_ui(
        self, mock_vector_store, mock_search_results_success
    ):
        """
        Test: Sources are tracked for frontend display.
        Debug scenario: Sources not appearing in UI response.
        """
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="machine learning")

        # Check sources were tracked
        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["title"] is not None

    def test_execute_deduplicates_sources_from_same_lesson(
        self, mock_vector_store
    ):
        """
        Test: Multiple chunks from same lesson produce single source entry.
        Debug scenario: Duplicate sources appearing in UI.
        """
        # Two chunks from the same lesson
        duplicate_results = SearchResults(
            documents=["Chunk 1 from lesson 1", "Chunk 2 from lesson 1"],
            metadata=[
                {"course_title": "ML Course", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "ML Course", "lesson_number": 1, "chunk_index": 1}
            ],
            distances=[0.1, 0.15],
            error=None
        )
        mock_vector_store.search.return_value = duplicate_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/l1"
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="machine learning")

        # Should only have one source entry
        assert len(tool.last_sources) == 1


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()"""

    def test_get_tool_definition_returns_valid_anthropic_format(self, mock_vector_store):
        """
        Test: Tool definition matches Anthropic's expected format.
        Debug scenario: Claude not recognizing the tool.
        """
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestToolManager:
    """Tests for ToolManager coordination"""

    def test_register_tool_makes_tool_available(self, mock_vector_store):
        """Test: Registered tools can be executed by name."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_execute_tool_with_unknown_tool_returns_error(self):
        """
        Test: Unknown tool name returns error message.
        Debug scenario: Tool registration failed silently.
        """
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_from_search_tool(
        self, mock_vector_store, mock_search_results_success
    ):
        """Test: Sources are retrievable after search execution."""
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_link.return_value = "https://example.com"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources_clears_tracked_sources(
        self, mock_vector_store, mock_search_results_success
    ):
        """Test: reset_sources clears sources from all tools."""
        mock_vector_store.search.return_value = mock_search_results_success
        mock_vector_store.get_lesson_link.return_value = "https://example.com"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        sources = manager.get_last_sources()

        assert len(sources) == 0

    def test_get_tool_definitions_returns_all_registered_tools(self, mock_vector_store):
        """Test: All registered tools are returned in definitions."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names
