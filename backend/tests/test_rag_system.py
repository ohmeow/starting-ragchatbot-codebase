"""
Tests for RAGSystem - validates the orchestration of all components.
These tests help debug "query failed" by testing the full query flow.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_returns_response_and_sources(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Query returns tuple of (response, sources).
        Debug scenario: Verify basic query flow works.
        """
        # Setup mocks
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Here is the answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Mock the tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(return_value=[
            {"title": "ML Course - Lesson 1", "link": "https://example.com"}
        ])
        rag.tool_manager.reset_sources = Mock()

        response, sources = rag.query("What is machine learning?")

        assert response == "Here is the answer"
        assert len(sources) == 1
        assert sources[0]["title"] == "ML Course - Lesson 1"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_session_id_retrieves_history(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Session ID triggers conversation history retrieval.
        Debug scenario: Context not being maintained between queries.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Answer with context"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = "User: Previous question"
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Follow up question", session_id="session_1")

        mock_session.get_conversation_history.assert_called_with("session_1")
        # Verify history was passed to AI generator
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: Previous question"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session_id_skips_history(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: No session ID means no history retrieval.
        Debug scenario: First query in conversation.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("First question")

        # Should not try to get history
        mock_session.get_conversation_history.assert_not_called()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_updates_session_after_response(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Session history is updated after successful query.
        Debug scenario: Conversation history not persisting.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "The answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Question", session_id="session_1")

        mock_session.add_exchange.assert_called_once()
        call_args = mock_session.add_exchange.call_args[0]
        assert call_args[0] == "session_1"
        assert "Question" in call_args[1]  # Original query is in the prompt
        assert call_args[2] == "The answer"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_resets_sources_after_retrieval(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Sources are reset after being retrieved.
        Debug scenario: Stale sources appearing in subsequent queries.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        mock_reset = Mock()
        rag.tool_manager.reset_sources = mock_reset
        rag.tool_manager.get_last_sources = Mock(return_value=[])

        rag.query("Question")

        mock_reset.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_ai_generator_error_propagates(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: AIGenerator errors propagate with user-friendly messages.
        Debug scenario: Identify where errors originate.
        """
        from ai_generator import AIGeneratorError

        mock_ai = Mock()
        mock_ai.generate_response.side_effect = AIGeneratorError("API rate limit exceeded")
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        with pytest.raises(AIGeneratorError) as exc_info:
            rag.query("Question")

        assert "rate limit" in str(exc_info.value).lower()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_passes_tools_to_ai_generator(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Tool definitions and manager are passed to AIGenerator.
        Debug scenario: Tools not available during generation.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("Question")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_formats_prompt_correctly(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """
        Test: Query is formatted into proper prompt.
        Debug scenario: Query not reaching Claude correctly.
        """
        mock_ai = Mock()
        mock_ai.generate_response.return_value = "Answer"
        mock_ai_class.return_value = mock_ai

        mock_session = Mock()
        mock_session.get_conversation_history.return_value = None
        mock_session_class.return_value = mock_session

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()

        rag.query("What is machine learning?")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "What is machine learning?" in call_kwargs["query"]


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_initialization_creates_all_components(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """Test: All components are initialized correctly."""
        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Verify all components were created
        mock_doc_class.assert_called_once()
        mock_vs_class.assert_called_once()
        mock_ai_class.assert_called_once()
        mock_session_class.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_initialization_registers_tools(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """Test: Search and outline tools are registered."""
        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Verify tools are registered
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools


class TestRAGSystemAnalytics:
    """Tests for RAGSystem analytics"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics_returns_stats(
        self, mock_session_class, mock_doc_class, mock_ai_class, mock_vs_class, mock_config
    ):
        """Test: Analytics returns course count and titles."""
        mock_vs = Mock()
        mock_vs.get_course_count.return_value = 3
        mock_vs.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]
        mock_vs_class.return_value = mock_vs

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
