"""
Tests for AIGenerator - validates Claude API interaction and tool handling.
These tests help debug "query failed" by isolating API and tool execution issues.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator, AIGeneratorError


class TestAIGeneratorGenerateResponse:
    """Tests for AIGenerator.generate_response()"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools_returns_text(
        self, mock_anthropic_class, mock_anthropic_response_text_only
    ):
        """
        Test: Simple query without tools returns text response.
        Debug scenario: Basic Claude API integration.
        """
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text_only
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        result = generator.generate_response(query="What is Python?")

        assert result == "Here is the answer to your question."
        mock_client.messages.create.assert_called_once()

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_includes_tool_definitions(
        self, mock_anthropic_class, mock_anthropic_response_text_only
    ):
        """
        Test: Tools are included in API request when provided.
        Debug scenario: Tool definitions not reaching Claude API.
        """
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text_only
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "test_tool", "description": "A test", "input_schema": {"type": "object"}}]

        generator.generate_response(query="Test", tools=tools, tool_manager=Mock())

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use_executes_tool(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool is executed when Claude returns tool_use.
        Debug scenario: Tool execution not happening.
        """
        mock_client = Mock()
        # First call returns tool_use, second returns final response
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        result = generator.generate_response(
            query="Search for ML basics",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )

        # Verify final response is returned
        assert "machine learning is" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history_includes_context(
        self, mock_anthropic_class, mock_anthropic_response_text_only
    ):
        """
        Test: Conversation history is included in system prompt.
        Debug scenario: Context not being maintained.
        """
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text_only
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        history = "User: What is ML?\nAssistant: ML is machine learning."

        generator.generate_response(query="Tell me more", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_api_error_raises_ai_generator_error(self, mock_anthropic_class):
        """
        Test: API errors are wrapped in AIGeneratorError with user-friendly message.
        Debug scenario: Identify API failures causing 500 errors.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Now wraps in AIGeneratorError with friendly message
        with pytest.raises(AIGeneratorError) as exc_info:
            generator.generate_response(query="Test")

        assert "Unexpected error" in str(exc_info.value) or "API" in str(exc_info.value)

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_network_error_raises_ai_generator_error(self, mock_anthropic_class):
        """
        Test: Network errors are wrapped in AIGeneratorError with user-friendly message.
        Debug scenario: Network connectivity issues.
        """
        mock_client = Mock()
        # Simulate a network/connection error
        mock_client.messages.create.side_effect = ConnectionError("Connection refused")
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        with pytest.raises(AIGeneratorError) as exc_info:
            generator.generate_response(query="Test")

        # Should wrap the error with user-friendly message
        assert "Unexpected error" in str(exc_info.value) or "Claude API" in str(exc_info.value)


class TestAIGeneratorToolExecution:
    """Tests for AIGenerator._handle_tool_execution()"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_formats_tool_results_correctly(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool results are formatted as expected by Anthropic API.
        Debug scenario: Tool results not being processed by Claude.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}],
            tool_manager=mock_tool_manager
        )

        # Check the second API call includes tool results
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_with_tool_error_includes_error_in_result(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool execution errors are passed back to Claude.
        Debug scenario: Tool failures not being communicated properly.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search error: Database unavailable"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}],
            tool_manager=mock_tool_manager
        )

        # The error message should be in the tool result sent to Claude
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result_content = second_call[1]["messages"][2]["content"][0]["content"]
        assert "Search error" in tool_result_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_api_error_on_second_call_raises(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use
    ):
        """
        Test: API error during tool result processing is wrapped in AIGeneratorError.
        Debug scenario: Second API call fails after tool execution.
        """
        mock_client = Mock()
        # First call succeeds, second fails
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            Exception("API connection error")
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        with pytest.raises(AIGeneratorError) as exc_info:
            generator.generate_response(
                query="Test",
                tools=[{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}],
                tool_manager=mock_tool_manager
            )

        assert "API" in str(exc_info.value) or "error" in str(exc_info.value).lower()


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator initialization and configuration"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generator_uses_correct_model(self, mock_anthropic_class, mock_anthropic_response_text_only):
        """Test: Configured model is used in API requests."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text_only
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generator_uses_temperature_zero(self, mock_anthropic_class, mock_anthropic_response_text_only):
        """Test: Temperature is set to 0 for deterministic responses."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response_text_only
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0


class TestSequentialToolCalling:
    """Tests for sequential tool calling (up to 2 rounds)"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calls_executes_two_tools(
        self, mock_anthropic_class,
        mock_anthropic_response_with_outline_tool,
        mock_anthropic_response_with_search_after_outline,
        mock_anthropic_final_response
    ):
        """
        Test: Sequential tool calls - outline then search.
        Scenario: Complex query requires two tool calls in separate rounds.
        """
        mock_client = Mock()
        # Round 1: get_course_outline, Round 2: search_course_content, Final: text response
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_outline_tool,
            mock_anthropic_response_with_search_after_outline,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: MCP\nLesson 1: Introduction\nLesson 2: Tool Integration",
            "Tool integration patterns include..."
        ]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "get_course_outline", "description": "Get outline", "input_schema": {"type": "object"}}]

        result = generator.generate_response(
            query="Find topics similar to lesson 2 of MCP course",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        # Verify 3 API calls: initial + after round 1 + after round 2
        assert mock_client.messages.create.call_count == 3
        assert "machine learning" in result

    @patch('ai_generator.anthropic.Anthropic')
    def test_parallel_tools_in_single_response_executes_all(
        self, mock_anthropic_class,
        mock_anthropic_response_two_tools_parallel,
        mock_anthropic_final_response
    ):
        """
        Test: Multiple tools in one response are all executed.
        Scenario: Claude requests two tools simultaneously.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_two_tools_parallel,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course A outline...",
            "Course B outline..."
        ]

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "get_course_outline", "description": "Get outline", "input_schema": {"type": "object"}}]

        generator.generate_response(
            query="Compare Course A and Course B",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify both tools in the same round were executed
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_failure_allows_graceful_response(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool execution exception is caught and passed to Claude.
        Scenario: Tool raises exception, Claude should still respond gracefully.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        # Should NOT raise - error is caught and passed to Claude
        result = generator.generate_response(
            query="Search for ML",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify the tool result contains the error message
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result_content = second_call[1]["messages"][2]["content"][0]
        assert "Tool execution failed" in tool_result_content["content"]
        assert tool_result_content["is_error"] is True

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_iterations_terminates_at_two(
        self, mock_anthropic_class,
        mock_anthropic_response_tool_use_only,
        mock_anthropic_final_response
    ):
        """
        Test: Loop terminates after 2 iterations even if Claude keeps requesting tools.
        Scenario: Claude always requests more tools, but we cap at 2 rounds.
        """
        mock_client = Mock()
        # All responses request more tools, but should stop after 2 rounds
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use_only,  # Initial response
            mock_anthropic_response_tool_use_only,  # After round 1
            mock_anthropic_response_tool_use_only,  # After round 2 (should hit limit)
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results..."

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        result = generator.generate_response(
            query="Keep searching",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify loop terminated after max_iterations (2)
        # Initial call + 2 rounds = 3 API calls
        assert mock_client.messages.create.call_count == 3
        # Should get fallback message since no text in final response
        assert "maximum" in result.lower() or "search" in result.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_error_string_passed_to_claude(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool returning error string is passed to Claude for synthesis.
        Scenario: Tool returns error message, Claude acknowledges limitation.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "No course found matching 'nonexistent'"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        generator.generate_response(
            query="Search for nonexistent course",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify error string was passed in tool result
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result_content = second_call[1]["messages"][2]["content"][0]["content"]
        assert "No course found" in tool_result_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_in_second_round_raises(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use
    ):
        """
        Test: API error during second API call is wrapped in AIGeneratorError.
        Scenario: First tool round succeeds, second API call fails.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            Exception("API connection error")
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        with pytest.raises(AIGeneratorError) as exc_info:
            generator.generate_response(
                query="Test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        assert "API" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_result_omits_is_error_on_success(
        self, mock_anthropic_class,
        mock_anthropic_response_with_tool_use,
        mock_anthropic_final_response
    ):
        """
        Test: Tool results omit is_error flag on success (Anthropic API convention).
        Scenario: Verify is_error is only included when True.
        """
        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            mock_anthropic_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Success result"

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object"}}]

        generator.generate_response(
            query="Test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify is_error is NOT included for successful execution
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result = second_call[1]["messages"][2]["content"][0]
        assert "is_error" not in tool_result
