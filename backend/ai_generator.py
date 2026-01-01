import anthropic
from anthropic import APIError, AuthenticationError, RateLimitError
from typing import List, Optional, Dict, Any

class AIGeneratorError(Exception):
    """Custom exception for AI Generator errors with user-friendly messages"""
    pass


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Strategy:
- For simple questions: Use a single tool call
- For complex, multi-step questions (e.g., "Find courses covering the same topic as lesson X of course Y"):
  1. First use get_course_outline to understand the course structure
  2. Then use search_course_content based on what you learned
- **Maximum 2 tool rounds per query** - use them wisely
- Synthesize all tool results into accurate, fact-based responses
- If a tool returns an error, acknowledge the limitation gracefully

Search Tool Usage:
- Use search_course_content for questions about specific course content or detailed educational materials
- You may search multiple times if needed to gather comprehensive information

Course Outline Tool Usage:
- Use get_course_outline for questions about course structure, what lessons are in a course, or course overview
- Returns: course title, course link, and complete lesson list with numbers and titles
- Combine with search_course_content for comprehensive answers about specific lessons

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude with error handling
        try:
            response = self.client.messages.create(**api_params)
        except AuthenticationError as e:
            raise AIGeneratorError(
                "API authentication failed. Please check your ANTHROPIC_API_KEY."
            ) from e
        except RateLimitError as e:
            raise AIGeneratorError(
                "API rate limit exceeded. Please wait a moment and try again."
            ) from e
        except APIError as e:
            raise AIGeneratorError(
                f"API error occurred: {e.message}"
            ) from e
        except Exception as e:
            raise AIGeneratorError(
                f"Unexpected error calling Claude API: {str(e)}"
            ) from e

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Handle empty response content
        if not response.content:
            raise AIGeneratorError(
                "Received empty response from Claude API."
            )

        # Extract text from response
        text_content = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_content.append(block.text)

        if not text_content:
            raise AIGeneratorError(
                "Received response without text content from Claude API."
            )

        return "\n".join(text_content)
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, max_iterations: int = 2):
        """
        Handle execution of tool calls and get follow-up response.
        Supports multiple rounds of tool calling up to max_iterations.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_iterations: Maximum number of tool execution rounds

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Add AI's tool use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                        is_error = False
                    except Exception as e:
                        tool_result = f"Tool execution failed: {str(e)}"
                        is_error = True

                    result_entry = {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                    # Only include is_error when True (Anthropic API convention)
                    if is_error:
                        result_entry["is_error"] = True
                    tool_results.append(result_entry)

            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare next API call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            # Include tools from original call
            if "tools" in base_params:
                next_params["tools"] = base_params["tools"]

            # Get next response with error handling
            try:
                next_response = self.client.messages.create(**next_params)
            except AuthenticationError as e:
                raise AIGeneratorError(
                    "API authentication failed. Please check your ANTHROPIC_API_KEY."
                ) from e
            except RateLimitError as e:
                raise AIGeneratorError(
                    "API rate limit exceeded. Please wait a moment and try again."
                ) from e
            except APIError as e:
                raise AIGeneratorError(
                    f"API error occurred: {e.message}"
                ) from e
            except Exception as e:
                raise AIGeneratorError(
                    f"Unexpected error calling Claude API: {str(e)}"
                ) from e

            # Check if we got a final text response
            if next_response.stop_reason != "tool_use":
                # No more tool calls - extract and return text
                if not next_response.content:
                    raise AIGeneratorError(
                        "Received empty response from Claude API after tool execution."
                    )

                text_content = []
                for block in next_response.content:
                    if hasattr(block, 'text'):
                        text_content.append(block.text)

                if text_content:
                    return "\n".join(text_content)
                else:
                    raise AIGeneratorError(
                        "Received response without text content from Claude API."
                    )

            # Claude wants to call more tools - continue the loop
            current_response = next_response

        # Max iterations reached - try to extract any text we have
        text_content = []
        for block in current_response.content:
            if hasattr(block, 'text'):
                text_content.append(block.text)

        if text_content:
            return "\n".join(text_content)

        return "I've searched the course materials but reached the maximum number of search attempts. Please try rephrasing your question."