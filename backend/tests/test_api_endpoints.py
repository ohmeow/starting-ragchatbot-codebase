"""
Tests for FastAPI endpoints - validates HTTP request/response handling.
These tests use a test app that mirrors production endpoints without static file dependencies.
"""
import pytest
from unittest.mock import Mock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_200_with_valid_request(self, test_client, sample_query_request):
        """
        Test: Valid query request returns 200 with proper response structure.
        """
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_creates_session_when_not_provided(self, test_client, mock_rag_system):
        """
        Test: Session ID is created when not provided in request.
        """
        response = test_client.post("/api/query", json={"query": "What is ML?"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, test_client, sample_query_request_with_session, mock_rag_system):
        """
        Test: Provided session ID is used and returned.
        """
        response = test_client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.query.assert_called_with(
            "Tell me more about neural networks",
            "existing-session-456"
        )

    def test_query_returns_answer_from_rag_system(self, test_client, mock_rag_system):
        """
        Test: Response contains answer from RAG system.
        """
        mock_rag_system.query.return_value = ("Custom test answer", [])

        response = test_client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Custom test answer"

    def test_query_returns_sources_from_rag_system(self, test_client, mock_rag_system):
        """
        Test: Response contains sources from RAG system.
        """
        mock_rag_system.query.return_value = ("Answer", [
            {"title": "Source 1", "link": "https://example.com/1"},
            {"title": "Source 2", "link": None}
        ])

        response = test_client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["title"] == "Source 1"
        assert data["sources"][0]["link"] == "https://example.com/1"
        assert data["sources"][1]["link"] is None

    def test_query_returns_422_for_missing_query(self, test_client):
        """
        Test: Missing query field returns 422 validation error.
        """
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_returns_422_for_invalid_json(self, test_client):
        """
        Test: Invalid JSON body returns 422 error.
        """
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_returns_500_on_rag_system_error(self, test_client, mock_rag_system):
        """
        Test: RAG system exceptions are caught and return 500.
        """
        mock_rag_system.query.side_effect = Exception("Internal RAG error")

        response = test_client.post("/api/query", json={"query": "Test"})

        assert response.status_code == 500
        assert "Internal RAG error" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_200(self, test_client):
        """
        Test: Courses endpoint returns 200 with proper structure.
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_courses_returns_correct_count(self, test_client, mock_rag_system):
        """
        Test: Response contains correct course count from RAG system.
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3

    def test_courses_returns_course_titles(self, test_client, mock_rag_system):
        """
        Test: Response contains course titles from RAG system.
        """
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["course_titles"] == ["Course A", "Course B", "Course C"]

    def test_courses_returns_empty_when_no_courses(self, test_client, mock_rag_system):
        """
        Test: Empty course list is handled correctly.
        """
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_returns_500_on_error(self, test_client, mock_rag_system):
        """
        Test: Analytics errors are caught and return 500.
        """
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint"""

    def test_root_returns_200(self, test_client):
        """
        Test: Root endpoint returns 200.
        """
        response = test_client.get("/")

        assert response.status_code == 200

    def test_root_returns_api_message(self, test_client):
        """
        Test: Root endpoint returns API identification message.
        """
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG System" in data["message"]


class TestAsyncEndpoints:
    """Async tests for API endpoints using httpx"""

    @pytest.mark.asyncio
    async def test_async_query_endpoint(self, async_test_client):
        """
        Test: Query endpoint works with async client.
        """
        response = await async_test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    @pytest.mark.asyncio
    async def test_async_courses_endpoint(self, async_test_client):
        """
        Test: Courses endpoint works with async client.
        """
        response = await async_test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, async_test_client):
        """
        Test: Multiple concurrent requests are handled properly.
        """
        import asyncio

        async def make_query(query_text):
            return await async_test_client.post(
                "/api/query",
                json={"query": query_text}
            )

        # Make 3 concurrent requests
        responses = await asyncio.gather(
            make_query("Query 1"),
            make_query("Query 2"),
            make_query("Query 3")
        )

        # All should succeed
        for response in responses:
            assert response.status_code == 200


class TestRequestValidation:
    """Tests for request validation and edge cases"""

    def test_query_with_empty_string(self, test_client):
        """
        Test: Empty query string is accepted (validation is application-level).
        """
        response = test_client.post("/api/query", json={"query": ""})

        # Empty string passes Pydantic validation, behavior is app-defined
        assert response.status_code == 200

    def test_query_with_long_text(self, test_client, mock_rag_system):
        """
        Test: Long query text is handled without error.
        """
        long_query = "What is " + "machine learning " * 500 + "?"

        response = test_client.post("/api/query", json={"query": long_query})

        assert response.status_code == 200

    def test_query_with_special_characters(self, test_client, mock_rag_system):
        """
        Test: Special characters in query are handled.
        """
        response = test_client.post("/api/query", json={
            "query": "What's the difference between 'neural networks' & CNNs? <test>"
        })

        assert response.status_code == 200

    def test_query_with_unicode(self, test_client, mock_rag_system):
        """
        Test: Unicode characters in query are handled.
        """
        response = test_client.post("/api/query", json={
            "query": "What is 机器学习? (machine learning in Chinese)"
        })

        assert response.status_code == 200

    def test_extra_fields_are_ignored(self, test_client):
        """
        Test: Extra fields in request are ignored (Pydantic default).
        """
        response = test_client.post("/api/query", json={
            "query": "Test",
            "extra_field": "ignored",
            "another": 123
        })

        assert response.status_code == 200
