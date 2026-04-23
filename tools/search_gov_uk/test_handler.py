import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from tools.search_gov_uk.handler import SemanticSearch


@pytest.fixture
def embedded_query():
    return "s0m3 3mb3dd3d qu3ry"


@pytest.mark.describe("SemanticSearch")
class TestSemanticSearch:
    @pytest.mark.it("formats valid hits into SearchResult objects")
    @patch("tools.search_gov_uk.handler.SemanticSearch.embed_text")
    @patch("tools.search_gov_uk.handler.SemanticSearch.get_connection")
    def test_search_gov_uk_formats_results(self, mock_get_connection, mock_embed_text):
        # GIVEN
        #   A search client that returns valid OpenSearch hits
        mock_embed_text.return_value = [0.1, 0.2, 0.3]
        mock_client = MagicMock()
        mock_get_connection.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "exact_path": "/some-guidance",
                            "document_type": "guidance",
                            "title": "Some Guidance Title",
                            "description": "A helpful description",
                            "heading_hierarchy": ["h1"],
                            "html_content": "<p>Content</p>",
                        },
                    }
                ]
            }
        }

        # WHEN
        #   A search is executed
        results = SemanticSearch.search_gov_uk("query")

        # THEN
        #   The result is correctly mapped to a list of SearchResult objects
        assert len(results) == 1
        assert results[0].url == "https://www.gov.uk/some-guidance"
        assert results[0].score == 0.95
        assert results[0].title == "Some Guidance Title"

    @pytest.mark.it("raises an error when the OpenSearch cluster connection fails")
    @patch("tools.search_gov_uk.handler.SemanticSearch.embed_text")
    @patch("tools.search_gov_uk.handler.SemanticSearch.get_connection")
    def test_search_gov_uk_handles_malformed_cluster(
        self, mock_get_connection, mock_embed_text
    ):
        # GIVEN
        #   A search client that raises a ConnectionError when called
        mock_embed_text.return_value = [0.1]
        mock_client = MagicMock()
        mock_get_connection.return_value = mock_client
        mock_client.search.side_effect = ConnectionError("Failed to connect to cluster")

        # THEN
        #   The exception should be propagated up
        with pytest.raises(ConnectionError):
            SemanticSearch.search_gov_uk("query")

    @pytest.mark.it("skips hits that are missing required fields")
    @patch("tools.search_gov_uk.handler.SemanticSearch.embed_text")
    @patch("tools.search_gov_uk.handler.SemanticSearch.get_connection")
    def test_search_gov_uk_skips_malformed_results(
        self, mock_get_connection, mock_embed_text
    ):
        # GIVEN
        #   A search client returning one valid hit and one malformed hit (missing exact_path)
        mock_embed_text.return_value = [0.1]
        mock_client = MagicMock()
        mock_get_connection.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            # Missing exact_path here
                            "document_type": "guidance",
                            "title": "Malformed Hit",
                            "description": "Will cause a KeyError",
                            "heading_hierarchy": ["h1"],
                            "html_content": "<p>Content</p>",
                        },
                    },
                    {
                        "_score": 0.85,
                        "_source": {
                            "exact_path": "/valid-path",
                            "document_type": "guidance",
                            "title": "Valid Hit",
                            "description": "Has all fields",
                            "heading_hierarchy": ["h1"],
                            "html_content": "<p>Content</p>",
                        },
                    },
                ]
            }
        }

        # WHEN
        #   A search is executed
        results = SemanticSearch.search_gov_uk("query")

        # THEN
        #   Only the valid hit is processed and returned
        assert len(results) == 1
        assert results[0].url == "https://www.gov.uk/valid-path"
        assert results[0].title == "Valid Hit"

    @pytest.mark.it("retrieves and parses the secret value from AWS Secrets Manager")
    @patch("boto3.session.Session")
    def test_get_secret_returns_value(self, mock_session):
        # GIVEN
        #   AWS Secrets Manager returning a valid JSON string
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"api_key": "12345", "url": "https://opensearch"}'
        }

        # WHEN
        #   The secret is requested
        result = SemanticSearch.get_secret("my_secret")

        # THEN
        #   The parsed dictionary should be returned
        assert result == {"api_key": "12345", "url": "https://opensearch"}

    @pytest.mark.it("raises an error when lacking permissions to access the secret")
    @patch("boto3.session.Session")
    def test_get_secret_handles_no_permissions(self, mock_session):
        # GIVEN
        #   AWS Secrets Manager raises a ClientError (e.g., AccessDenied)
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        error_response = {
            "Error": {"Code": "AccessDeniedException", "Message": "Access Denied"}
        }
        mock_client.get_secret_value.side_effect = ClientError(
            error_response, "GetSecretValue"
        )

        # THEN
        #   The ClientError should be propagated up
        with pytest.raises(ClientError):
            SemanticSearch.get_secret("my_secret")

    @pytest.mark.it("creates and caches a new OpenSearch connection if none exists")
    @patch("tools.search_gov_uk.handler.OpenSearch")
    @patch("tools.search_gov_uk.handler.SemanticSearch.get_secret")
    @patch(
        "tools.search_gov_uk.handler.SemanticSearch.connections", {}
    )  # Start with empty cache
    def test_gets_new_connection(self, mock_get_secret, mock_opensearch):
        # GIVEN
        #   A valid secret containing OpenSearch connection details
        mock_get_secret.return_value = {
            "url": "http://opensearch.local",
            "username": "admin",
            "password": "password123",
        }
        mock_opensearch_instance = MagicMock()
        mock_opensearch.return_value = mock_opensearch_instance

        # WHEN
        #   A connection is requested
        result = SemanticSearch.get_connection("new_conn")

        # THEN
        #   OpenSearch should be initialized and the connection cached and returned
        mock_opensearch.assert_called_once_with(
            hosts=["http://opensearch.local"], http_auth=("admin", "password123")
        )
        assert result == mock_opensearch_instance
        assert SemanticSearch.connections["new_conn"] == mock_opensearch_instance

    @pytest.mark.it("uses an existing connection if one exists")
    @patch(
        "tools.search_gov_uk.handler.SemanticSearch.connections", {"existing": "value"}
    )
    def test_uses_existing_connection(self):
        # GIVEN
        #  There is an existing connection
        result = SemanticSearch.get_connection("existing")

        # THEN
        #   The existing value should be returned
        assert result == "value"

    @pytest.mark.it(
        "produces an error when a connection does not have the required fields"
    )
    @patch("opensearchpy.OpenSearch")
    @patch("tools.search_gov_uk.handler.SemanticSearch.get_secret")
    def test_catches_malformed_connection(self, mock_get_secret, mock_opensearch):
        # GIVEN
        #   Connection parameters missing a password
        mock_get_secret.return_value = {
            "url": "http://some.com",
            "username": "some",
        }

        # THEN
        #   An error should be raised
        with pytest.raises(KeyError):
            SemanticSearch.get_connection("some conn")

    @pytest.mark.it("returns just the embedded text")
    @patch("boto3.client")
    def test_embed_text_returns_embedding(self, mock_client, embedded_query):
        # GIVEN
        #   Embedded output with extraneous information
        mock_embedding_streaming_body = MagicMock()
        mock_embedding_streaming_body.read.return_value = json.dumps(
            {"ignore": "this", "embedding": embedded_query}
        )
        mock_client.return_value.invoke_model.return_value = {
            "some": "response",
            "body": mock_embedding_streaming_body,
        }

        result = SemanticSearch.embed_text("Some text to embed")

        # THEN
        #   Only the embedded text should be returned
        assert result == embedded_query
