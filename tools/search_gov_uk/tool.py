"""
Lambda function which does semantic search of gov uk
"""

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError, OpenSearchException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)


def get_secret(secret_name: str) -> dict[str, str]:
    """
    Get secret from AWS Secrets Manager

    Args:
        secret_name (str): The name of the secret to get

    Returns:
        dict: Dictionary of secret keys and values

    Raises:
        ClientError: When connection to AWS fails, typically due to insufficient permissions.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name="eu-west-2",
    )
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logging.error(f"Accessing secret {secret_name} failed. Error: {e}")
        raise e
    return json.loads(get_secret_value_response["SecretString"])


class SearchResult(BaseModel):
    url: str
    score: float
    document_type: str
    title: str
    description: Optional[str]
    heading_hierarchy: list[str]
    html_content: str


def semantic_search(search_query: str) -> list[SearchResult]:
    try:
        opensearch_config = get_secret("GdsChat")
        opensearch_url = opensearch_config["url"]
        opensearch_username = opensearch_config["username"]
        opensearch_password = opensearch_config["password"]
        index_name = "govuk_chat_chunked_content"

        try:
            search_client = OpenSearch(
                hosts=[opensearch_url],
                http_auth=(
                    opensearch_username,
                    opensearch_password,
                ),
            )
        except TypeError:
            logging.error(
                "The credentials required to connect to OpenSearch are missing from Secrets Manager or incorrect."
            )
            raise

        bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

        try:
            embedding_response = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": search_query}),
            )
        except NoCredentialsError:
            logging.error("Check you have authenticated with AWS in this console")
            raise

        response_body = json.loads(embedding_response["body"].read())

        try:
            search_response = search_client.search(
                index=index_name,
                body={
                    "size": 5,
                    "query": {
                        "knn": {
                            "titan_embedding": {
                                "vector": response_body["embedding"],
                                "k": 5,
                            }
                        }
                    },
                    "_source": {"exclude": ["titan_embedding"]},
                },
            )
        except ConnectionError:
            logging.error(
                "OpenSearch has failed to connect. Ensure the environment variables are set correctly."
            )
            raise
        except OpenSearchException:
            logging.error(
                "OpenSearch connected but did not return a result. The request may be malformed."
            )
            raise

        results = []
        for hit in search_response["hits"]["hits"]:
            result = hit["_source"]
            result["url"] = f"https://www.gov.uk{result['exact_path']}"
            result["score"] = hit["_score"]
            results.append(SearchResult(**result))

        return results
    except Exception as e:
        logging.error(e)
        raise


class SemanticSearch:
    connections = {}

    @classmethod
    def search_gov_uk(cls, search_query: str) -> list[SearchResult]:
        try:
            search_client = cls.get_connection("GdsChat")
            index_name = "govuk_chat_chunked_content"

            search_query_embedded = cls.embed_text(search_query)

            try:
                search_response = search_client.search(
                    index=index_name,
                    body={
                        "size": 5,
                        "query": {
                            "knn": {
                                "titan_embedding": {
                                    "vector": search_query_embedded,
                                    "k": 5,
                                }
                            }
                        },
                        "_source": {"exclude": ["titan_embedding"]},
                    },
                )
            except ConnectionError:
                logging.error(
                    "OpenSearch has failed to connect. Ensure the environment variables are set correctly."
                )
                raise
            except OpenSearchException:
                logging.error(
                    "OpenSearch connected but did not return a result. The request may be malformed."
                )
                raise
            except TypeError:
                logging.error(
                    "The credentials required to connect to OpenSearch were not set correctly."
                )
                raise

            results = []

            for hit in search_response["hits"]["hits"]:
                try:
                    result = hit["_source"]
                    result["url"] = f"https://www.gov.uk{result['exact_path']}"
                    result["score"] = hit["_score"]
                    results.append(SearchResult(**result))
                except KeyError:
                    logging.warning(f"Hit malformed: {hit}\nSkipping.")
                    continue
            return results
        except Exception as e:
            logging.error(e)
            raise

    @classmethod
    def get_connection(cls, connection_name) -> tuple[str, str, str]:
        """
        Get connection details for Opensearch cluster
        Args:
            connection_name (str): The name of the connection to get
        Returns:
            tuple: url, username, password
        """
        if not cls.connections.get(connection_name):
            connection_params = cls.get_secret(connection_name)
            try:
                url = connection_params["url"]
                username = connection_params["username"]
                password = connection_params["password"]

                search_client = OpenSearch(
                    hosts=[url],
                    http_auth=(
                        username,
                        password,
                    ),
                )
                cls.connections[connection_name] = search_client
            except KeyError:
                logging.error(
                    f"The connection {connection_name} is corrupted. It should have the fields url, username and password. It has the fields {connection_params.keys()}"
                )
                raise

        return cls.connections[connection_name]

    @staticmethod
    def get_secret(secret_name: str) -> dict[str, str]:
        """
        Get secret from AWS Secrets Manager

        Args:
            secret_name (str): The name of the secret to get

        Returns:
            dict: Dictionary of secret keys and values

        Raises:
            ClientError: When connection to AWS fails, typically due to insufficient permissions.
        """
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager",
            region_name="eu-west-2",
        )
        try:
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        except ClientError as e:
            logging.error(f"Accessing secret {secret_name} failed. Error: {e}")
            raise e
        return json.loads(get_secret_value_response["SecretString"])

    @staticmethod
    def embed_text(text: str) -> str:
        bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

        try:
            embedding_response = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text}),
            )
        except NoCredentialsError:
            logging.error("Check you have authenticated with AWS in this console")
            raise

        response_body = json.loads(embedding_response["body"].read())
        return response_body["embedding"]


def lambda_handler(event, context):
    delimiter = "___"

    original_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
    tool_name = original_tool_name[
        original_tool_name.index(delimiter) + len(delimiter) :
    ]

    if tool_name == "searchGovUk":
        search_query = event.get("query")

        if not search_query:
            raise KeyError("The field 'query' must be passed in the event object")

        results = semantic_search(search_query)

        return [r.model_dump() for r in results]
    else:
        raise NotImplementedError(
            f"The tool {tool_name} was not implemented. Only searchGovUk is a valid tool."
        )
