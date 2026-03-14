import unittest
from unittest.mock import Mock, patch

from src.infrastructure.termene_client import TermeneClient


class TermeneClientTests(unittest.TestCase):
    def test_search_company_uses_basic_auth_and_schema_key_payload(self):
        client = TermeneClient(
            base_url="https://api.termene.ro/v2",
            username="user",
            password="pass",
            schema_key="schema-123",
        )

        response = Mock()
        response.raise_for_status.return_value = None
        response.content = b"x"
        response.json.return_value = {"results": [{"name": "AUCHAN ROMANIA SA", "cui": "456"}]}

        with patch("src.infrastructure.termene_client.requests.post", return_value=response) as mocked_post:
            result = client.search_company("AUCHAN")

        self.assertEqual(result["name"], "AUCHAN ROMANIA SA")
        mocked_post.assert_called_once()
        _, kwargs = mocked_post.call_args
        self.assertEqual(kwargs["auth"], ("user", "pass"))
        self.assertEqual(kwargs["json"]["schemaKey"], "schema-123")
        self.assertEqual(kwargs["json"]["query"], "AUCHAN")

    def test_search_company_returns_none_when_credentials_missing(self):
        client = TermeneClient(base_url="https://api.termene.ro/v2", username=None, password=None, schema_key=None)

        with patch("src.infrastructure.termene_client.requests.post") as mocked_post:
            result = client.search_company("AUCHAN")

        self.assertIsNone(result)
        mocked_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()

