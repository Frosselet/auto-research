from __future__ import annotations

from auto_research.secrets.base import Secrets


class SecretsAWS(Secrets):
    """Secrets impl backed by AWS Secrets Manager.

    `key` is the secret name (or ARN). The secret value is returned verbatim — if you
    store JSON in the secret, parse it on the caller side. For OPENAI_API_KEY, the
    convention is to store the raw key string.
    """

    def __init__(self, region_name: str | None = None, client=None):
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("secretsmanager", region_name=region_name)

    def get(self, key: str) -> str:
        resp = self._client.get_secret_value(SecretId=key)
        # SecretsManager returns either SecretString or SecretBinary; for our use
        # case (API keys) it's always SecretString.
        secret = resp.get("SecretString")
        if secret is None:
            raise RuntimeError(f"secret {key!r} has no SecretString")
        return secret
