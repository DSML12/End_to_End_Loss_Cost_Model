"""storage.py
"""
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from src.config import Config


class ModelStore:
    """Loads the lgbm model from Azure Blob Storage.
    """

    def __init__(self, config: Config  = None, client=None):
        self.config = config or Config()
        if client is not None:
            self.container: ContainerClient = client
        else:
            credential = DefaultAzureCredential()
            svc = BlobServiceClient(
                account_url=f"https://{self.config.azure_storage_account_name}.blob.core.windows.net",
                credential=credential,
            )
            self.container = svc.get_container_client(self.config.azure_container)

    def get_bytes(self, key: str) -> bytes:
        return self.container.get_blob_client(key).download_blob().readall()

    def put_bytes(self, key: str, body: bytes) -> None:
        self.container.upload_blob(name=key, data=body, overwrite=True)

    def load_model(self):
        import lightgbm as lgb
        return lgb.Booster(model_str=self.get_bytes(self.config.model_key).decode())

    def download_data(self, dest: str) -> str:
        with open(dest, "wb") as f:
            self.container.get_blob_client(self.config.data_key).download_blob().readinto(f)
        return dest

    def upload_model(self, model_str) -> None:
        self.put_bytes(self.config.model_key, model_str.encode())

