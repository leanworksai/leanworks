from google.cloud import storage
from pathlib import Path
from google.cloud.storage import transfer_manager

from google.cloud import storage  # Ensure you have this import
from pathlib import Path

class CloudStorage:
    def __init__(self, gcp_credential_path, bucket='leanworks'):
        self.storage_client = storage.Client.from_service_account_json(gcp_credential_path)
        
        # Check if the bucket exists
        bucket_obj = self.storage_client.lookup_bucket(bucket)
        if bucket_obj is None:
            # Bucket does not exist; create it
            bucket_obj = self.storage_client.create_bucket(bucket)
            print(f"Bucket '{bucket}' created.")
        else:
            print(f"Bucket '{bucket}' already exists.")
        self.bucket = bucket_obj

    def upload_blob_from_memory(self, contents, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(contents)

    def download_blob_to_memory(self, source_blob_name):
        blob = self.bucket.blob(source_blob_name)
        try:
            contents = blob.download_as_string()
            return contents
        except Exception as e:
            # Check if it's a 404 error
            if "404" in str(e):
                print(f"File {source_blob_name} not found in bucket {self.bucket.name}. Error: {e}")
                return None
            else:
                # Re-raise other exceptions
                print(f"Error downloading {source_blob_name}: {e}")
                raise

    def list_files(self, folder_path):
        """
        List all files under a specified folder in the bucket.
        
        Args:
            folder_path (str): Path to the folder in the bucket
            
        Returns:
            list: List of file paths in the folder
        """
        if not folder_path.endswith('/'):
            folder_path += '/'
            
        blobs = self.bucket.list_blobs(prefix=folder_path)
        file_paths = [blob.name for blob in blobs if not blob.name.endswith('/')]
        return file_paths

    def bulk_upload_directory(
            self, 
            source_dir, 
            destination_dir,
            exclusion_keywords=[],
            workers=8
            ):
        directory_as_path_obj = Path(source_dir)
        paths = directory_as_path_obj.rglob("*")
        file_paths = [path for path in paths if path.is_file()]
        relative_paths = [path.relative_to(source_dir) for path in file_paths]
        if exclusion_keywords is not None:
            string_paths = []
            for path in relative_paths:
                for k in exclusion_keywords:
                    if k not in str(path):
                        string_paths.append(str(path))            

        # string_paths = [str(path) for path in relative_paths if ".git/" not in str(path) and "__pycache__" not in str(path)]
        results = transfer_manager.upload_many_from_filenames(
            self.bucket, string_paths, source_directory=source_dir, blob_name_prefix=destination_dir, max_workers=workers
        )
        for name, result in zip(string_paths, results):
            if isinstance(result, Exception):
                print("Failed to upload {} due to exception: {}".format(name, result))
            else:
                print("Uploaded {} to {}.".format(name, self.bucket.name))

    def bulk_delete_folder(self, folder_path, batch_size=100):
        # List all blobs (files and subfolders) in the specified folder
        blobs = list(self.bucket.list_blobs(prefix=folder_path))
        
        # Delete blobs in batches
        for i in range(0, len(blobs), batch_size):
            batch = self.storage_client.batch()  # Create a new batch
            with batch:  # Start batch mode
                for blob in blobs[i:i+batch_size]:
                    print(f"Scheduling deletion of {blob.name}")
                    blob.delete(client=self.storage_client)