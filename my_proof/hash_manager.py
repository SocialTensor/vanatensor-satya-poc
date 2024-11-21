import boto3
import json
from datetime import datetime
import logging
import hashlib

class HashManager:
    def __init__(self, bucket_name, remote_file_key, aws_access_key_id, aws_secret_access_key):
        # Initialize S3 client with credentials
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.bucket_name = bucket_name
        self.remote_file_key = remote_file_key

    def _initialize_empty_hash_file(self):
        """Initialize an empty hash file in S3"""
        data = {
            'hashes': [],
            'lastUpdated': datetime.utcnow().isoformat() + 'Z'
        }
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=self.remote_file_key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        return []

    def get_remote_hashes(self):
        """Fetch hashes from remote S3 JSON file"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.remote_file_key
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data.get('hashes', [])
        except self.s3_client.exceptions.NoSuchKey:
            # If file doesn't exist, create it and return empty list
            return self._initialize_empty_hash_file()
        except Exception as e:
            logging.error(f"Error fetching remote hashes: {str(e)}")
            return []

    def update_remote_hashes(self, new_hashes):
        """Update remote JSON file with new hashes"""
        try:
            data = {
                'hashes': new_hashes,
                'lastUpdated': datetime.utcnow().isoformat() + 'Z'
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.remote_file_key,
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            logging.error(f"Error updating remote hashes: {str(e)}")
            

    def add_hash(self, new_hash):
        """Add a single hash to the remote file"""
        current_hashes = self.get_remote_hashes()
        if new_hash not in current_hashes:
            current_hashes.append(new_hash)
            self.update_remote_hashes(current_hashes)
            return True
        return False

    def remove_hash(self, hash_to_remove):
        """Remove a hash from the remote file"""
        current_hashes = self.get_remote_hashes()
        if hash_to_remove in current_hashes:
            current_hashes.remove(hash_to_remove)
            self.update_remote_hashes(current_hashes)
            return True
        return False

    def generate_hash(self, input_string):
        """Generate a SHA-256 hash from an input string
        
        Args:
            input_string (str): The string to hash
            
        Returns:
            str: The hexadecimal representation of the hash
        """
        # Encode the string to bytes and generate hash
        hash_object = hashlib.sha256(str(input_string).encode())
        return hash_object.hexdigest()
