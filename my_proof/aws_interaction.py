import boto3
import json
import os
from datetime import datetime

def download_json_from_s3(bucket_name, file_key, aws_access_key_id, aws_secret_access_key):

    # Initialize S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        # Download the file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the content and parse it as JSON
        content = response['Body'].read().decode('utf-8')
        json_data = json.loads(content)
        
        return json_data
    except Exception as e:
        print(f"Error downloading or parsing JSON from S3: {str(e)}")
        return None

