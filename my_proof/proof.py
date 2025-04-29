import json
import logging
import os
from typing import Dict, Any, List
from my_proof.hash_manager import HashManager
from rich.console import Console
from rich.table import Table
from my_proof.models.proof_response import ProofResponse
from my_proof.tests import *

top_weights = {
    'Authenticity':0.2,
    'Quality':0.7,
    'Uniquness':0.1
}
test_weights = {
    'Time_Minimums':0.2,
    'Time_Correlation':0.2,
    'Time_Distribution':0.1,
    'Repeat_Anwsers':0.15,
    'Both_Sides':0.15,
    'Model_Distribution':0.05,
    'Poisin_Data':0.15,
}

class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logging.info(f"Config: {self.config}")
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])
        self.aws_access_key_id = config['aws_access_key_id']
        self.aws_secret_access_key = config['aws_secret_access_key']

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        for input_filename in os.listdir(self.config['input_dir']):
            input_file = os.path.join(self.config['input_dir'], input_filename)
            if os.path.splitext(input_file)[1].lower() == '.json':
                with open(input_file, 'r') as f:
                    input_data = json.load(f)

        qualityRes = Quality(input_data, self.aws_access_key_id, self.aws_secret_access_key)
        self.proof_response.score = (qualityRes['score'])*(len(input_data)/100)
        self.proof_response.valid = qualityRes['score'] > 0.25
        self.proof_response.time_minimums = qualityRes['Time_Minimums']['score']
        self.proof_response.time_correlation = qualityRes['Time_Correlation']['score']
        self.proof_response.time_distribution = qualityRes['Time_Distribution']['score']
        self.proof_response.repeat_anwsers = qualityRes['Repeat_Anwsers']['score']
        self.proof_response.both_sides = qualityRes['Both_Sides']['score']
        self.proof_response.model_distribution = qualityRes['Model_Distribution']['score']
        self.proof_response.poison_data = qualityRes['Poisin_Data']['score']
        
        self.proof_response.uniqueness = Uniqueness(input_data, self.aws_access_key_id, self.aws_secret_access_key)

        # original fields
        self.proof_response.quality = qualityRes['score']
        self.proof_response.ownership = 1.0
        self.proof_response.authenticity = 1.0
        self.proof_response.attributes = {
            'total_score': qualityRes['score'],
            'score_threshold': qualityRes['score'],
            'email_verified': True,
        }


        if len(input_data) > 105: #UI does not allow more than 100
            self.proof_response.score = 0.0
            self.proof_response.valid = False

        return self.proof_response

def Quality(data_list: List[Dict[str, Any]], aws_access_key_id: str, aws_secret_access_key: str) -> float:
    report = {
        'Time_Minimums':Time_Minimums(data_list),
        'Time_Correlation':Character_Timing(data_list),
        'Time_Distribution':Time_Distribution(data_list),
        'Repeat_Anwsers':Duplicate_ID_Check(data_list),
        'Both_Sides':Choice_Distribution(data_list),
        'Model_Distribution':Model_Bias(data_list),
        'Poisin_Data':Poison_Consistency(data_list, aws_access_key_id, aws_secret_access_key),
        'score':0
    }
    report['score'] = sum(test_weights[test] * report[test]['score'] for test in test_weights)
    return report

def Uniqueness(data_list: List[Dict[str, Any]], aws_access_key_id: str, aws_secret_access_key: str) -> float:
    hash_manager = HashManager(bucket_name="vanatensordlp", remote_file_key="verified_hashes/hashes.json", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    generated_hash = hash_manager.generate_hash(data_list)
    existing_hashes = hash_manager.get_remote_hashes()
    if generated_hash in existing_hashes or generated_hash == []:
        return 0.0
    else:
        hash_manager.update_remote_hashes(generated_hash)
        return 1.0