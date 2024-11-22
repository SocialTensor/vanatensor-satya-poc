import json
import logging
import os
from typing import Dict, Any, List
from my_proof.hash_manager import HashManager
from rich.console import Console
from rich.table import Table

import requests

from my_proof.models.proof_response import ProofResponse
from my_proof.tests import *

top_weights = {
    'Authenticity':0.2,
    'Quality':0.7,
    'Uniquness':0.1
}
test_weights = {
    'Time_Minimums':0.1,
    'Time_Correlation':0.2,
    'Time_Distribution':0.1,
    'Repeat_Anwsers':0.15,
    'Both_Sides':0.15,
    'Model_Distribution':0.05,
    'Poisin_Data':0.25,
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

        # Iterate through files and calculate data validity
        # account_email = None
        # total_score = 0

        for input_filename in os.listdir(self.config['input_dir']):
            input_file = os.path.join(self.config['input_dir'], input_filename)
            if os.path.splitext(input_file)[1].lower() == '.json':
                with open(input_file, 'r') as f:
                    input_data = json.load(f)

        qualityRes = Quality(input_data, self.aws_access_key_id, self.aws_secret_access_key)
        self.proof_response.score = qualityRes['score']
        self.proof_response.valid = qualityRes['score'] > 0.65
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

        return self.proof_response

def Quality(data_list: List[Dict[str, Any]], aws_access_key_id: str, aws_secret_access_key: str) -> float:
    #all tests
    #average time taken is less than 5 seconds
    #time correlates to time
    #distribution in times taken
    #anwsering repeat questions the same way,Check for duplicate uniqueIDs with different 'chosen' values
    #choosing option 1 and option 2, Analyze the distribution of 'chosen' values
    #Check for model bias in 'chosen' responses, might be dumb we expect 70b to do better than 7b so will not be even distribution
    #Perform randomness test using Chi-squared test, make sure there is some distribution in what gets chosen
    #Check if it is poisoned data and make sure that they chose the same response
    # 8 seperate tests
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
    print(report)
    display_report(report)
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

def display_report(report: dict) -> None:
    console = Console()
    
    # Create main score table
    main_score = Table(title="[bold magenta]Quality Assessment Report[/bold magenta]", 
                      show_header=True,
                      header_style="bold cyan")
    main_score.add_column("Overall Score", justify="center", style="bold")
    main_score.add_row(f"{report['score']:.2%}")
    
    # Create detailed results table
    results = Table(show_header=True, header_style="bold cyan", 
                   title="[bold magenta]Detailed Test Results[/bold magenta]")
    results.add_column("Test", style="bold green")
    results.add_column("Score", justify="center")
    results.add_column("Status", justify="center")
    results.add_column("Details", justify="left")

    # Test result emojis
    PASS = "✅"
    PARTIAL = "⚠️"
    FAIL = "❌"

    # Mapping of score ranges to status
    def get_status(score):
        if score >= 0.8: return (PASS, "green")
        if score >= 0.4: return (PARTIAL, "yellow")
        return (FAIL, "red")

    # Add each test result
    for test_name, data in report.items():
        if test_name == 'score':
            continue
            
        score = data['score']
        status_emoji, color = get_status(score)
        
        # Format comments as a single string with line breaks
        comments = '\n'.join(data['comments'])
        
        results.add_row(
            test_name.replace('_', ' '),
            f"[{color}]{score:.2%}[/{color}]",
            status_emoji,
            comments
        )

    # Print the report
    console.print()
    console.print(main_score, justify="center")
    console.print()
    console.print(results)
    console.print()

    # Add a summary footer
    if report['score'] >= 0.8:
        console.print("[bold green]Overall Assessment: EXCELLENT[/bold green]", justify="center")
    elif report['score'] >= 0.6:
        console.print("[bold yellow]Overall Assessment: GOOD[/bold yellow]", justify="center")
    elif report['score'] >= 0.4:
        console.print("[bold yellow]Overall Assessment: FAIR[/bold yellow]", justify="center")
    else:
        console.print("[bold red]Overall Assessment: NEEDS IMPROVEMENT[/bold red]", justify="center")

def fetch_random_number() -> float:
    """Demonstrate HTTP requests by fetching a random number from random.org."""
    try:
        response = requests.get('https://www.random.org/decimal-fractions/?num=1&dec=2&col=1&format=plain&rnd=new')
        return float(response.text.strip())
    except requests.RequestException as e:
        logging.warning(f"Error fetching random number: {e}. Using local random.")
        return __import__('random').random()
