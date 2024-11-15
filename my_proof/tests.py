from typing import List, Dict, Any

from my_proof.aws_interaction import download_json_from_s3

MINIMUM_TOTAL_AVERAGE_TIME=15 #minimum average time to anwser a questsion
MINIMUM_CHARACTER_TIME=0.05 #minimum time to anwsers per characters https://irisreading.com/what-is-the-average-reading-speed/

# could also define this as a minimum time per character, ie everychar mean should take at least 0.1 seconds, than could grade number of pass fails on that.
# lets do that as another test
def Time_Minimums(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        total_time = sum(float(item['time_taken']) for item in data_list)
        average_time = total_time / len(data_list)
        
        if average_time < MINIMUM_TOTAL_AVERAGE_TIME:
            return {
                'score': 0.0,
                'comments': [
                    'The total average time is less than the defined minimum average time',
                    f'Average time to answer is {average_time:.2f} and the minimum is {MINIMUM_TOTAL_AVERAGE_TIME}'
                ]
            }
        
        return {
            'score': 1.0,
            'comments': [
                'Test passed',
                f'Average time to answer is {average_time:.2f} and the minimum is {MINIMUM_TOTAL_AVERAGE_TIME}'
            ]
        }
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while processing the average minimum time: {str(e)}']
        }

def Character_Timing(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        passes = []
        for item in data_list:
            total_char_len = len(item['prompt'])+sum([len(i['response']) for i in item['responses']])
            min_response_time = total_char_len*MINIMUM_CHARACTER_TIME
            passes.append(min_response_time<float(item['time_taken']))
        pass_rate = sum(passes)/len(data_list)

        return {
            'score': pass_rate,
            'comments': [f'Passed {sum(passes)} out of {len(data_list)} character timing checks']
        }
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while processing character timing: {str(e)}']
        }

def Time_Distribution(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        # Collect pairs of (char_length, time_taken)
        pairs = []
        for item in data_list:
            total_char_len = len(item['prompt']) + sum(len(i['response']) for i in item['responses'])
            time_taken = float(item['time_taken'])
            pairs.append((total_char_len, time_taken))
        
        if len(pairs) < 2:
            return {
                'score': 0.0,
                'comments': ['Not enough data points to calculate correlation']
            }
        
        # Calculate means
        mean_x = sum(x for x, _ in pairs) / len(pairs)
        mean_y = sum(y for _, y in pairs) / len(pairs)
        
        # Calculate correlation coefficient
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        denominator = (
            (sum((x - mean_x) ** 2 for x, _ in pairs) *
             sum((y - mean_y) ** 2 for _, y in pairs)) ** 0.5
        )
        
        if denominator == 0:
            return {
                'score': 0.0,
                'comments': ['No variation in data points']
            }
        
        correlation = numerator / denominator
        
        # Convert correlation to a score between 0 and 1
        # Only consider positive correlations, negative ones get 0
        score = max(0, correlation)
        
        return {
            'score': score,
            'comments': [
                f'Correlation coefficient: {correlation:.3f}',
                'Strong positive correlation' if correlation > 0.7 else
                'Moderate positive correlation' if correlation > 0.3 else
                'Weak positive correlation' if correlation > 0 else
                'No positive correlation'
            ]
        }
            
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while processing time distribution: {str(e)}']
        }

def Duplicate_ID_Check(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        id_choice_map = {}
        duplicates = []
        
        for item in data_list:
            uid = item['uniqueID']
            chosen = item['chosen']
            if uid in id_choice_map:
                if id_choice_map[uid] != chosen:
                    duplicates.append(uid)
            else:
                id_choice_map[uid] = chosen
        
        if duplicates:
            return {
                'score': 0.0,
                'comments': [
                    f'Found {len(duplicates)} duplicate IDs with conflicting choices',
                    f'Duplicate IDs: {", ".join(str(d) for d in duplicates)}'
                ]
            }
        
        return {
            'score': 1.0,
            'comments': ['No duplicate IDs with conflicting choices found']
        }
    except (KeyError, ValueError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while checking for duplicate IDs: {str(e)}']
        }

def Choice_Distribution(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from collections import Counter
        
        # Count choices
        chosen_counts = Counter(item['chosen'] for item in data_list)
        total_choices = sum(chosen_counts.values())
        
        # Calculate proportions and check for bias
        max_proportion = 0
        distribution_info = []
        for choice, count in chosen_counts.items():
            proportion = count / total_choices
            max_proportion = max(max_proportion, proportion)
            distribution_info.append(f"Choice {choice}: {proportion*100:.2f}%")
        
        # Score is inverse of maximum proportion, scaled
        # If max proportion is 0.9 or higher, score will be 0
        # If max proportion is 0.5 or lower, score will be 1
        score = max(0, min(1, 2 * (0.9 - max_proportion)))
        
        return {
            'score': score,
            'comments': [
                'Choice distribution analysis:',
                *distribution_info,
                'Distribution is balanced' if score > 0.8 else
                'Distribution shows moderate bias' if score > 0.3 else
                'Distribution shows strong bias'
            ]
        }
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while analyzing choice distribution: {str(e)}']
        }

def Model_Bias(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from collections import defaultdict
        
        model_choice_counts = defaultdict(int)
        total_choices = 0
        
        for item in data_list:
            chosen = item['chosen']
            if isinstance(chosen, float):
                # If chosen is a float, assume it represents the probability of the first response
                chosen_index = 0 if chosen >= 0.5 else 1
            else:
                chosen_index = chosen
            
            chosen_model = item['responses'][chosen_index]['model']
            model_choice_counts[chosen_model] += 1
            total_choices += 1

        # Calculate proportions and build distribution info
        distribution_info = []
        max_proportion = 0
        for model, count in model_choice_counts.items():
            proportion = count / total_choices
            max_proportion = max(proportion, max_proportion)
            distribution_info.append(f"Model '{model}': {proportion*100:.2f}%")

        # Score calculation: similar to Choice_Distribution
        # If max proportion is 0.9 or higher, score will be 0
        # If max proportion is 0.5 or lower, score will be 1
        score = max(0, min(1, 2 * (0.9 - max_proportion)))

        return {
            'score': score,
            'comments': [
                'Model selection distribution:',
                *distribution_info,
                'Model selection is balanced' if score > 0.8 else
                'Model selection shows moderate bias' if score > 0.3 else
                'Model selection shows strong bias'
            ]
        }
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while analyzing model bias: {str(e)}']
        }
    
def Poison_Consistency(data_list: List[Dict[str, Any]], aws_access_key_id: str, aws_secret_access_key: str) -> Dict[str, Any]:
    try:
        # Download poisoned data from S3
        poisoned_data = download_json_from_s3('vanatensorpoisondata', 'poisin.json', aws_access_key_id, aws_secret_access_key)
        if not poisoned_data:
            return {
                'score': 0.0,
                'comments': ['Failed to retrieve poisoned data from S3']
            }

        # Check for consistency between current and poisoned data
        inconsistencies = []
        for item in data_list:
            poisoned_item = next((pi for pi in poisoned_data if pi['uniqueID'] == item['uniqueID']), None)
            if poisoned_item and item['chosen'] != poisoned_item['chosen']:
                inconsistencies.append(item['uniqueID'])

        if inconsistencies:
            return {
                'score': 0.0,
                'comments': [
                    f'Found {len(inconsistencies)} inconsistencies with poisoned data',
                    f'Inconsistent IDs: {", ".join(str(i) for i in inconsistencies)}'
                ]
            }

        return {
            'score': 1.0,
            'comments': ['All poisoned data choices are consistent']
        }
    except Exception as e:
        return {
            'score': 0.0,
            'comments': [f'An error occurred while checking poison consistency: {str(e)}']
        }



