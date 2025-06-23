from base_trainer import BaseTrainer
from math_verify import parse, verify
from func_timeout import func_timeout, FunctionTimedOut
import os
from datetime import datetime

class Math500Trainer(BaseTrainer):
    """Trainer specialized for Math500 problems"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="math500"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)
    
    def _create_experiment_directory(self):
        """Create Math500-specific experiment directory with category info"""
        # Get dataset info to create descriptive name
        try:
            stats = self.dataset.get_stats()
            categories = list(stats['categories'].keys())
            
            # Create a descriptive name
            if len(categories) == stats['num_categories'] and stats['num_categories'] > 3:
                # All categories - use "all"
                category_str = "all"
            else:
                # Specific categories - join with hyphens, limit length
                category_str = "-".join(categories[:3])  # Take first 3 to avoid too long names
                if len(categories) > 3:
                    category_str += f"-plus{len(categories)-3}more"
        except:
            category_str = "all"
        
        # Create directory name: model-dataset-categories-timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.model_name}-{self.dataset_name}-{category_str}-{timestamp}"
        
        experiment_path = os.path.join("results", dir_name)
        os.makedirs(experiment_path, exist_ok=True)
        
        return experiment_path
    
    def create_result_entry(self, item, prompt, response, ground_truth, prediction, is_correct, problem_index):
        """Create Math500-specific result entry with category, level, and verification details"""
        return {
            "problem_index": problem_index,
            "problem": item['problem'],
            "category": item['category'],
            "level": item['level'],
            "ground_truth": ground_truth,
            "full_solution": item.get('solution', ''),
            "prompt": prompt,
            "model_response": response,
            "extracted_prediction": prediction,
            "is_correct": is_correct
        }
    
    def extract_answer(self, output):
        """Extract answer from Math500 model output using math_verify - return only the value for JSON"""
        parsed_result = parse(output)
        # If parse returns a tuple/list, take index 1, otherwise return as is
        if isinstance(parsed_result, (tuple, list)) and len(parsed_result) > 1:
            return parsed_result[1]
        return parsed_result
        
    def prepare_prompt(self, item):
        """Prepare prompt for Math500 problem"""
        problem = item['problem']
        
        if self.model_name.lower() in ["qwen-base", "qwen-instruct"]:
            question = (
                "<|im_start|>system\n"
                "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{problem}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            return question
            
        elif self.model_name.lower() in ["deepseek-distilled", "open-reasoner", "llama-distilled"]:
            question = (  # <-- Correct indentation (4 spaces)
                "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
                "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
                "User: You must put your answer inside <answer> tags, i.e., <answer> answer here </answer>. "
                "And your final answer will be extracted automatically by the \\boxed{{}} tag. "
                f"{problem} "
                "Assistant:"
            )
            return question  # <-- Correct indentation (4 spaces)
            
        elif self.model_name.lower() in ["llama-base"]:

            question = (
                "Question:\n" 
                f"{problem}\n"
                "Answer:\n" 
                "Let's think step by step."
            )
            return question

        elif self.model_name.lower() in ["llama-rl"]:

            question = (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{problem}\n"
                "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            return question
            
        elif self.model_name.lower() == "llama-instruct":
            
            question = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful mathematics assistant. Please solve the problem step by step and provide your final answer within \\boxed{}.<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{problem}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
            return question
            
        else:
            print(f"Unknown model {self.model_name}, using default prompt")
            return f"Solve this math problem step by step:\n{problem}"

    def get_ground_truth(self, item):
        """Get ground truth answer from Math500 item"""
        return item['solution']

    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for math problems using math_verify"""
        gold = parse(ground_truth)
        answer = parse(prediction)
        return verify(gold, answer)
