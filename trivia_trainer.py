import re
from base_trainer import BaseTrainer
import os
from datetime import datetime

class TriviaTrainer(BaseTrainer):
    """Trainer specialized for TriviaQA factual recall"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="trivia"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)
    
    def _create_experiment_directory(self):
        """Create TriviaQA-specific experiment directory"""
        # TriviaQA doesn't have complex filtering, so keep it simple
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.model_name}-{self.dataset_name}-{timestamp}"
        
        experiment_path = os.path.join("results", dir_name)
        os.makedirs(experiment_path, exist_ok=True)
        
        return experiment_path
    
    def create_result_entry(self, item, prompt, response, ground_truth, prediction, is_correct, problem_index):
        """Create TriviaQA-specific result entry with question source and answer aliases"""
        return {
            "problem_index": problem_index,
            "question": item['question'],
            "ground_truth": ground_truth,
            "answer_aliases": item.get('answer_aliases', []),
            "normalized_aliases": item.get('normalized_aliases', []),
            "question_id": item.get('question_id', ''),
            "question_source": item.get('question_source', ''),
            "prompt": prompt,
            "model_response": response,
            "extracted_prediction": prediction,
            "is_correct": is_correct
        }
    
    def extract_answer(self, output):
        """Return the full output - we'll check if correct answer is contained within"""
        return output.strip() if output else ""

    def prepare_prompt(self, item):
        """Prepare prompt for trivia question"""
        question = item['question']
        
        if self.model_name.lower() in ["qwen-base", "qwen-instruct"]:
            prompt = (
                "<|im_start|>system\n"
                "Please answer the trivia question directly and concisely.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{question}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            return prompt
            
        elif self.model_name.lower() in ["deepseek-distilled", "open-reasoner", "llama-distilled"]:
            prompt = (
                "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
                "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
                "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
                f"User: {question}\n"
                "Assistant: <think>"
            )
            return prompt
            
        elif self.model_name.lower() in ["llama-base", "llama-rl"]:
            prompt = (
                f"Question: {question}\n"
                "Answer: Let's think step by step."
            )
            return prompt
            
        elif self.model_name.lower() == "llama-instruct":
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful assistant. Please answer the trivia question directly and accurately.<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{question}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
            return prompt
            
        else:
            print(f"Unknown model {self.model_name}, using default prompt")
            return question
    
    def get_ground_truth(self, item):
        """Get ground truth answer from TriviaQA item"""
        return item['answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if ground truth answer appears anywhere in the model's response"""
        if not prediction:
            return False
            
        # Convert both to lowercase for case-insensitive matching
        pred_lower = prediction.lower()
        truth_lower = ground_truth.lower()
        
        # Check if the answer appears in the response
        return truth_lower in pred_lower
    
    def check_correctness_with_aliases(self, prediction, item):
        """Enhanced correctness check using all answer aliases"""
        if not prediction:
            return False
            
        pred_lower = prediction.lower()
        
        # Check against all normalized aliases
        for alias in item.get('normalized_aliases', []):
            if alias in pred_lower:
                return True
                
        return False
