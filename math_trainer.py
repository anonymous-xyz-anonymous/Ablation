from base_trainer import BaseTrainer
from math_verify import parse, verify

class MathTrainer(BaseTrainer):
    """Trainer specialized for math problems (GSM8K)"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="math"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)
    
    def create_result_entry(self, item, prompt, response, ground_truth, prediction, is_correct, problem_index):
        """Create GSM8K-specific result entry with question, solution steps"""
        return {
            "problem_index": problem_index,
            "question": item['question'],
            "ground_truth": ground_truth,
            "solution_steps": item.get('solution_steps', ''),
            "prompt": prompt,
            "model_response": response,
            "extracted_prediction": prediction,
            "is_correct": is_correct
        }
    
    def extract_answer(self, output):
        """Extract numerical answer from math model output - return only the value for JSON"""
        parsed_result = parse(output)
        # If parse returns a tuple/list, take index 1, otherwise return as is
        if isinstance(parsed_result, (tuple, list)) and len(parsed_result) > 1:
            return parsed_result[1]
        return parsed_result
    
    def prepare_prompt(self, item):
        """Prepare prompt for math problem"""
        problem = item['question']

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
            question = (
                "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
                "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
                "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
                f"User: {problem}\n"
                "Assistant: <think>"
            )
            return question
            
        elif self.model_name.lower() in ["llama-base", "llama-rl"]:
            question = (
                f"Question: {problem}\n"
                "Answer: Let's think step by step."
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
        """Get ground truth answer from GSM8K item"""
        return item['final_answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for math problems using math_verify"""
        
        # Parse both ground truth and prediction
        gold = parse(ground_truth)
        answer = parse(prediction)
        
        # Order is important: verify(gold, answer)
        return verify(gold, answer)
