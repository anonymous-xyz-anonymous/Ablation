import argparse
import os
import json
from datetime import datetime
from vllm_model import (
    Qwen257BBase, Qwen257BInstruct, DeepSeekR1DistillQwen7B,
    Llama318BBase, Llama318BInstruct, DeepSeekR1DistilledLlama, 
    OpenReasonerBase, Llama31SimpleRLZoo
)
from gsm8 import GSM8KLoader, TriviaQALoader, MMLULoader, Math500Loader
from math_trainer import MathTrainer
from trivia_trainer import TriviaTrainer
from mmlu_trainer import MMLUTrainer
from math_500_trainer import Math500Trainer

def get_model(model_name):
    """Factory function to get model instance"""
    model_classes = {
        "qwen-instruct": Qwen257BInstruct,
        "deepseek-distilled": DeepSeekR1DistillQwen7B,
        "qwen-base": Qwen257BBase,
        "llama-base": Llama318BBase,
        "llama-instruct": Llama318BInstruct,
        "llama-distilled": DeepSeekR1DistilledLlama,
        "open-reasoner": OpenReasonerBase,
        "llama-rl": Llama31SimpleRLZoo
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_classes[model_name]()

def parse_args():
    parser = argparse.ArgumentParser(description="Layer ablation study for LLMs")
    parser.add_argument("--model", type=str, default="qwen-instruct",
                        choices=["qwen-instruct", "deepseek-distilled", "qwen-base", 
                                "llama-base", "llama-instruct", "llama-distilled", "open-reasoner", "llama-rl"])
    parser.add_argument("--dataset", type=str, default="math", 
                        choices=["math", "trivia", "mmlu", "math500"])
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # MMLU-specific arguments
    parser.add_argument("--mmlu_question_type", type=str, default=None,
                        choices=["factual", "reasoning"],
                        help="Filter MMLU questions by type: factual (knowledge recall) or reasoning (logical deduction)")
    parser.add_argument("--mmlu_subjects", type=str, nargs="+", default=None,
                        help="Filter MMLU to specific subjects (e.g., --mmlu_subjects college_mathematics high_school_physics)")
    
    # Math500-specific arguments
    parser.add_argument("--math500_categories", type=str, nargs="+", default=None,
                        choices=["Prealgebra", "Algebra", "Number Theory", 
                                "Counting & Probability", "Geometry", 
                                "Intermediate Algebra", "Precalculus"],
                        help="Filter Math500 to specific categories (e.g., --math500_categories Algebra Geometry)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model and dataset
    model = get_model(args.model)
    
    if args.dataset == "math":
        dataset = GSM8KLoader()
        trainer = MathTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="math"
        )
    elif args.dataset == "trivia":
        dataset = TriviaQALoader()
        trainer = TriviaTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="trivia"
        )
    elif args.dataset == "mmlu":
        # Create MMLU dataset with filtering options
        dataset = MMLULoader(
            split="test",
            subject_filter=args.mmlu_subjects,
            question_type_filter=args.mmlu_question_type
        )
        
        # Print dataset statistics
        stats = dataset.get_stats()
        print(f"MMLU Dataset loaded:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Number of subjects: {stats['num_subjects']}")
        print(f"  Question types: {stats['question_types']}")
        print(f"  Subjects: {list(stats['subjects'].keys())}")
        
        trainer = MMLUTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="mmlu"
        )
    elif args.dataset == "math500":
        # Create Math500 dataset with filtering options
        dataset = Math500Loader(
            split="test",  # Math500 only has test split
            category_filter=args.math500_categories
        )
        
        # Print dataset statistics
        stats = dataset.get_stats()
        print(f"Math500 Dataset loaded:")
        print(f"  Total problems: {stats['total_problems']}")
        print(f"  Number of categories: {stats['num_categories']}")
        print(f"  Categories: {list(stats['categories'].keys())}")
        if 'levels' in stats:
            print(f"  Levels: {list(stats['levels'].keys())}")
        
        trainer = Math500Trainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="math500"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Run layer ablation experiment
    results = trainer.run_layer_ablation()
    
    print(f"\n🎉 Layer ablation experiment completed!")
    print(f"📁 Results saved in: results/{args.model}-{args.dataset}/")

if __name__ == "__main__":
    main()
