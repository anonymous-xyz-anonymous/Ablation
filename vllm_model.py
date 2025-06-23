from vllm import LLM, SamplingParams
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, gc, copy
import random
from datetime import datetime
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory


class BaseVllmModel:
    """Base class for adoell vLLM models with common functionality"""
    
    def __init__(self, model_id, checkpoint_path, **llm_kwargs):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        
        print("Loading HuggingFace model...", flush=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Default LLM parameters - consistent max_model_len across all models
        default_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_model_len": 9000,
            "gpu_memory_utilization": 0.95,
            "enforce_eager": True  # ADDED: Disable compilation
        }
        default_params.update(llm_kwargs)
        
        self.llm = LLM(model=model_id, **default_params)
        print("HuggingFace model loaded.", flush=True)
        
        # Consistent sampling params with max_tokens=8000
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def generate(self, prompt, **kwargs):
        if isinstance(prompt, str):
            params = self.sampling_params
            if kwargs:
                params = SamplingParams(**{**vars(self.sampling_params), **kwargs})
            
            outputs = self.llm.generate(prompt, params)
            return outputs[0].outputs[0].text
        else:
            raise ValueError("For batch processing, use generate_batch instead")
    
    def generate_batch(self, prompts, **kwargs):
        params = self.sampling_params
        if kwargs:
            params = SamplingParams(**{**vars(self.sampling_params), **kwargs})

        print("I am generaitng the full batch!!!", flush=True)
        
        batch_outputs = self.llm.generate(prompts, params)
        print("Full batch generation is complete!!!!", flush=True)
        results = [output.outputs[0].text for output in batch_outputs]
        return results

    def _cleanup_and_copy(self, layer_number):
        """Cleanup and deepcopy with func_timeout - both operations protected"""
        import time
        from func_timeout import func_timeout, FunctionTimedOut
        import gc
        import copy
        
        print(f"🚀 Starting cleanup for layer {layer_number}...", flush=True)
        start_time = time.time()
        
        # Step 1: Cleanup with timeout
        def do_cleanup():
            """The actual cleanup function with comprehensive vLLM shutdown"""
            
            # Method 1: Process cleanup FIRST
            print("📝 Cleaning up vLLM processes...", flush=True)
            try:
                import subprocess
                subprocess.run(['pkill', '-f', 'vllm'], check=False, timeout=5)
                time.sleep(1)
                print("cleaned VLLM process", flush=True)
            except:
                pass

            try:
                print("📝 Shutting down vLLM engine...", flush=True)
                if hasattr(self, 'llm') and self.llm is not None:
                    
                    try:
                        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory  
                        cleanup_dist_env_and_memory(shutdown_ray=True)
                        print("✅ cleanup_dist_env_and_memory() completed", flush=True)
                    except Exception as e:
                        print(f"⚠️ cleanup_dist_env_and_memory() error (ignored): {e}", flush=True)
                    
                    # Method 2: PyTorch distributed cleanup
                    try:
                        print("📝 Destroying PyTorch process group...", flush=True)
                        import contextlib
                        with contextlib.suppress(Exception):
                            torch.distributed.destroy_process_group()
                        print("✅ PyTorch process group cleanup completed", flush=True)
                    except Exception as e:
                        print(f"⚠️ PyTorch distributed cleanup error (ignored): {e}", flush=True)

                    # Method 3: CUDA cleanup
                    print("📝 Clearing CUDA cache...", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            torch.cuda.ipc_collect()
                        except:
                            pass
                    
                    # Method 4: Garbage collection (multiple passes)
                    print("📝 Running garbage collection...", flush=True)
                    for i in range(3):
                        collected = gc.collect()
                        time.sleep(0.1)
                        print(f"📝 GC pass {i+1}: collected {collected} objects", flush=True)
                    
                    # Method 5: Delete LLM object LAST
                    print("📝 Deleting LLM...", flush=True)
                    del self.llm
                    self.llm = None
                    
                    print("✅ Comprehensive cleanup completed successfully", flush=True)
                    
            except Exception as e:
                print(f"❌ Cleanup error: {e}", flush=True)
                return False
        
        try:
            # Run cleanup with 3-minute timeout (increased for comprehensive cleanup)
            result = func_timeout(180, do_cleanup)  # 180 seconds = 3 minutes
            cleanup_time = time.time() - start_time
            print(f"✅ Cleanup finished in {cleanup_time:.1f}s", flush=True)
            
        except FunctionTimedOut:
            print("❌ Cleanup timed out after 3 minutes, proceeding to deepcopy anyway", flush=True)
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}, proceeding to deepcopy anyway", flush=True)
        
        # Step 2: Deep copy with timeout
        def do_deepcopy():
            """The actual deepcopy function"""
            try:
                print("📝 Moving model to CPU...", flush=True)
                self.hf_model = self.hf_model.cpu()
                
                print("📝 Creating deep copy...", flush=True)
                model_copy = copy.deepcopy(self.hf_model)
                
                print("✅ Deep copy completed successfully", flush=True)
                return model_copy
                
            except Exception as e:
                print(f"❌ Deep copy error: {e}", flush=True)
                return None
        
        try:
            # Run deepcopy with 2-minute timeout
            model_copy = func_timeout(120, do_deepcopy)  # 120 seconds = 2 minutes
            
            if model_copy is not None:
                total_time = time.time() - start_time
                print(f"✅ Deep copy finished in {total_time:.1f}s total", flush=True)
                return model_copy
            else:
                print("❌ Deep copy returned None, using fresh copy", flush=True)
                return self._create_fresh_model_copy()
                
        except FunctionTimedOut:
            print("❌ Deep copy timed out after 2 minutes, using fresh copy", flush=True)
            return self._create_fresh_model_copy()
            
        except Exception as e:
            print(f"❌ Deep copy failed: {e}, using fresh copy", flush=True)
            return self._create_fresh_model_copy()


    def _create_fresh_model_copy(self):
        """Fallback: load fresh model when deep copy fails"""
        print("Loading fresh model copy as fallback...", flush=True)
        
        try:
            fresh_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ Fresh model loaded as fallback", flush=True)
            return fresh_model
        except Exception as e:
            print(f"❌ Fresh model loading failed: {e}", flush=True)
            raise e

    def _save_and_reload(self, model_copy, **llm_kwargs):
        """Save ablated model and reload in vLLM with consistent parameters"""
        print("Saving ablated model...", flush=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Clean up existing files
        if os.path.exists(self.checkpoint_path):
            for file in os.listdir(self.checkpoint_path):
                file_path = os.path.join(self.checkpoint_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        model_copy.save_pretrained(self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)

        print("Ablated file saved", flush=True)
        
        # Clean up
        del model_copy
        
        print("Loading ablated model in VLLM...", flush=True)
        # CONSISTENT parameters for reloading - use same max_model_len
        reload_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_model_len": 9000,
            "gpu_memory_utilization": 0.95,
            "enforce_eager": True  # CRITICAL: Always disable compilation
        }
        reload_params.update(llm_kwargs)
        
        self.llm = LLM(model=self.checkpoint_path, **reload_params)

    def zero_ablate(self, layer_number):
        """Zero ablate a single layer - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement zero_ablate method")


class QwenModelMixin:
    """Mixin for Qwen-based models"""
    
    def _ablate_layer(self, layer, layer_number):
        """Ablate a Qwen-style layer"""
        try:
            # Zero out self-attention weights
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(attn, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
            
            # Zero out MLP weights
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    proj.weight.data.zero_()
                    if hasattr(proj, 'bias') and proj.bias is not None:
                        proj.bias.data.zero_()
        
        except AttributeError as e:
            print(f"Error accessing model architecture: {e}", flush=True)
            raise


class LlamaModelMixin:
    """Mixin for Llama-based models"""
    
    def _ablate_layer(self, layer, layer_number):
        """Ablate a Llama-style layer"""
        try:
            # Zero out self-attention weights
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(attn, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
            
            # Zero out MLP weights
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
        
        except AttributeError as e:
            print(f"Error accessing model architecture: {e}", flush=True)
            raise


class Qwen7BChat(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen-7B-Chat",
            checkpoint_path=f"./ablated_model_qwen7bchat_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)
    
    def _format_prompt(self, prompt):
        """Format prompt for chat model compatibility"""
        return prompt

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.transformer.h[layer_number]  # Qwen chat uses transformer.h
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, enforce_eager=True)
        return f"Layer {layer_number} ablated successfully"


class Qwen257BBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen2.5-7B",
            checkpoint_path=f"./ablated_model_qwen257bbase_{timestamp}",
            dtype="half",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Qwen257BInstruct(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            checkpoint_path=f"./ablated_model_qwen257binstruct_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, enforce_eager=True)
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistillQwen7B(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="deepseek-ai/deepseek-R1-Distill-Qwen-7B",
            checkpoint_path=f"./ablated_model_deepseekqwen7b_{timestamp}",
            dtype="float16",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            dtype="float16",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BBase(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B",
            checkpoint_path=f"./ablated_model_llama318bbase_{timestamp}",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BInstruct(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            checkpoint_path=f"./ablated_model_llama318binstruct_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistilledLlama(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            checkpoint_path=f"./ablated_model_deepseek_r1_8b_{timestamp}",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class OpenReasonerBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
            checkpoint_path=f"./ablated_model_open_reasoner_zero_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama31SimpleRLZoo(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo",
            checkpoint_path=f"./ablated_model_llama318bsimplerl_{timestamp}",
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...", flush=True)
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            gpu_memory_utilization=0.95,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"
