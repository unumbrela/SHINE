# from abc import ABC, abstractmethod
# from openai import OpenAI
# from json_repair import repair_json
# from transformers import AutoTokenizer
# from transformers import AutoConfig
# import tiktoken
# from vllm import LLM, SamplingParams
# import re
# import sys
# import os

# # tpc_agent/llms.py 在 chinatravel/agent/tpc_agent/ 下
# # 需要往上 4 层到达项目根目录
# project_root_path = os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )

# if project_root_path not in sys.path:
#     sys.path.insert(0, project_root_path)

# def chat_template(messages):
#     """
#     将 messages 列表转成符合 Chat 模板格式的字符串
#     用于 tiktoken.encode 计算 token 数。
#     """
#     formatted = ""
#     for msg in messages:
#         role = msg["role"]
#         content = msg["content"]
#         formatted += f"<|{role}|>\n{content}\n"
#     formatted += "<|assistant|>\n"  # 留空表示用户希望 assistant 继续回复
#     return formatted

# def merge_repeated_role(messages):
#     ptr = len(messages) - 1
#     last_role = ""
#     while ptr >= 0:
#         cur_role = messages[ptr]["role"]
#         if cur_role == last_role:
#             messages[ptr]["content"] += "\n" + messages[ptr + 1]["content"]
#             del messages[ptr + 1]
#         last_role = cur_role
#         ptr -= 1
#     return messages


# class InsufficientBalanceError(Exception):
#     """Custom exception for API balance insufficient errors"""
#     pass


# class AbstractLLM(ABC):
#     class ModeError(Exception):
#         pass

#     def __init__(self):
#         self.input_token_count = 0
#         self.output_token_count = 0
#         self.input_token_maxx = 0
#         pass

#     def __call__(self, messages, one_line=True, json_mode=False):
#         if one_line and json_mode:
#             raise self.ModeError(
#                 "one_line and json_mode cannot be True at the same time"
#             )
#         return self._get_response(messages, one_line, json_mode)

#     def _check_and_raise_balance_error(self, exception):
#         """
#         Unified method to check if an exception is related to insufficient API balance or authentication.
#         If it is, raises InsufficientBalanceError to trigger automatic interruption.
#         """
#         error_msg = str(exception).lower()
#         exception_type = type(exception).__name__

#         # Check for OpenAI SDK NotFoundError and AuthenticationError
#         # These are strong indicators of auth issues when using OpenAI-compatible APIs
#         if exception_type in ['NotFoundError', 'AuthenticationError']:
#             llm_name = getattr(self, 'name', self.__class__.__name__)
#             print(f"\n{'='*60}")
#             print(f"⚠️  [{llm_name}] API AUTHENTICATION ERROR ({exception_type})")
#             print(f"{'='*60}")
#             print(f"Error message: {exception}")
#             print(f"\nThis usually indicates:")
#             print(f"  1. Invalid or missing API key")
#             print(f"  2. Incorrect API endpoint")
#             print(f"  3. Model not accessible with this key")
#             print(f"\nPlease check your API key configuration.")
#             print(f"{'='*60}\n")
#             raise InsufficientBalanceError(f"API authentication failed for {llm_name}: {exception}")

#         # Authentication/API key related errors - these should always stop execution
#         auth_keywords = [
#             # English
#             'api_key',
#             'api key',
#             'authentication',
#             'unauthorized',
#             'invalid api key',
#             'invalid_api_key',
#             'api key not set',
#             'must be set',
#             'invalid key',
#             'invalid credentials',
#             'authentication failed',
#             'invalid token',
#             'access denied',
#             '401',
#             '403',
#             # Chinese
#             'api密钥',
#             '无效的api',
#             '认证失败',
#             '未授权',
#         ]

#         # Balance-related errors
#         balance_keywords = [
#             # Chinese keywords
#             '余额不足',
#             '账户余额不足',
#             '余额已用尽',
#             '账户欠费',
#             # English keywords
#             'insufficient balance',
#             'insufficient_user_balance',
#             'insufficient funds',
#             'quota exceeded',
#             'rate limit exceeded',
#             'billing hard limit',
#             'account balance',
#             'insufficient quota',
#             'balance too low',
#             'credit limit',
#             # API-specific error codes
#             'insufficient_quota',
#             'quota_exceeded',
#             'billing_not_active',
#             'payment_required',
#         ]

#         # Check for authentication errors first
#         if any(keyword in error_msg for keyword in auth_keywords):
#             llm_name = getattr(self, 'name', self.__class__.__name__)
#             print(f"\n{'='*60}")
#             print(f"⚠️  [{llm_name}] API AUTHENTICATION ERROR")
#             print(f"{'='*60}")
#             print(f"Error message: {exception}")
#             print(f"\nThis error indicates an invalid or missing API key.")
#             print(f"Please check your API key configuration.")
#             print(f"{'='*60}\n")
#             raise InsufficientBalanceError(f"API authentication failed for {llm_name}: {exception}")

#         # Check for balance errors
#         if any(keyword in error_msg for keyword in balance_keywords):
#             llm_name = getattr(self, 'name', self.__class__.__name__)
#             print(f"\n{'='*60}")
#             print(f"⚠️  [{llm_name}] API BALANCE INSUFFICIENT")
#             print(f"{'='*60}")
#             print(f"Error message: {exception}")
#             print(f"{'='*60}\n")
#             raise InsufficientBalanceError(f"API balance insufficient for {llm_name}: {exception}")

#     @abstractmethod
#     def _get_response(self, messages, one_line, json_mode):
#         pass


# class GLM46(AbstractLLM):
#     def __init__(self):
#         super().__init__()
#         self.name = "GLM46"
#         api_key = os.environ.get("ZHIPUAI_API_KEY")
#         if not api_key:
#             raise ValueError("ZHIPUAI_API_KEY environment variable not set.")
#         self.llm = OpenAI(
#             api_key=api_key,
#             base_url="https://open.bigmodel.cn/api/paas/v4"
#         )

#     def _get_response(self, messages, one_line, json_mode):
#         kwargs = {
#             "model": "glm-4.6",
#             "max_tokens": 4096, # Using a safe default from the project
#             "temperature": 0,    # Using 0 for deterministic output, as is project standard
#             "thinking": {"type": "enabled"}
#         }

#         try:
#             response = self.llm.chat.completions.create(
#                 messages=messages,
#                 **kwargs
#             )
#             res_str = response.choices[0].message.content
#             res_str = res_str.strip()
#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#             elif one_line:
#                 res_str = res_str.split('\n')[0]
#         except InsufficientBalanceError:
#             # Re-raise balance errors to stop execution
#             raise
#         except Exception as e:
#             # Check for balance-related errors first
#             self._check_and_raise_balance_error(e)
#             # If not a balance error, handle as normal error
#             print(f"Error calling GLM-4.6 API: {e}")
#             res_str = '{"error": "Request failed."}'

#         return res_str

# class Deepseek(AbstractLLM):
#     def __init__(self):
#         super().__init__()
#         self.llm = OpenAI(
#             base_url="https://api.siliconflow.cn/v1/chat/completions",
#         )
#         self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "deepseek_v3_tokenizer"
#         )
#         self.name = "DeepSeek-V3"

#         self.tokenizer = AutoTokenizer.from_pretrained(self.path)

#     def _send_request(self, messages, kwargs):

#         text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         input_tokens = self.tokenizer(text)["input_ids"]

#         self.input_token_count += len(input_tokens)
#         self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
        
#         res_str = (
#             self.llm.chat.completions.create(messages=messages, **kwargs)
#             .choices[0]
#             .message.content
#         )
#         output_tokens = self.tokenizer(res_str)["input_ids"]
#         self.output_token_count += len(output_tokens)
        
#         res_str = res_str.strip()
#         return res_str

#     def _get_response(self, messages, one_line, json_mode):
#         kwargs = {
#             "model": "deepseek-chat",
#             "max_tokens": 4096,
#             "temperature": 0,
#             "top_p": 0.00000001,
#         }
#         if one_line:
#             kwargs["stop"] = ["\n"]
#         elif json_mode:
#             kwargs["response_format"] = {"type": "json_object"}
#         try:
#             res_str = self._send_request(messages, kwargs)
#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#         except InsufficientBalanceError:
#             # Re-raise balance errors to stop execution
#             raise
#         except Exception as e:
#             # Check for balance-related errors first
#             self._check_and_raise_balance_error(e)
#             # If not a balance error, handle as normal error
#             print(e)
#             res_str = '{"error": "Request failed, please try again."}'
#         return res_str


# class GLM4Plus(AbstractLLM):
#     def __init__(self):
#         super().__init__()
#         self.llm = OpenAI(
#             base_url="https://open.bigmodel.cn/api/paas/v4",
#         )
#         self.name = "GLM4Plus"

#     def _send_request(self, messages, kwargs):
#         res_str = (
#             self.llm.chat.completions.create(messages=messages, **kwargs)
#             .choices[0]
#             .message.content
#         )
#         res_str = res_str.strip()
#         return res_str

#     def _get_response(self, messages, one_line, json_mode):
#         kwargs = {
#             "model": "glm-4-plus",
#             "max_tokens": 4095,
#             "temperature": 0,
#             "top_p": 0.01,
#         }
#         if one_line:
#             kwargs["stop"] = ["<STOP>"]
#         try:
#             res_str = self._send_request(messages, kwargs)
#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#         except InsufficientBalanceError:
#             # Re-raise balance errors to stop execution
#             raise
#         except Exception as e:
#             # Check for balance-related errors first
#             self._check_and_raise_balance_error(e)
#             # If not a balance error, handle as normal error
#             res_str = '{"error": "Request failed, please try again."}'
#         return res_str


# class GPT4o(AbstractLLM):
#     def __init__(self):
#         super().__init__()
#         self.llm = OpenAI()
#         self.name = "GPT4o"
#         self.tokenizer = tiktoken.encoding_for_model("gpt-4o")


#     def _send_request(self, messages, kwargs):

#         # print(messages)
#         tokens = self.tokenizer.encode(chat_template(messages))
#         self.input_token_count += len(tokens)
#         self.input_token_maxx = max(self.input_token_maxx, len(tokens))

#         # print(tokens)
#         # print(self.input_token_count)
#         # exit(0)

#         res_str = (
#             self.llm.chat.completions.create(messages=messages, **kwargs)
#             .choices[0]
#             .message.content
#         )
        
#         tokens = self.tokenizer.encode(res_str)
#         self.output_token_count += len(tokens)

#         res_str = res_str.strip()
#         return res_str

#     def _get_response(self, messages, one_line, json_mode):
#         kwargs = {
#             "model": "chatgpt-4o-latest",
#             "max_tokens": 4095,
#             "temperature": 0,
#             "top_p": 0.01,
#         }
#         if one_line:
#             kwargs["stop"] = ["\n"]
#         elif json_mode:
#             kwargs["response_format"] = {"type": "json_object"}
#         try:
#             res_str = self._send_request(messages, kwargs)
#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#         except InsufficientBalanceError:
#             # Re-raise balance errors to stop execution
#             raise
#         except Exception as e:
#             # Check for balance-related errors first
#             self._check_and_raise_balance_error(e)
#             # If not a balance error, handle as normal error
#             print(e)
#             res_str = '{"error": "Request failed, please try again."}'
#         return res_str


		
# class Qwen(AbstractLLM):
#     def __init__(self, model_name, max_model_len=None):
#         super().__init__()
#         # 统一从 chinatravel/local_llm 目录下加载模型
#         self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", model_name
#         )
#         os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
#         if "Qwen3" in model_name:    
#             self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096)
#         else:
#             self.sampling_params = SamplingParams(temperature=0, top_p=0.001, max_tokens=4096)
#         if max_model_len is not None and max_model_len > 32768:
#             config = AutoConfig.from_pretrained(self.path)
#             config.rope_scaling = {
#                     "type": "yarn",
#                     "factor": max_model_len//32768, # 2.0,  # 原长 32,768 → 扩展到 32,768 * 2 = 65536
#                     "original_max_position_embeddings": 32768
#                 }
#             config.save_pretrained(self.path)
#             os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
#         else:
#             config = AutoConfig.from_pretrained(self.path)
#             if "rope_scaling" in config.to_dict():
#                 del config.rope_scaling
#             config.save_pretrained(self.path)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.path)
#         if max_model_len is None:
#             max_model_len = 32768

#         # 针对 24GB GPU 的优化配置
#         # Qwen3-8B 也需要优化显存设置
#         print(f"[{model_name}] 正在配置显存优化参数...")

#         # 降低 max_model_len 以减少 KV cache 占用
#         if max_model_len > 8192:
#             max_model_len = 8192
#             print(f"[{model_name}] 降低 max_model_len 到 {max_model_len} 以适配 24GB GPU")

#         self.llm = LLM(
#             model=self.path,
#             gpu_memory_utilization=0.75,  # 从 0.90 降到 0.75
#             max_model_len=max_model_len,
#             max_num_seqs=8,              # 限制并发序列数
#             enable_prefix_caching=False,  # 禁用前缀缓存以节省显存
#             enforce_eager=True,           # 使用 eager 模式,避免 CUDA graph 占用
#         )
#         self.name = model_name
#         self.max_model_len = max_model_len
        
#     def _get_response(self, messages, one_line, json_mode):
        
#         if "Qwen3" in self.name:
#             text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#                 enable_thinking=False  # 禁用thinking模式以加快推理速度
#             )
#             input_tokens = self.tokenizer(text)["input_ids"]
#             self.input_token_count += len(input_tokens)      
#             self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
            
#             if len(input_tokens) >= self.max_model_len:
#                 return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})
#             outputs = self.llm.generate([text], self.sampling_params)
#             generated_text = outputs[0].outputs[0].text
#             output_token_ids = outputs[0].outputs[0].token_ids
#             self.output_token_count += len(output_token_ids)
#             try:
#                 m = re.match(r"<think>\n(.+)</think>\n\n", generated_text, flags=re.DOTALL)
#                 content = generated_text[len(m.group(0)):]
#                 thinking_content = m.group(1).strip()
#             except Exception as e:
#                 thinking_content = ""
#                 content = generated_text.strip()
            
#             res_str = content
#         else:
#             text = self.tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
            
#             input_tokens = self.tokenizer(text)["input_ids"]
#             self.input_token_count += len(input_tokens)        
#             self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
            
#             if len(input_tokens) >= self.max_model_len:
#                 return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})
#             outputs = self.llm.generate([text], self.sampling_params)
#             res_str = outputs[0].outputs[0].text
#             output_token_ids = outputs[0].outputs[0].token_ids
#             self.output_token_count += len(output_token_ids)
#         try:
#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#             elif one_line:
#                 res_str = res_str.split("\n")[0]
#         except Exception as e:
#             res_str = '{"error": "Request with specific format failed, please try again."}'
#         return res_str


# class Mistral(AbstractLLM):
#     def __init__(self, max_model_len=None):
#         super().__init__()
#         self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "Mistral-7B-Instruct-v0.3",
#         )
#         self.sampling_params = SamplingParams(
#             temperature=0, top_p=0.001, max_tokens=4096
#         )

#         if max_model_len is not None and max_model_len > 32768:
#             config = AutoConfig.from_pretrained(self.path)
#             config.rope_scaling = {
#                 "type": "yarn", 
#                 "factor": max_model_len // 32768,
#                 "original_max_position_embeddings": 32768
#             }
#             config.save_pretrained(self.path)
#             os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
#         else:
#             config = AutoConfig.from_pretrained(self.path)
#             if "rope_scaling" in config.to_dict():
#                 del config.rope_scaling
#             config.save_pretrained(self.path)

#         self.tokenizer = AutoTokenizer.from_pretrained(self.path)

#         if max_model_len is None:
#             max_model_len = 32768

#         self.llm = LLM(
#             model=self.path,
#             gpu_memory_utilization=0.95,
#             max_model_len=max_model_len,
#             # max_num_seqs = 1,           # Limit batch size
#             # tensor_parallel_size=2,     # GPUs=2
#             enable_prefix_caching=(max_model_len>=32768),  # 可选：启用前缀缓存优化长文本
#         )
#         self.name = "Mistral-7B-Instruct-v0.3"
#         self.max_model_len = max_model_len

#     def _get_response(self, messages, one_line, json_mode):
#         messages = merge_repeated_role(messages)
#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
        
#         input_tokens = self.tokenizer(text)["input_ids"]
#         self.input_token_count += len(input_tokens)
#         self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

#         if len(input_tokens) >= self.max_model_len:
#             return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

#         # try:
#         outputs = self.llm.generate([text], self.sampling_params)
#         res_str = outputs[0].outputs[0].text
        
#         output_token_ids = outputs[0].outputs[0].token_ids
#         self.output_token_count += len(output_token_ids)
        
#         if json_mode:
#             res_str = repair_json(res_str, ensure_ascii=False)
#         elif one_line:
#             res_str = res_str.split("\n")[0]
#         # except Exception as e:
#         #     print("error: ", e)
#         #     res_str = '{"error": "Request failed, please try again."}'
#         return res_str


# class Llama(AbstractLLM):
#     def __init__(self, model_name, max_model_len=None):
#         super().__init__()


#         Llama_supported = ["Llama3-3B", "Llama3-8B", "Llama8B-TravelPlanner"]
#         if model_name not in Llama_supported:
#             raise ValueError(f"Unsupported model name: {model_name}. Supported models: {Llama_supported}")

#         if model_name == "Llama3-3B":
#             self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "Llama-3.2-3B-Instruct"
#             )
#         elif model_name == "Llama3-8B":
#             self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "Meta-Llama-3.1-8B-Instruct"
#             )
#         elif model_name == "Llama8B-TravelPlanner":
#             self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "Llama-3.1-8B-Instruct-travelplanner-SFT"
#             )

#         self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
#         self.sampling_params = SamplingParams(
#             temperature=0, top_p=0.001, max_tokens=4096
#         )

#         # Set default max_model_len if not specified
#         if max_model_len is None:
#             max_model_len = 32768  # Default to 32K to fit in 24GB GPU

#         self.max_model_len = max_model_len
#         self.llm = LLM(
#             model=self.path,
#             max_model_len=max_model_len,
#             gpu_memory_utilization=0.90
#         )
#         self.name = model_name

#     def _get_response(self, messages, one_line, json_mode):
#         # print(messages)
#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         input_tokens = self.tokenizer(text)["input_ids"]
#         self.input_token_count += len(input_tokens)
#         self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

#         if len(input_tokens) >= self.max_model_len:
#             return f'{{"error": "Input prompt is longer than {self.max_model_len} tokens."}}'
        
        
#         try:
#             outputs = self.llm.generate([text], self.sampling_params)
#             res_str = outputs[0].outputs[0].text
            
#             output_token_ids = outputs[0].outputs[0].token_ids
#             self.output_token_count += len(output_token_ids)

#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#             elif one_line:
#                 res_str = res_str.split("\n")[0]
#         except Exception as e:
#             res_str = '{"error": "Request failed, please try again."}'
#         # print("---")
#         # print(res_str)
#         # print("---")
#         print(res_str)
#         return res_str

# class TravelPlanner(AbstractLLM):
#     def __init__(self, max_model_len=None):
#         super().__init__()
#         self.path = os.path.join(
#             project_root_path, "chinatravel", "local_llm", "Llama-3.1-8B-Instruct-travelplanner-SFT"
#         )
#         self.sampling_params = SamplingParams(
#             temperature=0, top_p=0.001, max_tokens=4096
#         )

#         if max_model_len is not None and max_model_len > 32768:
#             config = AutoConfig.from_pretrained(self.path)
#             config.rope_scaling = {
#                 "type": "yarn",
#                 "factor": max_model_len // 32768,
#                 "original_max_position_embeddings": 32768
#             }
#             config.save_pretrained(self.path)
#             os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
#         else:
#             config = AutoConfig.from_pretrained(self.path)
#             if "rope_scaling" in config.to_dict():
#                 del config.rope_scaling
#             config.save_pretrained(self.path)

#         self.tokenizer = AutoTokenizer.from_pretrained(self.path)

#         if max_model_len is None:
#             max_model_len = 32768

#         self.llm = LLM(
#             model=self.path,
#             gpu_memory_utilization=0.95,
#             max_model_len=max_model_len,
#             enable_prefix_caching=(max_model_len>=32768),
#         )
#         self.name = "TravelPlanner"
#         self.max_model_len = max_model_len

#     def _get_response(self, messages, one_line, json_mode):
#         messages = merge_repeated_role(messages)
#         text = self.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         input_tokens = self.tokenizer(text)["input_ids"]
#         self.input_token_count += len(input_tokens)
#         self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

#         if len(input_tokens) >= self.max_model_len:
#             return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

#         try:
#             outputs = self.llm.generate([text], self.sampling_params)
#             res_str = outputs[0].outputs[0].text

#             output_token_ids = outputs[0].outputs[0].token_ids
#             self.output_token_count += len(output_token_ids)

#             if json_mode:
#                 res_str = repair_json(res_str, ensure_ascii=False)
#             elif one_line:
#                 res_str = res_str.split("\n")[0]
#         except Exception as e:
#             print("error: ", e)
#             res_str = '{"error": "Request failed, please try again."}'
#         return res_str


# class EmptyLLM(AbstractLLM):
#     def __init__(self):
#         super().__init__()
#         self.name = "EmptyLLM"

#     def _get_response(self, messages, one_line, json_mode):
#         return "Empty LLM response"

# if __name__ == "__main__":
#     # model = Mistral()
#     model = GPT4o()
#     print(model([{"role": "user", "content": "hello!"}], one_line=False))
