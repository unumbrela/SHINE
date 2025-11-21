import os
import sys
import re

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root_path not in sys.path:
    sys.path.append(project_root_path)
if os.path.dirname(project_root_path) not in sys.path:
    sys.path.append(os.path.dirname(project_root_path))

from ..llms import AbstractLLM

# 导入 vLLM 相关库
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig


class TPCLLM(AbstractLLM):
    """
    优化后的 Qwen3-8B 本地推理类，专门针对 24GB 显存优化
    """
    def __init__(self):
        super().__init__()

        # 指定 Qwen3-8B 模型路径
        model_name = "Qwen3-8B"
        self.path = os.path.join(
            project_root_path, "chinatravel", "agent", "tpc_agent", "local_llm", model_name
        )

        # 如果 tpc_agent 目录下没有模型，回退到全局目录
        if not os.path.exists(self.path):
            self.path = os.path.join(
                project_root_path, "chinatravel", "local_llm", model_name
            )

        # 设置环境变量
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=4096
        )

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

        # 优化显存配置 - 针对 24GB 显存
        max_model_len = 8192  # 降低上下文长度，从 32768 降到 8192

        print(f"[TPCLLM] 正在加载模型: {self.path}")
        print(f"[TPCLLM] 显存优化配置: max_model_len={max_model_len}")

        self.llm = LLM(
            model=self.path,
            gpu_memory_utilization=0.85,  # 从 0.95 降到 0.75
            max_model_len=max_model_len,   # 降低到 8192
            enable_prefix_caching=False,   # 关闭前缀缓存节省显存
        )

        self.name = "TPCLLM_Qwen3-8B"
        self.max_model_len = max_model_len

        # Token 统计
        self.input_token_count = 0
        self.output_token_count = 0
        self.input_token_maxx = 0

        print(f"[TPCLLM] 模型加载完成!")

    def _get_response(self, messages, one_line, json_mode):
        """
        获取模型响应
        """
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  
        )

        # Token 计数
        input_tokens = self.tokenizer(text)["input_ids"]
        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

        # 检查输入长度
        if len(input_tokens) >= self.max_model_len:
            return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

        # 生成响应
        outputs = self.llm.generate([text], self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Token 计数
        output_token_ids = outputs[0].outputs[0].token_ids
        self.output_token_count += len(output_token_ids)

        # 解析思考内容和实际回复
        try:
            m = re.match(r"<think>\n(.+)</think>\n\n", generated_text, flags=re.DOTALL)
            content = generated_text[len(m.group(0)):]
            thinking_content = m.group(1).strip()
        except Exception as e:
            thinking_content = ""
            content = generated_text.strip()

        res_str = content

        # 处理 one_line 模式
        if one_line and not json_mode:
            res_str = res_str.split("\n")[0]

        return res_str
