"""
TPC Agent 专用的模型加载模块
用于在提交时确保使用 tpc_agent 目录内的 llms.py
"""

def init_llm(llm_name, max_model_len=None):
    """
    初始化 LLM，优先使用 tpc_agent 目录内的 llms.py
    这样即使外部 llms.py 未修改，也能正确加载 tpc_agent/local_llm 中的模型
    """
    # 优先使用当前目录的 llms.py
    try:
        from .my_llms import Deepseek, GPT4o, GLM4Plus, Qwen, Mistral, Llama, TravelPlanner, EmptyLLM, GLM46
    except ImportError:
        # 如果当前目录没有，fallback 到父目录
        from ..llms import Deepseek, GPT4o, GLM4Plus, Qwen, Mistral, Llama, TravelPlanner, EmptyLLM, GLM46

    from .my_llm import TPCLLM

    if llm_name == "deepseek":
        llm = Deepseek()
    elif llm_name == "gpt-4o":
        llm = GPT4o()
    elif llm_name == "glm4-plus":
        llm = GLM4Plus()
    elif llm_name == "GLM46":
        llm = GLM46()
    elif "Qwen" in llm_name:
        if llm_name == "Qwen-4B":
            model_dir_name = "Qwen3-4B"
        elif llm_name == "Qwen3-4B":
            model_dir_name = "Qwen3-8B"  # 映射到实际存在的 Qwen3-8B 模型
        elif llm_name == "Qwen-14B":
            model_dir_name = "Qwen2.5-14B-Instruct"
        else:
            model_dir_name = llm_name
        llm = Qwen(model_dir_name, max_model_len=max_model_len)
    elif llm_name == "mistral":
        llm = Mistral(max_model_len=max_model_len)
    elif llm_name == "Llama8B":
        llm = Llama("Llama8B-TravelPlanner", max_model_len=max_model_len)
    elif "Llama" in llm_name:
        llm = Llama(llm_name, max_model_len=max_model_len)
    elif llm_name == "travelplanner":
        llm = TravelPlanner(max_model_len=max_model_len)
    elif llm_name == "rule":
        return EmptyLLM()
    elif llm_name == "TPCLLM":
        llm = TPCLLM()
    else:
        raise Exception("Not Implemented")

    return llm
