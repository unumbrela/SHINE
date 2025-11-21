"""
TPC Agent 模块

这个模块是自包含的，包含运行所需的所有依赖：
- base.py: BaseAgent 基类
- llms.py: LLM 初始化（模型路径已配置为 tpc_agent/local_llm）
- load_model.py: 模型加载函数
- tpc_agent.py: UrbanTrip Agent
- tpc_agent_llm.py: UrbanTripLLM Agent
- local_llm/: 模型权重文件

使用方法：
    python run_exp.py --splits tpc_phase_2_online_test --agent UrbanTripLLM --llm Qwen3-8B
"""

__all__ = ['TPCAgent', 'UrbanTrip', 'UrbanTripLLM', 'TPCLLM', 'init_llm']
