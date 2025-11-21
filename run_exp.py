import argparse

import numpy as np

import sys
import os
import json
from datetime import datetime

project_root_path = os.path.dirname(os.path.abspath(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from copy import deepcopy

from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv

# Import InsufficientBalanceError from both locations to handle different agents
try:
    from chinatravel.agent.llms import InsufficientBalanceError
except ImportError:
    pass

try:
    from chinatravel.agent.UrbanTrip.llms import InsufficientBalanceError
except ImportError:
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument(
        "--splits",
        "-s",
        type=str,
        default="easy",
        help="query subset (e.g., easy, medium, human, easy_100, easy_200, easy_300)",
    )
    parser.add_argument("--index", "-id", type=str, default=None, help="query index")
    parser.add_argument(
        "--skip", "-sk", type=int, default=0, help="skip if the plan exists"
    )
    parser.add_argument('--restart_from', type=str, default=None, help='Restart Data ID')
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        default=None,
        choices=["RuleNeSy", "LLMNeSy", "LLM-modulo", "ReAct", "ReAct0", "Act", "TPCAgent", "UrbanTrip", "MyAgent"],
    )
    parser.add_argument(
        "--llm",
        "-l",
        type=str,
        default=None,
        choices=["deepseek", "gpt-4o", "glm4-plus", "glm-4.6", "glm4.6", "Qwen2.5-72B-Instruct", "Qwen3-8B","mistral", "Llama-3.1-70B-Instruct","Llama3-3B","rule", "Llama3-8B","TPCLLM"],
    )
    
    parser.add_argument('--oracle_translation', action='store_true', help='Set this flag to enable oracle translation.')
    parser.add_argument('--preference_search', action='store_true', help='Set this flag to enable preference search.')
    parser.add_argument('--no_cache', action='store_true', help='Set this flag to disable loading translation cache and force regeneration.')
    parser.add_argument('--refine_steps', type=int, default=10, help='Steps for refine-based method, such as LLM-modulo, Reflection')
    parser.add_argument(
        '--output_dir',
        '-o',
        type=str,
        default=None,
        help='Specify output directory name (e.g., LLMNeSy_glm4-plus_oracletranslation_20251116_1430). If provided, will resume from this directory and skip completed tasks.'
    )


    args = parser.parse_args()

    print(args)

    query_index, query_data = load_query(args)
    print(len(query_index), "samples")

    if args.index is not None:
        query_index = [args.index]

    cache_dir = os.path.join(project_root_path, "cache")

    # Determine output directory name
    if args.output_dir is not None:
        # Use specified output directory (for resuming)
        method = args.output_dir
        print(f"[RESUME MODE] Using specified output directory: {method}")
    else:
        # Generate new output directory name with timestamp
        method = args.agent + "_" + args.llm
        if args.agent == "LLM-modulo":
            method += f"_{args.refine_steps}steps"

            if not args.oracle_translation:
                raise Exception("LLM-modulo must use oracle translation")

        if args.oracle_translation:
            method = method + "_oracletranslation"
        if args.preference_search:
            method = method + "_preferencesearch"

        # Add timestamp to method name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        method = method + "_" + timestamp
        print(f"[NEW RUN] Created new output directory: {method}")

    res_dir = os.path.join(
        project_root_path, "results", method
    )
    log_dir = os.path.join(
        project_root_path, "cache"
    )
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # Don't create log_dir with timestamp here - Agent will create its own log directory

    print("res_dir: ", res_dir)
    print("cache_dir:", log_dir)

    if args.agent in ["LLM-modulo"]:
        max_model_len = 65536
    elif args.agent in ["LLMNeSy"]:
        max_model_len = 8192
    else:
        max_model_len = None
    kwargs = {
        "method": args.agent,
        "env": WorldEnv(),
        "backbone_llm": init_llm(args.llm, max_model_len=max_model_len),
        "cache_dir": cache_dir,
        "debug": True,
        "refine_steps": args.refine_steps,
    }
    agent = init_agent(kwargs)


    white_list = []

    succ_count, eval_count = 0, 0
    skip_count = 0  # Count of already completed tasks

    # Check existing results for resume mode
    if args.output_dir is not None:
        existing_results = set()
        if os.path.exists(res_dir):
            existing_files = [f for f in os.listdir(res_dir) if f.endswith('.json')]
            existing_results = {f.replace('.json', '') for f in existing_files}
            print(f"[RESUME MODE] Found {len(existing_results)} completed tasks in output directory")
        else:
            print(f"[WARNING] Specified output directory does not exist: {res_dir}")

    for i, data_idx in enumerate(query_index):
        if (args.restart_from is not None) and (data_idx != args.restart_from):
            continue
        else:
            args.restart_from = None

        sys.stdout = sys.__stdout__
        print("------------------------------")
        print(
            "Process [{}/{}], Success [{}/{}], Skipped [{}]:".format(
                i + 1, len(query_index), succ_count, eval_count, skip_count
            )
        )
        print("data uid: ", data_idx)

        # Skip if result already exists (resume mode or --skip flag)
        result_file = os.path.join(res_dir, f"{data_idx}.json")
        if os.path.exists(result_file):
            if args.output_dir is not None:
                print(f"[RESUME MODE] Skipping already completed task: {data_idx}")
            elif args.skip:
                print(f"[SKIP MODE] Skipping existing result: {data_idx}")
            skip_count += 1
            continue

        if i in white_list:
            continue
        eval_count += 1
        query_i = query_data[data_idx]
        print(query_i)

        try:
            if args.agent in ["ReAct", "ReAct0", "Act"]:
                plan_log = agent(query_i["nature_language"])
                plan = plan_log["ans"]
                if isinstance(plan, str):
                    try:
                        plan = json.loads(plan)
                    except:
                        plan = {"plan": plan}
                plan["input_token_count"] = agent.backbone_llm.input_token_count
                plan["output_token_count"] = agent.backbone_llm.output_token_count
                plan["input_token_maxx"] = agent.backbone_llm.input_token_maxx
                plan["llm_time"] = round(agent.backbone_llm.llm_call_time, 2)
                log = plan_log["log"]
                save_json_file(
                    json_data=log, file_path=os.path.join(log_dir, f"{data_idx}.json")
                )
                succ = 1
            elif args.agent in ["LLM-modulo"]:

                succ, plan = agent.solve(query_i, prob_idx=data_idx, oracle_verifier=True)

            elif args.agent in ["LLMNeSy", "RuleNeSy"]:
                # Use cache by default unless --no_cache is specified
                use_cache = not args.no_cache
                succ, plan = agent.run(query_i, load_cache=use_cache, oralce_translation=args.oracle_translation, preference_search=args.preference_search)

            elif args.agent == "TPCAgent":
                succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)

            elif args.agent == "UrbanTrip":
                # Use cache by default unless --no_cache is specified
                use_cache = not args.no_cache
                # For minimal data (only nature_language), force NL2SL translation
                is_minimal_data = len(query_i.keys()) <= 2  # Only uid and nature_language
                oracle_trans = args.oracle_translation and not is_minimal_data
                if is_minimal_data:
                    print("[UrbanTrip] Detected minimal data format - forcing NL2SL translation")
                succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=oracle_trans, load_cache=use_cache)

            elif args.agent == "MyAgent":
                succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)

            if succ:
                succ_count += 1

            save_json_file(
                json_data=plan, file_path=os.path.join(res_dir, f"{data_idx}.json")
            )

        except InsufficientBalanceError as e:
            # API balance is insufficient - stop execution
            print("\n" + "="*60)
            print("⚠️  API BALANCE INSUFFICIENT - EXECUTION STOPPED")
            print("="*60)
            print(f"Stopped at sample {i+1}/{len(query_index)}: {data_idx}")
            print(f"Error: {e}")
            print(f"\nProgress Summary:")
            print(f"  - Total samples: {len(query_index)}")
            print(f"  - Evaluated: {eval_count}")
            print(f"  - Successful: {succ_count}")
            print(f"  - Skipped (already completed): {skip_count}")
            print(f"\nTo resume after adding API balance:")
            print(f"  1. Add a new API key with sufficient balance:")
            print(f"     export OPENAI_API_KEY='your_new_key'")
            print(f"  2. Resume execution from where it stopped:")
            print(f"     python run_exp.py --output_dir {method}")
            print("="*60)
            break  # Stop execution

    # Print final summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(query_index)}")
    print(f"Evaluated: {eval_count}")
    print(f"Successful: {succ_count}")
    print(f"Skipped (already completed): {skip_count}")
    print(f"Output directory: {method}")
    print("="*60)
    if skip_count > 0:
        print(f"\n[INFO] To resume this run later, use:")
        print(f"  --output_dir {method}")
    print()
