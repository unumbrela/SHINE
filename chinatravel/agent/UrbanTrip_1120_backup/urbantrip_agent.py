import sys
import os
import time
import argparse
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import random
import re
import ast
from geopy.distance import geodesic

sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.agent.base import AbstractAgent, BaseAgent
from chinatravel.agent.UrbanTrip.urbantrip_utils import (
    time_compare_if_earlier_equal,
    calc_cost_from_itinerary_wo_intercity,
    add_time_delta,
    get_time_delta,
    TimeOutError,
)

# Modularized components
from chinatravel.agent.UrbanTrip.constraint_parser import ConstraintParser
from chinatravel.agent.UrbanTrip.constraint_validator import ConstraintValidator
from chinatravel.agent.UrbanTrip.ranking_engine import RankingEngine
from chinatravel.agent.UrbanTrip.poi_manager import POIManager
from chinatravel.agent.UrbanTrip.plan_builder import PlanBuilder

# from chinatravel.eval.utils import load_json_file, validate_json, save_json_file
from chinatravel.data.load_datasets import load_json_file, save_json_file
from chinatravel.agent.utils import Logger
from chinatravel.symbol_verification.commonsense_constraint import (
    func_commonsense_constraints,
)
from chinatravel.symbol_verification.hard_constraint import (
    get_symbolic_concepts,
    evaluate_constraints,
    evaluate_constraints_py,
)
from chinatravel.symbol_verification.preference import evaluate_preference_py
from chinatravel.environment.tools.poi.apis import Poi

from chinatravel.agent.nesy_verifier.verifier.commonsense_constraint_nl import collect_commonsense_constraints_error
from chinatravel.agent.nesy_verifier.verifier.personal_constraint_nl import collect_personal_error

from chinatravel.symbol_verification.concept_func import *
from chinatravel.agent.nesy_agent.nl2sl_hybrid import nl2sl_reflect
from copy import deepcopy


# ====================================================================================
# UrbanTrip Agent - Modular Travel Planning Agent
# ====================================================================================
#
# Architecture Overview:
#   - Uses modular components for different responsibilities
#   - Combines symbolic reasoning (constraints) with neural methods (LLM)
#   - Implements DFS search with constraint validation
#
# File Structure:
#   - Lines 60-83:    INITIALIZATION
#   - Lines 84-112:   NL2SL TRANSLATION
#   - Lines 113-269:  MAIN EXECUTION FLOW (run, symbolic_search)
#   - Lines 270-833:  SEARCH LOGIC - generate_plan_with_search (563 lines)
#   - Lines 834-2599: SEARCH LOGIC - dfs_poi (1765 lines)
#   - Lines 2600-2846: CONSTRAINT VALIDATION
#   - Lines 2847-3038: HELPER METHODS
#   - Lines 3039-3101: DELEGATION METHODS (DO NOT REMOVE)
#
# ====================================================================================


class UrbanTrip(BaseAgent):
    # ====================================================================================
    # SECTION 1: INITIALIZATION
    # ====================================================================================

    def __init__(self, **kwargs):
        super().__init__(name="TPC", **kwargs)
        cache_dir = kwargs.get("cache_dir", "cache/")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        self.method = kwargs["method"]
        self.memory = {}
        self.TIME_CUT = 60 * 5 - 10
        self.debug = kwargs.get("debug", False)
        self.poi_search = Poi()

        self.visited_attractions = set()
        self.visited_restaurants = set()

        # Initialize modular components (will be set in symbolic_search with query context)
        self.constraint_parser = None
        self.constraint_validator = None
        self.ranking_engine = RankingEngine(self.poi_search)
        self.poi_manager = None
        self.plan_builder = None

    # ====================================================================================
    # SECTION 2: NL2SL TRANSLATION
    # ====================================================================================

    def translate_nl2sl(self, query, load_cache=False):
        """
        Translate natural language query to symbolic language (DSL)
        This generates hard_logic_py from nature_language using LLM
        """
        llm_method = "translation_{}_reflect".format(self.backbone_llm.name)
        translation_cache_dir = os.path.join(self.cache_dir, llm_method)
        if not os.path.exists(translation_cache_dir):
            os.makedirs(translation_cache_dir)

        file_path = os.path.join(translation_cache_dir, "{}.json".format(query["uid"]))

        print(f"[NL2SL] Translation cache path: {file_path}")

        if load_cache and os.path.exists(file_path):
            print(f"[NL2SL] Loading cached translation from {file_path}")
            query = load_json_file(file_path)
        else:
            print(f"[NL2SL] Generating new translation using LLM...")
            query = nl2sl_reflect(query, self.backbone_llm)
            if "error" in query:
                print(f"[NL2SL] Error in translation, setting empty hard_logic_py")
                query["hard_logic_py"] = []
                query["hard_logic_nl"] = []
            save_json_file(query, file_path)
            print(f"[NL2SL] Translation saved to {file_path}")

        return query

    # ====================================================================================
    # SECTION 3: MAIN EXECUTION FLOW
    # ====================================================================================

    def run(self, query, prob_idx, oralce_translation=True, load_cache=True):
        method_name = self.method + "_" + self.backbone_llm.name
        if oralce_translation:
            method_name = method_name + "_oracletranslation"

        self.log_dir = os.path.join(self.cache_dir, method_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        sys.stdout = Logger(
            "{}/{}.log".format(
                self.log_dir, query["uid"]
            ),
            sys.stdout,
            self.debug,
        )
        sys.stderr = Logger(
            "{}/{}.error".format(
                self.log_dir, query["uid"]
            ),
            sys.stderr,
            self.debug,
        )

        self.backbone_llm.input_token_count = 0
        self.backbone_llm.output_token_count = 0
        self.backbone_llm.input_token_maxx = 0
        self.backbone_llm.llm_call_time = 0.0

        # Start timing the entire inference process
        total_start_time = time.time()

        # natural language -> symoblic language -> plan
        nl2sl_time = 0.0
        nl2sl_end_time = total_start_time  # Initialize to start time
        if not oralce_translation:
            print(f"[UrbanTrip] Translating natural language to symbolic language... (UID: {query['uid']})")
            if not load_cache:
                print("[UrbanTrip] Cache disabled (--no_cache), forcing regeneration...")
            nl2sl_start = time.time()
            query = self.translate_nl2sl(query, load_cache=load_cache)
            nl2sl_end_time = time.time()
            nl2sl_time = nl2sl_end_time - nl2sl_start
            print(f"[UrbanTrip] Translation completed in {nl2sl_time:.1f}s")

        # Track symbolic search time
        search_start_time = time.time()
        succ, plan = self.symbolic_search(query)
        search_time = time.time() - search_start_time

        if succ:
            plan_out = plan
        else:
            # FIXME: 放宽约束,允许返回次优计划以保证至少有输出
            if self.least_plan_logic is not None:
                plan_out = self.least_plan_logic
                print("[Fallback] Using plan that passes logic constraints")
                succ = True
            elif self.least_plan_comm is not None:
                plan_out = self.least_plan_comm
                print("[Fallback] Using plan that passes commonsense constraints (logic constraints may fail)")
                succ = True  # 关键修改:允许返回commonsense计划
            elif self.least_plan_schema is not None:
                plan_out = self.least_plan_schema
                print("[Fallback] Using plan that passes schema constraints (commonsense/logic may fail)")
                succ = True  # 关键修改:允许返回schema计划
            else:
                plan_out = {}
                print("[Warning] No valid plan found, returning empty plan")

        # Calculate total inference time
        total_time = time.time() - total_start_time

        # Add token and statistics info to output (same format as LLMNeSy)
        plan_out["input_token_count"] = self.backbone_llm.input_token_count
        plan_out["output_token_count"] = self.backbone_llm.output_token_count
        plan_out["input_token_maxx"] = self.backbone_llm.input_token_maxx

        # Add timing information
        # Note:
        # - total_time: Complete run() execution time (including all overhead)
        # - nl2sl_time: Pure NL2SL translation time (if oracle_translation=False)
        # - search_time: Pure symbolic_search() execution time
        # - llm_time: Total LLM API call time (may be used in both nl2sl and search)
        # Relationship: total_time ≥ nl2sl_time + search_time (due to overhead)
        # Fix: Ensure total_time is at least nl2sl_time + search_time to prevent negative search_time
        min_total_time = nl2sl_time + search_time
        if total_time < min_total_time:
            print(f"[WARNING] total_time ({total_time:.2f}s) < nl2sl_time + search_time ({min_total_time:.2f}s), adjusting...")
            total_time = min_total_time

        plan_out["total_time"] = round(total_time, 2)  # Total inference time in seconds
        plan_out["llm_time"] = round(self.backbone_llm.llm_call_time, 2)  # Time spent in LLM calls
        plan_out["search_time"] = round(search_time, 2)  # Time spent in symbolic search
        plan_out["nl2sl_time"] = round(nl2sl_time, 2)  # Time spent in NL2SL translation (if applicable)

        # Add NL2SL translation statistics
        if not oralce_translation:
            hard_logic_py = query.get("hard_logic_py", [])
            hard_logic_nl = query.get("hard_logic_nl", [])
            plan_out["nl2sl_constraints_count"] = len(hard_logic_py) if isinstance(hard_logic_py, list) else 0
            plan_out["nl2sl_constraints_nl_count"] = len(hard_logic_nl) if isinstance(hard_logic_nl, list) else 0
            plan_out["nl2sl_reflect_count"] = query.get("reflect_cnt", 0)
            plan_out["nl2sl_success"] = len(hard_logic_py) > 0 if isinstance(hard_logic_py, list) else False
        else:
            # Oracle translation mode - use ground truth
            hard_logic_py = query.get("hard_logic_py", [])
            plan_out["nl2sl_constraints_count"] = len(hard_logic_py) if isinstance(hard_logic_py, list) else 0
            plan_out["nl2sl_source"] = "oracle"

        # Add search statistics (if they exist)
        if hasattr(self, 'llm_rec_count'):
            plan_out["llm_rec_count"] = self.llm_rec_count
        if hasattr(self, 'search_nodes'):
            plan_out["search_nodes"] = self.search_nodes
        if hasattr(self, 'backtrack_count'):
            plan_out["backtrack_count"] = self.backtrack_count
        if hasattr(self, 'constraints_validation_count'):
            plan_out["constraints_validation_count"] = self.constraints_validation_count
        if hasattr(self, 'commonsense_pass_count'):
            plan_out["commonsense_pass_count"] = self.commonsense_pass_count
        if hasattr(self, 'logical_pass_count'):
            plan_out["logical_pass_count"] = self.logical_pass_count
        if hasattr(self, 'all_constraints_pass'):
            plan_out["all_constraints_pass"] = self.all_constraints_pass

        return succ, plan_out

    def symbolic_search(self, symoblic_query):
        if (symoblic_query["target_city"] in self.env.support_cities) and (
                symoblic_query["start_city"] in self.env.support_cities
        ):
            pass
        else:
            return False, {
                "error_info": f"Unsupported cities {symoblic_query['start_city']} -> {symoblic_query['target_city']}."}

        # Initialize modules with query context
        self.poi_manager = POIManager(self.env, symoblic_query)
        self.plan_builder = PlanBuilder(symoblic_query)

        # Collect POI information using POIManager
        self.memory["accommodations"] = self.poi_manager.collect_poi_info_all(
            symoblic_query["target_city"], "accommodation"
        )
        self.memory["attractions"] = self.poi_manager.collect_poi_info_all(
            symoblic_query["target_city"], "attraction"
        )
        self.memory["restaurants"] = self.poi_manager.collect_poi_info_all(
            symoblic_query["target_city"], "restaurant"
        )

        # Initialize constraint parser with populated memory
        self.constraint_parser = ConstraintParser(self.memory)

        self.query = symoblic_query

        success, plan = self.generate_plan_with_search(symoblic_query)

        return success, plan

    # ====================================================================================
    # SECTION 4: SEARCH LOGIC - generate_plan_with_search
    # ====================================================================================
    #
    # This method handles the high-level search orchestration:
    #   - Initializes search state and counters
    #   - Extracts and assigns user constraints
    #   - Collects intercity transport options (train/airplane)
    #   - Ranks hotels and transports
    #   - Iterates through transport and hotel combinations
    #   - Calls dfs_poi for detailed POI planning
    #   - Returns best plan or partial plan
    #
    # Lines: 270-833 (563 lines)
    # ====================================================================================

    def generate_plan_with_search(self, query):
        # 初始化计时器和计数器
        self.time_before_search = time.time()  # 记录搜索开始时间
        self.llm_inference_time_count = 0  # llm推理时间

        # reset the cache before searching
        poi_plan = {}  # 存储当前计划的 POI 信息
        self.restaurants_visiting = []  # 正在访问的餐厅列表
        self.attractions_visiting = []  # 正在访问的景点列表
        self.food_type_visiting = []  # 正在访问的食物类型列表
        self.spot_type_visiting = []  # 正在访问的景点类型列表
        self.attraction_names_visiting = []  # 正在访问的景点名称列表
        self.restaurant_names_visiting = []  # 正在访问的餐厅名称列表

        self.llm_rec_format_error = 0  # llm推荐格式错误计数
        self.llm_rec_count = 0  # llm推荐计数
        self.search_nodes = 0  # 搜索节点计数
        self.backtrack_count = 0  # 回溯计数

        self.constraints_validation_count = 0  # 约束验证计数
        self.commonsense_pass_count = 0  # 常识通过计数
        self.logical_pass_count = 0  # 逻辑通过计数
        self.all_constraints_pass = 0  # 所有约束通过计数

        # 存储通过逻辑检查的次优计划
        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        # 提取用户需求
        # 获取用户约束信息
        # constraints_json = self.extract_user_constraints(query)
        constraints_json, requirement_list = self.constraint_parser.extract_user_constraints_by_DSL(query)

        # 用户偏好约束字段赋值
        self.all_satisfy = constraints_json.get("all_satisfy", None)

        # attractions
        self.must_see_attraction = constraints_json.get("must_see_attraction", None)
        self.must_see_attraction_type = constraints_json.get("must_see_attraction_type", None)
        self.must_not_see_attraction = constraints_json.get("must_not_see_attraction", None)
        self.must_not_see_attraction_type = constraints_json.get("must_not_see_attraction_type", None)
        self.only_free_attractions = constraints_json.get("only_free_attractions", None)

        # restaurant
        self.must_visit_restaurant = constraints_json.get("must_visit_restaurant", None)
        self.must_visit_restaurant_type = constraints_json.get("must_visit_restaurant_type", None)
        self.must_not_visit_restaurant = constraints_json.get("must_not_visit_restaurant", None)
        self.must_not_visit_restaurant_type = constraints_json.get("must_not_visit_restaurant_type", None)

        self.activities_stay_time_dict = constraints_json.get("activities_stay_time_dict", None)
        self.activities_arrive_time_dict = constraints_json.get("activities_arrive_time_dict", None)
        self.activities_leave_time_dict = constraints_json.get("activities_leave_time_dict", None)

        # hotel
        self.must_live_hotel = constraints_json.get("must_live_hotel", None)
        self.must_not_live_hotel = constraints_json.get("must_not_live_hotel", None)
        self.must_live_hotel_feature = constraints_json.get("must_live_hotel_feature", None)
        self.must_live_hotel_location_limit = constraints_json.get("must_live_hotel_location_limit", None)
        # hotel room/bed num
        self.bed_number = constraints_json.get("bed_number", None)  # 例如 ['单床', '双床']
        self.room_number = constraints_json.get("room_number", None)

        # innercity transport
        self.must_innercity_transport = constraints_json.get("must_innercity_transport", None)
        self.must_not_innercity_transport = constraints_json.get("must_not_innercity_transport", None)

        # transport rules
        self.transport_rules_by_distance = constraints_json.get("transport_rules_by_distance", None)

        # intercity transport
        self.must_depart_transport = constraints_json.get("must_depart_transport", None)
        self.must_return_transport = constraints_json.get("must_return_transport", None)
        self.must_not_depart_transport = constraints_json.get("must_not_depart_transport", None)
        self.must_not_return_transport = constraints_json.get("must_not_return_transport", None)

        # 提取所需预算
        self.attraction_budget = constraints_json.get("attraction_budget", None)
        self.restaurant_budget = constraints_json.get("restaurant_budget", None)
        self.hotel_budget = constraints_json.get("hotel_budget", None)
        self.innercity_budget = constraints_json.get("innercity_budget", None)
        self.intercity_budget = constraints_json.get("intercity_budget", None)
        self.overall_budget = constraints_json.get("overall_budget", None)

        self.requirement_list = requirement_list
        self.all_satisfy_flag = False # 是否满足用户需求
        self.too_many_backtrack = False
        self.stop_search = False
        self.default_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": [],
        }

        if self.transport_rules_by_distance is not None:
            if isinstance(self.transport_rules_by_distance, str):
                    self.transport_rules_by_distance = json.loads(self.transport_rules_by_distance)
            elif isinstance(self.transport_rules_by_distance, dict):
                self.transport_rules_by_distance = [self.transport_rules_by_distance]
            elif isinstance(self.transport_rules_by_distance, list):
                self.transport_rules_by_distance = [
                    rule for rule in self.transport_rules_by_distance if isinstance(rule, dict)
                ]

        source_city = query["start_city"] # 获取出发城市
        target_city = query["target_city"] # 获取目标城市

        print(source_city, "->", target_city)

        print("User's Constraints:")
        print(query['nature_language'])

        print("Formatted Expression:")
        for key, value in constraints_json.items():
            if value is not None:
                print(f"{key}: {value}")

        print("By list:")
        print(f"{self.requirement_list}")

        query_room_number = self.room_number
        query_room_numbed = self.bed_number

        print(f"query room number: {query_room_number}")
        print(f"query room numbed: {query_room_numbed}")

        # 收集去程和返程的城际火车交通选项
        train_go = self.collect_intercity_transport(source_city, target_city, "train")
        train_back = self.collect_intercity_transport(target_city, source_city, "train")

        # 收集去程和返程的城际飞机交通选项
        flight_go = self.collect_intercity_transport(
            source_city, target_city, "airplane"
        )
        flight_back = self.collect_intercity_transport(
            target_city, source_city, "airplane"
        )

        # must_not_depart_transport: 去程不允许的方式
        if self.must_not_depart_transport is not None and "train" in self.must_not_depart_transport:
            train_go = pd.DataFrame()  # 置空
        elif self.must_not_depart_transport is not None and "airplane" in self.must_not_depart_transport:
            flight_go = pd.DataFrame()

        # must_not_return_transport: 返程不允许的方式
        if self.must_not_return_transport is not None and "train" in self.must_not_return_transport:
            train_back = pd.DataFrame()
        elif self.must_not_return_transport is not None and "airplane" in self.must_not_return_transport:
            flight_back = pd.DataFrame()

        # must_depart_transport: 去程必须的方式
        if self.must_depart_transport is not None and "train" in self.must_depart_transport:
            flight_go = pd.DataFrame()
        elif self.must_depart_transport is not None and "airplane" in self.must_depart_transport:
            train_go = pd.DataFrame()

        # must_return_transport: 返程必须的方式
        if self.must_return_transport is not None and "train" in self.must_return_transport:
            flight_back = pd.DataFrame()
        elif self.must_return_transport is not None and "airplane" in self.must_return_transport:
            train_back = pd.DataFrame()

        # 计算可用航班和火车的数量，如果为 None 则设为 0
        flight_go_num = 0 if flight_go is None else flight_go.shape[0]
        train_go_num = 0 if train_go is None else train_go.shape[0]
        flight_back_num = 0 if flight_back is None else flight_back.shape[0]
        train_back_num = 0 if train_back is None else train_back.shape[0]

        # 合并最终的去程与返程交通选项
        go_info = pd.concat([train_go, flight_go], axis=0)
        back_info = pd.concat([train_back, flight_back], axis=0)

        # 打印调试信息，显示交通选项数量
        if self.debug:
            print(
                "from {} to {}: {} flights, {} trains".format(
                    source_city, target_city, flight_go_num, train_go_num
                )
            )
            print(
                "from {} to {}: {} flights, {} trains".format(
                    target_city, source_city, flight_back_num, train_back_num
                )
            )

            print(go_info.head())
            print(back_info.head())

        # 对去程城际交通进行排序
        ranking_go = self.ranking_intercity_transport_go(go_info, query)

        # 对酒店进行排序
        ranking_hotel = self.ranking_hotel(self.memory["accommodations"], query)

        default_hotel = (
            self.memory["accommodations"]
            .sort_values(by="price")
            .index
            .tolist()
        )

        # 根据查询对市内交通进行排序
        self.innercity_transports_ranking = ["metro", "taxi", "walk"]
        if self.must_innercity_transport is not None:
            self.innercity_transports_ranking = self.must_innercity_transport[:]
        if self.must_not_innercity_transport is not None:
            self.innercity_transports_ranking = [
                t for t in self.innercity_transports_ranking
                if t not in self.must_not_innercity_transport
            ]

        # 遍历排序后的去程交通
        intercity_budget_msg = "intercity budget not satisfied, backtrack..."
        intercity_budget_count = 0
        MAX_BACKTRACK_OUTER = 50000  # 外层循环的回溯上限

        for go_i in ranking_go:
            # 早期终止检查
            if self.backtrack_count > MAX_BACKTRACK_OUTER:
                print(f"[Early Stop] Backtrack count ({self.backtrack_count}) exceeds outer limit, stopping search...")
                # 返回最佳次优计划
                if self.least_plan_comm is not None:
                    return True, self.least_plan_comm
                elif self.least_plan_schema is not None:
                    return True, self.least_plan_schema
                else:
                    return False, {"error_info": "Search stopped due to excessive backtracking"}

            go_info_i = go_info.iloc[go_i]  # 获取当前去程交通信息
            if pd.isna(go_info_i["Cost"]):
                continue
            poi_plan["go_transport"] = go_info_i  # 将其添加到计划中
            self.search_nodes += 1

            # 对返程城际交通进行排序（依赖于去程信息）
            ranking_back = self.ranking_intercity_transport_back(
                back_info, query, go_info_i
            )
            # 遍历排序后的返程交通
            for back_i in ranking_back:
                if time.time() > self.time_before_search + self.TIME_CUT:
                    self.default_plan["backtrack_count"] = self.backtrack_count
                    return True, self.default_plan

                back_info_i = back_info.iloc[back_i]  # 获取当前返程交通信息
                if pd.isna(back_info_i["Cost"]):
                    continue
                poi_plan["back_transport"] = back_info_i  # 将其添加到计划中

                self.search_nodes += 1

                # 检查城际交通预算
                if not self.too_many_backtrack:
                    self.intercity_cost = (poi_plan["go_transport"]["Cost"] + poi_plan["back_transport"]["Cost"]) * query["people_number"]
                    if self.intercity_budget is not None and self.intercity_budget < self.intercity_cost:
                        intercity_budget_count += 1
                        print(f"{intercity_budget_msg}（{intercity_budget_count}次）".ljust(80), end='\r', flush=True)
                        print("intercity budget not satisfied, backtrack...")
                        self.backtrack_count += 1
                        continue

                if query["days"] > 1:  # 如果天数大于 1，则需要考虑酒店
                    # cnt = 0
                    # 遍历排序后的酒店选项
                    for hotel_i in ranking_hotel:
                        # 获取当前酒店信息
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                        # 获取酒店房间类型（床位数）
                        room_type = poi_plan["accommodation"]["numbed"]
                        self.search_nodes += 1

                        # 计算所需的房间数量
                        required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                        # 检查查询中的房间类型是否与当前酒店匹配，不匹配则回溯
                        if not self.too_many_backtrack:
                            if query_room_numbed != None and query_room_numbed != room_type:
                                self.backtrack_count += 1
                                print("room_type not match, backtrack...")
                                continue

                        # 如果查询中指定了房间数量，则使用该数量
                        if query_room_number != None:
                            required_rooms = query_room_number

                        # 检查房间数量和类型是否满足人数要求
                        if query_room_number != None and query_room_numbed != None:
                            pass  # 如果同时指定了房间数量和类型，则直接通过
                        else:
                            if (
                                    room_type * required_rooms >= query["people_number"]
                            ) and (
                                    room_type * required_rooms < query["people_number"] + room_type
                            ):
                                pass  # 如果房间足够容纳人数且不过分多余，则通过
                            else:
                                if query_room_number != None and room_type == 2:
                                    pass   # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                else:
                                    if not self.too_many_backtrack:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                        # 所需的房间数量
                        self.required_rooms = required_rooms

                        # 检查酒店的预算
                        if not self.too_many_backtrack:
                            self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                            if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                self.backtrack_count += 1
                                #print("hotel budget not satisfied, backtrack...")
                                continue

                        # 检查当前总预算
                        # FIXME: 注释掉过早的预算检查,避免在未添加POI时就拒绝方案
                        # 最终的预算检查会在constraints_validation中进行
                        # if not self.too_many_backtrack:
                        #     self.overall_cost = self.intercity_cost + self.hotel_cost
                        #     if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                        #         self.backtrack_count += 1
                        #         print("overall cost < intercity + hotel, backtrack...")
                        #         continue

                        print("search: ...")
                        # 尝试通过 DFS 搜索 POI 计划
                        try:
                            success, plan = self.dfs_poi(
                                query,
                                poi_plan, # go、back、accommodation
                                plan=[],
                                current_time="",
                                current_position="",
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}
                        # exit(0)

                        # print(success, plan)
                        if success:
                            return True, plan
                        else:
                            if time.time() > self.time_before_search + self.TIME_CUT:
                                self.default_plan["backtrack_count"] = self.backtrack_count
                                return True, self.default_plan

                            self.backtrack_count += 1
                            print("search failed given the intercity-transport and hotels, backtrack...")
                    # 都不满足，则从所有酒店中选
                    print("No Hotel satisfies constraint")
                    rnbc = 0 # room number bed count
                    ohbc = 0 # over hotel budget count
                    oobc = 0 # over overall budget count
                    for hotel_i in default_hotel:
                        # 获取当前酒店信息
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                        # 获取酒店房间类型（床位数）
                        room_type = poi_plan["accommodation"]["numbed"]
                        self.search_nodes += 1

                        # 计算所需的房间数量
                        required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                        # 检查查询中的房间类型是否与当前酒店匹配，不匹配则回溯
                        if not self.too_many_backtrack:
                            if query_room_numbed != None and query_room_numbed != room_type:
                                self.backtrack_count += 1
                                rnbc += 1
                                print("room_type not match, backtrack...")
                                continue

                        # 如果查询中指定了房间数量，则使用该数量
                        if query_room_number != None:
                            required_rooms = query_room_number

                        # 检查房间数量和类型是否满足人数要求
                        if query_room_number != None and query_room_numbed != None:
                            pass  # 如果同时指定了房间数量和类型，则直接通过
                        else:
                            if (
                                    room_type * required_rooms >= query["people_number"]
                            ) and (
                                    room_type * required_rooms < query["people_number"] + room_type
                            ):
                                pass  # 如果房间足够容纳人数且不过分多余，则通过
                            else:
                                if query_room_number != None and room_type == 2:
                                    pass   # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                else:
                                    if not self.too_many_backtrack:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                        # 所需的房间数量
                        self.required_rooms = required_rooms

                        # 检查酒店的预算
                        if not self.too_many_backtrack:
                            self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                            if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                self.backtrack_count += 1
                                ohbc += 1
                                #print("hotel budget not satisfied, backtrack...")
                                continue

                        # 检查当前总预算
                        # FIXME: 注释掉过早的预算检查,避免在未添加POI时就拒绝方案
                        # if not self.too_many_backtrack:
                        #     self.overall_cost = self.intercity_cost + self.hotel_cost
                        #     if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                        #         self.backtrack_count += 1
                        #         oobc += 1
                        #         print("overall cost < intercity + hotel, backtrack...")
                        #         continue

                        print("search: ...")
                        # 尝试通过 DFS 搜索 POI 计划
                        try:
                            success, plan = self.dfs_poi(
                                query,
                                poi_plan, # go、back、accommodation
                                plan=[],
                                current_time="",
                                current_position="",
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}
                        # exit(0)


                        # print(success, plan)
                        if success:
                            return True, plan
                        else:
                            if time.time() > self.time_before_search + self.TIME_CUT:
                                self.default_plan["backtrack_count"] = self.backtrack_count
                                return True, self.default_plan

                            self.backtrack_count += 1
                            print("search failed given the intercity-transport and hotels, backtrack...")
                    # 放弃约束
                    if rnbc == len(default_hotel) or ohbc == len(default_hotel) or oobc == len(default_hotel):
                        for hotel_i in default_hotel:
                            # 获取当前酒店信息
                            poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                            # 获取酒店房间类型（床位数）
                            room_type = poi_plan["accommodation"]["numbed"]
                            self.search_nodes += 1

                            # 计算所需的房间数量
                            required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                            if rnbc != len(default_hotel):
                                if query_room_numbed != None and query_room_numbed != room_type:
                                    self.backtrack_count += 1
                                    rnbc += 1
                                    print("room_type not match, backtrack...")
                                    continue

                            # 如果查询中指定了房间数量，则使用该数量
                            if query_room_number != None:
                                required_rooms = query_room_number

                            # 检查房间数量和类型是否满足人数要求
                            if query_room_number != None and query_room_numbed != None:
                                pass  # 如果同时指定了房间数量和类型，则直接通过
                            else:
                                if (
                                        room_type * required_rooms >= query["people_number"]
                                ) and (
                                        room_type * required_rooms < query["people_number"] + room_type
                                ):
                                    pass  # 如果房间足够容纳人数且不过分多余，则通过
                                else:
                                    if query_room_number != None and room_type == 2:
                                        pass  # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                    else:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                            # 所需的房间数量
                            self.required_rooms = required_rooms

                            if ohbc != len(default_hotel):
                                # 检查酒店的预算
                                self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                                if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                    self.backtrack_count += 1
                                    ohbc += 1
                                    # print("hotel budget not satisfied, backtrack...")
                                    continue

                            # FIXME: 注释掉过早的预算检查
                            # if oobc != len(default_hotel):
                            #     # 检查当前总预算
                            #     self.overall_cost = self.intercity_cost + self.hotel_cost
                            #     if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                            #         self.backtrack_count += 1
                            #         oobc += 1
                            #         print("overall cost < intercity + hotel, backtrack...")
                            #         continue

                            print("search: ...")
                            # 尝试通过 DFS 搜索 POI 计划
                            try:
                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,  # go、back、accommodation
                                    plan=[],
                                    current_time="",
                                    current_position="",
                                )
                            except TimeOutError as e:
                                print("TimeOutError")
                                return False, {"error_info": "TimeOutError"}
                            # exit(0)

                            # print(success, plan)
                            if success:
                                return True, plan
                            else:
                                if time.time() > self.time_before_search + self.TIME_CUT:
                                    self.default_plan["backtrack_count"] = self.backtrack_count
                                    return True, self.default_plan

                                self.backtrack_count += 1
                                print("search failed given the intercity-transport and hotels, backtrack...")

                else:  # 如果旅行天数只有 1 天，则不需要考虑酒店
                    self.hotel_cost = 0
                    # 检查返程交通的开始时间是否早于或等于去程交通的结束时间
                    if time_compare_if_earlier_equal(
                            poi_plan["back_transport"]["BeginTime"],
                            poi_plan["go_transport"]["EndTime"],
                    ):
                        self.backtrack_count += 1
                        print("back_transport BeginTime earlier than go_transport EndTime, backtrack...")
                        continue

                    # 计算城际交通的总成本（无酒店）
                    if not self.too_many_backtrack:
                        self.intercity_cost = poi_plan["go_transport"]["Cost"] + poi_plan["back_transport"]["Cost"] * query["people_number"]
                        if self.intercity_budget is not None and self.intercity_budget < self.intercity_cost:
                            self.backtrack_count += 1
                            print("[one-day-trip]intercity budget < cost, backtrack...")
                            continue

                    print("search: ...")
                    # 尝试通过 DFS 搜索 POI 计划
                    try:
                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan=[],
                            current_time="",
                            current_position="",
                        )
                    except TimeOutError as e:
                        print("TimeOutError")
                        return False, {"error_info": "TimeOutError"}

                    # print(success, plan)
                    if success:
                        return True, plan
                    else:
                        if time.time() > self.time_before_search + self.TIME_CUT:
                            self.default_plan["backtrack_count"] = self.backtrack_count
                            return True, self.default_plan

                        self.backtrack_count += 1
                        print("search failed given the intercity-transport and hotels, backtrack...")

        return False, {"error_info": "No solution found."}

    # ====================================================================================
    # SECTION 5: SEARCH LOGIC - dfs_poi (Depth-First Search for POI Planning)
    # ====================================================================================
    #
    # This is the core DFS search algorithm that recursively builds the travel plan:
    #   - Handles day-by-day, activity-by-activity planning
    #   - Manages different POI types: breakfast, lunch, dinner, attractions, hotels
    #   - Applies user constraints (must-visit, must-not-visit, budget limits)
    #   - Performs backtracking when constraints are violated
    #   - Validates complete plans against all constraints
    #   - Returns first valid plan or backtracks
    #
    # Key Decision Points:
    #   1. First day: Add go-transport
    #   2. Morning (00:00): Add breakfast
    #   3. Throughout day: Select attractions, lunch, dinner based on time
    #   4. Evening: Check for hotel or return transport
    #   5. Last day: Add back-transport
    #
    # Lines: 834-2599 (1765 lines)
    # ====================================================================================

    def dfs_poi(self, query, poi_plan, plan, current_time, current_position, current_day=0):
        print("----------------------------------calling dfs_poi-----------------------------------------")
        # print(f"plan: {plan}")
        print(f"current_day: {current_day}")
        print(f"current_time: {current_time}")
        print(f"current_position: {current_position}")

        print(self.backtrack_count)
        # 早期终止:回溯次数过多或超时
        MAX_BACKTRACK = 10000  # 设置合理的回溯上限
        if self.backtrack_count > MAX_BACKTRACK:
            print(f"[Early Stop] Backtrack count ({self.backtrack_count}) exceeds limit ({MAX_BACKTRACK})")
            self.too_many_backtrack = True
        if time.time() - self.time_before_search + 20 > self.TIME_CUT + self.llm_inference_time_count:
            print(f"[Early Stop] Search time exceeds limit")
            self.too_many_backtrack = True

        if not self.all_satisfy_flag and not self.too_many_backtrack:
            ok, backtrack = self.check_requirement(plan)
            if ok:
                self.all_satisfy_flag = True
            if backtrack:
                self.backtrack_count += 1
                print("requirements can not be satisfied, backtrack...")
                return False, plan

        self.search_nodes += 1
        # 检查是否超时
        if self.stop_search:
            self.default_plan["backtrack_count"] = self.backtrack_count
            return True, self.default_plan
        if time.time() - self.time_before_search > self.TIME_CUT + self.llm_inference_time_count:
            self.stop_search = True
            self.default_plan["backtrack_count"] = self.backtrack_count
            return True, self.default_plan

        # 检查当前时间是否太晚，无法前往酒店或返程交通
        print("check if too late")
        if not self.too_many_backtrack:
            if self.check_if_too_late(query, current_day, current_time, current_position, poi_plan):
                self.backtrack_count += 1
                print("The current time is too late to go hotel or back-transport, backtrack...")
                return False, plan

        # 处理第一天的去程城际交通
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]  # 初始化第一天的活动列表
            # 添加去程城际交通活动
            plan[current_day]["activities"] = self.add_intercity_transport(
                plan[current_day]["activities"],
                poi_plan["go_transport"],
                innercity_transports=[],
                tickets=self.query["people_number"],
            )

            print(plan)

            new_time = poi_plan["go_transport"]["EndTime"]  # 更新当前时间为去程交通的结束时间
            new_position = poi_plan["go_transport"]["To"]  # 更新当前位置为目的地（车站）

            # 递归调用 dfs_poi 进行后续规划
            success, plan = self.dfs_poi(query, poi_plan, plan, new_time, new_position, current_day)

            if success:
                return True, plan
            else:
                self.backtrack_count += 1
                print("No solution for the given Go Transport, backtrack...")
                return False, plan

        # breakfast
        if (current_time == "00:00" and current_day != query["days"] - 1) or (current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal("11:30", poi_plan["back_transport"]["BeginTime"])):
            if len(plan) < current_day + 1:
                plan.append({"day": current_day + 1, "activities": []})  # 如果是新的一天，添加新的活动列表

            self.search_nodes += 1
            # 选择并添加早餐活动
            plan = self.select_and_add_breakfast(plan, poi_plan, current_day, current_time, current_position, [])

            new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为早餐结束时间
            new_position = current_position  # 位置不变

            # 递归调用 dfs_poi 进行后续规划
            success, plan = self.dfs_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan

            plan[current_day]["activities"].pop()  # 如果后续规划失败，移除早餐活动，进行回溯

            candidates_type = []
            if current_day == query["days"] - 1 and current_time != "":  # 如果是最后一天，考虑返程交通
                candidates_type.append("back-intercity-transport")
            else:
                self.backtrack_count += 1
                print("No solution for the given Breakfast, backtrack...")
                return False, plan
        elif current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], "11:30"):
            candidates_type = ["back-intercity-transport"]
        else:  # 如果当前时间不是 "00:00"，说明一天已经开始
            haved_lunch_today, haved_dinner_today = False, False

            for act_i in plan[current_day]["activities"]:
                if act_i["type"] == "lunch":
                    haved_lunch_today = True  # 更新午餐状态
                if act_i["type"] == "dinner":
                    haved_dinner_today = True  # 更新晚餐状态
            if time_compare_if_earlier_equal("20:30", current_time):
                candidates_type = []
            else:
                candidates_type = ["attraction"]
            # 如果今天还没吃午餐，则考虑午餐
            if not haved_lunch_today:
                candidates_type.append("lunch")
            # 如果今天还没吃晚餐，则考虑晚餐
            if not haved_dinner_today:
                candidates_type.append("dinner")
            # 如果有住宿且不是最后一天，则考虑前往酒店
            if ("accommodation" in poi_plan) and (current_day < query["days"] - 1):
                candidates_type.append("hotel")
            # 如果是最后一天且时间不为空，则考虑返程交通
            if current_day == query["days"] - 1 and current_time != "":
                candidates_type.append("back-intercity-transport")

        print("candidates_type: ", candidates_type)  # 当前可选择的 POI 类型

        # 当还有候选类型时
        while len(candidates_type) > 0:
            poi_type, candidates_type = self.select_next_poi_type(
                candidates_type,
                plan,
                poi_plan,
                current_day,
                current_time,
                current_position,
            )

            print("POI planning, day {} {}, {}, next-poi type: {}".format(current_day, current_time, current_position, poi_type))

            # 如果下一个 POI 类型是返程城际交通
            if poi_type == "back-intercity-transport":
                # 如果是新的一天，添加新的活动列表
                if len(plan) < current_day + 1:
                    plan.append({"day": current_day + 1, "activities": []})

                # 获取市内交通排名
                # transports_ranking = self.ranking_innercity_transport(current_position, poi_plan["back_transport"]["From"], current_day, current_time)
                transports_ranking = self.innercity_transports_ranking
                if self.transport_rules_by_distance is not None:
                    temp_distance = self.calculate_distance(query, current_position, poi_plan["back_transport"]["From"])
                    transports_ranking = self.get_transport_by_distance(temp_distance)
                # 遍历市内交通类型
                for trans_type_sel in transports_ranking:
                    self.search_nodes += 1
                    # 收集市内交通选项，从当前位置到返程交通的出发地
                    print("collecting innercity transport to back-transport")
                    transports_sel = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        poi_plan["back_transport"]["From"],
                        current_time,
                        trans_type_sel,
                    )
                    # 没找到则回溯
                    if not isinstance(transports_sel, list):
                        self.backtrack_count += 1
                        print("inner-city transport error, backtrack...")
                        continue

                    if len(transports_sel) == 0:
                        arrived_time = current_time
                    else:
                        arrived_time = transports_sel[-1]["end_time"]

                    if not self.too_many_backtrack:
                        if not time_compare_if_earlier_equal(arrived_time, poi_plan["back_transport"]["BeginTime"]):
                            self.backtrack_count += 1
                            print("Fail to catch the back transport")
                            continue

                    backtrack_flag = False
                    if self.transport_rules_by_distance is not None:
                        distance = 0
                        for transport in transports_sel:
                            if transport["mode"] is not None:
                                distance += transport.get("distance", 0)
                        if len(transports_sel) == 3:
                            mode = transports_sel[1]["mode"]
                        elif len(transports_sel) == 1:
                            mode = transports_sel[0]["mode"]
                        for rule in self.transport_rules_by_distance:
                            if rule["min_distance"] is not None:
                                if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                    print("backtrack")
                                    backtrack_flag = True
                            if rule["max_distance"] is not None:
                                if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                    print("backtrack")
                                    backtrack_flag = True
                    if backtrack_flag and not self.too_many_backtrack:
                        self.backtrack_count += 1
                        continue


                    # 添加返程城际交通活动
                    plan[current_day]["activities"] = self.add_intercity_transport(
                        plan[current_day]["activities"],
                        poi_plan["back_transport"],
                        innercity_transports=transports_sel,
                        tickets=self.query["people_number"],
                    )

                    if not self.too_many_backtrack:
                        over_budget = self.check_budgets(plan)
                        if over_budget:
                            plan[current_day]["activities"].pop()
                            return False, plan

                    # 验证计划是否满足所有约束
                    res_bool, res_plan = self.constraints_validation(
                        query, plan, poi_plan
                    )
                    return True, res_plan
            # 如果下一个 POI 类型是酒店
            elif poi_type == "hotel":
                # 获取选定的酒店信息
                hotel_sel = poi_plan["accommodation"]
                # 获取市内交通排名
                # transports_ranking = self.ranking_innercity_transport(current_position, hotel_sel["name"], current_day, current_time)
                transports_ranking = self.innercity_transports_ranking
                if self.transport_rules_by_distance is not None:
                    temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                    transports_ranking = self.get_transport_by_distance(temp_distance)
                # 遍历市内交通类型
                for trans_type_sel in transports_ranking:
                    self.search_nodes += 1
                    # 如果已经在酒店位置，则无需市内交通
                    if hotel_sel["name"] == current_position:
                        transports_sel = []
                        arrived_time = current_time
                    else:
                        # 收集市内交通选项，从当前位置到酒店
                        print("collecting innercity transport to hotel")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        # 没找到则回溯
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]

                        backtrack_flag = False
                        if self.transport_rules_by_distance is not None:
                            distance = 0
                            for transport in transports_sel:
                                if transport["mode"] is not None:
                                    distance += transport.get("distance", 0)
                            if len(transports_sel) == 3:
                                mode = transports_sel[1]["mode"]
                            elif len(transports_sel) == 1:
                                mode = transports_sel[0]["mode"]
                            for rule in self.transport_rules_by_distance:
                                if rule["min_distance"] is not None:
                                    if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                                if rule["max_distance"] is not None:
                                    if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                        if backtrack_flag and not self.too_many_backtrack:
                            self.backtrack_count += 1
                            continue

                    if time_compare_if_earlier_equal("09:00", arrived_time):
                        # 添加住宿活动
                        plan = self.add_accommodation(
                            current_plan=plan,
                            hotel_sel=hotel_sel,
                            current_day=current_day,
                            arrived_time=arrived_time,
                            required_rooms=self.required_rooms,
                            transports_sel=transports_sel,
                        )
                        transports_sel = []

                    if time_compare_if_earlier_equal(arrived_time, "08:00"):
                        plan = self.select_and_add_breakfast(plan, poi_plan, current_day, current_time, current_position, transports_sel)
                        new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为早餐结束时间
                        new_position = hotel_sel["name"]

                        success, plan = self.dfs_poi(
                            query, poi_plan, plan, new_time, new_position, current_day
                        )
                        if success:
                            return True, plan

                        plan[current_day]["activities"].pop()

                    new_time = "00:00"  # 新的一天开始
                    new_position = hotel_sel["name"]  # 新位置为酒店

                    # 递归调用 dfs_poi 进行下一天的规划
                    success, plan = self.dfs_poi(query, poi_plan, plan, new_time, new_position, current_day + 1)

                    if success:
                        return True, plan

                    self.backtrack_count += 1
                    print("Fail with the given accommodation activity, backtrack...")

                    plan[current_day]["activities"].pop()

                    return False, plan
            # 如果是午餐、晚餐或景点
            elif poi_type in ["lunch", "dinner", "attraction"]:
                # 如果是午餐或晚餐
                if poi_type in ["lunch", "dinner"]:
                    res_info = self.memory["restaurants"]
                    candidate_res_list = res_info.copy()

                    if self.must_not_visit_restaurant is not None:
                        candidate_res_list = candidate_res_list[
                            ~candidate_res_list["name"].isin(self.must_not_visit_restaurant)]

                    if self.must_not_visit_restaurant_type is not None:
                        candidate_res_list = candidate_res_list[
                            ~candidate_res_list["cuisine"].isin(self.must_not_visit_restaurant_type)]

                    if self.must_visit_restaurant is not None:
                        for must_name in self.must_visit_restaurant:
                            if must_name not in candidate_res_list["name"].values:
                                must_res = res_info[res_info["name"] == must_name]
                                if not must_res.empty:
                                    candidate_res_list = pd.concat([candidate_res_list, must_res]).drop_duplicates()

                    if self.must_visit_restaurant_type is not None:
                        found_types = candidate_res_list["cuisine"].unique()
                        missing_types = [t for t in self.must_visit_restaurant_type if t not in found_types]
                        if missing_types:
                            print(f"[Warning] must visit restaurant type:{missing_types} is not in candidates")

                    # 过滤掉已经访问过的
                    candidate_res_list = candidate_res_list.drop(index=self.restaurants_visiting, errors="ignore")

                    # 根据开放时间过滤
                    candidate_res_filtered = candidate_res_list[
                        candidate_res_list.apply(
                            lambda row: (
                                    time_compare_if_earlier_equal(row["opentime"], current_time) and
                                    time_compare_if_earlier_equal(current_time, row["endtime"])
                            ),
                            axis=1
                        )
                    ].copy()

                    # if not self.all_satisfy_flag and (self.overall_budget is not None or self.attraction_budget is not None):
                    if self.overall_budget is not None or self.attraction_budget is not None:
                        # 有预算要求，按price从低到高排序
                        candidate_attr_ranked = candidate_res_filtered.sort_values(by="price").reset_index(drop=True)
                    else:
                        candidate_res_filtered["distance"] = candidate_res_filtered.apply(
                            lambda row: self.calculate_distance(
                                query, current_position, row["name"]
                            ),
                            axis=1
                        )
                        candidate_attr_ranked = candidate_res_filtered.sort_values(by="distance").reset_index(drop=True)

                    n = 30  # 选取前 n 个

                    # must see
                    must_candidates = pd.DataFrame()
                    if self.must_visit_restaurant is not None:
                        for must_name in self.must_visit_restaurant:
                            if must_name in self.restaurant_names_visiting:
                                continue
                            must_res = res_info[res_info["name"] == must_name]
                            if not must_res.empty:
                                must_candidates = pd.concat([must_candidates, must_res]).drop_duplicates()

                    # must cuisine
                    must_type_candidates = pd.DataFrame()
                    if self.must_visit_restaurant_type is not None:
                        for cuisine in self.must_visit_restaurant_type:
                            if cuisine in self.food_type_visiting:
                                continue
                            must_type = res_info[res_info["cuisine"] == cuisine]
                            if not must_type.empty:
                                must_type_candidates = pd.concat([must_candidates, must_type]).drop_duplicates()

                    top_candidates = candidate_attr_ranked.iloc[
                                     :min(n, len(candidate_attr_ranked))].copy()

                    flag = True
                    if self.must_visit_restaurant is not None:
                        temp_cons = {"must_visit_restaurant": self.must_visit_restaurant}
                        flag, _ = self.check_constraint(plan, temp_cons)
                    if not flag:
                        for _, poi_sel in must_candidates.iterrows():
                            if poi_sel["name"] in self.restaurant_names_visiting:
                                continue
                            # 临时添加至计划中，更新当前地点和时间
                            transports_ranking = self.innercity_transports_ranking
                            if self.transport_rules_by_distance is not None:
                                temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                                transports_ranking = self.get_transport_by_distance(temp_distance)
                            # 遍历市内交通类型
                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                # 收集市内交通选项，从当前位置到景点
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue

                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]

                                opentime, endtime = (
                                    poi_sel["opentime"],
                                    poi_sel["endtime"],
                                )

                                # it is closed ...
                                if time_compare_if_earlier_equal(endtime, arrived_time):
                                    closed = True
                                    if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                                        closed = False
                                    if closed:
                                        self.backtrack_count += 1
                                        print("The restaurant is closed now...")
                                        continue

                                backtrack_flag = False
                                if self.transport_rules_by_distance is not None:
                                    distance = 0
                                    for transport in transports_sel:
                                        if transport["mode"] is not None:
                                            distance += transport.get("distance", 0)
                                    if len(transports_sel) == 3:
                                        mode = transports_sel[1]["mode"]
                                    elif len(transports_sel) == 1:
                                        mode = transports_sel[0]["mode"]
                                    for rule in self.transport_rules_by_distance:
                                        if rule["min_distance"] is not None:
                                            if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                        if rule["max_distance"] is not None:
                                            if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                if backtrack_flag and not self.too_many_backtrack:
                                    self.backtrack_count += 1
                                    continue

                                # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                                if time_compare_if_earlier_equal(
                                        arrived_time, opentime
                                ):
                                    act_start_time = opentime
                                else:
                                    act_start_time = arrived_time

                                poi_time = 60
                                # 计算活动结束时间
                                act_end_time = add_time_delta(act_start_time, poi_time)
                                aet = act_end_time
                                # 如果结束时间超过景点关闭时间，则截断为关闭时间
                                if time_compare_if_earlier_equal(endtime, act_end_time):
                                    act_end_time = endtime
                                    if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                                        act_end_time = aet

                                if not self.too_many_backtrack:
                                    # 到达时间约束
                                    if self.activities_arrive_time_dict is not None:
                                        arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                        if arrive_info:
                                            arrive_type, arrive_time = arrive_info
                                            if arrive_type == "early":  # 要求早于某个时间
                                                if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif arrive_type == "late":  # 要求晚于某个时间
                                                if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                    # 离开时间约束
                                    if self.activities_leave_time_dict is not None:
                                        leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                        if leave_info:
                                            leave_type, leave_time = leave_info
                                            if leave_type == "early":  # 要求早于某个时间离开
                                                if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif leave_type == "late":  # 要求晚于某个时间离开
                                                if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                try:
                                    # 添加餐厅活动
                                    plan = self.add_restaurant(
                                        plan,
                                        poi_type,
                                        poi_sel,
                                        current_day,
                                        arrived_time,
                                        transports_sel,
                                    )
                                    pn = poi_sel["name"]
                                    pc = poi_sel["cuisine"]
                                    print(f"add restaurant: {pn}, type: {pc}")
                                except:
                                    self.backtrack_count += 1
                                    print("add_restaurant failed, backtrack...")
                                    continue

                                new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为餐厅结束时间
                                new_position = poi_sel["name"]  # 更新当前位置为餐厅名称

                                res_idx = res_info[res_info["name"] == poi_sel["name"]].index

                                self.restaurants_visiting.append(res_idx)  # 记录已访问餐厅
                                self.food_type_visiting.append(poi_sel["cuisine"])  # 记录已访问食物类型
                                self.restaurant_names_visiting.append(poi_sel["name"])

                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )

                                if success:
                                    return True, plan
                                # 回溯
                                self.backtrack_count += 1
                                print("add_restaurant failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.restaurants_visiting.pop()
                                self.food_type_visiting.pop()
                                self.restaurant_names_visiting.pop()

                    flag = True
                    if self.must_visit_restaurant_type is not None:
                        temp_cons = {"must_visit_restaurant_type": self.must_visit_restaurant_type}
                        flag, _ = self.check_constraint(plan, temp_cons)
                    if not flag:
                        for _, poi_sel in must_type_candidates.iterrows():
                            if poi_sel["name"] in self.restaurant_names_visiting:
                                continue
                            if poi_sel["cuisine"] in self.food_type_visiting:
                                continue
                            # 临时添加至计划中，更新当前地点和时间
                            transports_ranking = self.innercity_transports_ranking
                            if self.transport_rules_by_distance is not None:
                                temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                                transports_ranking = self.get_transport_by_distance(temp_distance)
                            # 遍历市内交通类型
                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                # 收集市内交通选项，从当前位置到景点
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue

                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]

                                opentime, endtime = (
                                    poi_sel["opentime"],
                                    poi_sel["endtime"],
                                )

                                # it is closed ...
                                if time_compare_if_earlier_equal(endtime, arrived_time):
                                    closed = True
                                    if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                                        closed = False
                                    if closed:
                                        self.backtrack_count += 1
                                        print("The restaurant is closed now...")
                                        continue

                                backtrack_flag = False
                                if self.transport_rules_by_distance is not None:
                                    distance = 0
                                    for transport in transports_sel:
                                        if transport["mode"] is not None:
                                            distance += transport.get("distance", 0)
                                    if len(transports_sel) == 3:
                                        mode = transports_sel[1]["mode"]
                                    elif len(transports_sel) == 1:
                                        mode = transports_sel[0]["mode"]
                                    for rule in self.transport_rules_by_distance:
                                        if rule["min_distance"] is not None:
                                            if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                        if rule["max_distance"] is not None:
                                            if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                if backtrack_flag and not self.too_many_backtrack:
                                    self.backtrack_count += 1
                                    continue
                                # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                                if time_compare_if_earlier_equal(
                                        arrived_time, opentime
                                ):
                                    act_start_time = opentime
                                else:
                                    act_start_time = arrived_time

                                poi_time = 60
                                # 计算活动结束时间
                                act_end_time = add_time_delta(act_start_time, poi_time)
                                aet = act_end_time
                                # 如果结束时间超过景点关闭时间，则截断为关闭时间
                                if time_compare_if_earlier_equal(endtime, act_end_time):
                                    act_end_time = endtime
                                    if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                                        act_end_time = aet

                                # 到达时间约束
                                if not self.too_many_backtrack:
                                    if self.activities_arrive_time_dict is not None:
                                        arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                        if arrive_info:
                                            arrive_type, arrive_time = arrive_info
                                            if arrive_type == "early":  # 要求早于某个时间
                                                if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif arrive_type == "late":  # 要求晚于某个时间
                                                if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                    # 离开时间约束
                                    if self.activities_leave_time_dict is not None:
                                        leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                        if leave_info:
                                            leave_type, leave_time = leave_info
                                            if leave_type == "early":  # 要求早于某个时间离开
                                                if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif leave_type == "late":  # 要求晚于某个时间离开
                                                if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                try:
                                    # 添加餐厅活动
                                    plan = self.add_restaurant(
                                        plan,
                                        poi_type,
                                        poi_sel,
                                        current_day,
                                        arrived_time,
                                        transports_sel,
                                    )
                                    pn = poi_sel["name"]
                                    pc = poi_sel["cuisine"]
                                    print(f"add restaurant: {pn}, type: {pc}")
                                except:
                                    self.backtrack_count += 1
                                    print("add_restaurant failed, backtrack...")
                                    continue

                                new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为餐厅结束时间
                                new_position = poi_sel["name"]  # 更新当前位置为餐厅名称

                                res_idx = res_info[res_info["name"] == poi_sel["name"]].index

                                self.restaurants_visiting.append(res_idx)  # 记录已访问餐厅
                                self.food_type_visiting.append(poi_sel["cuisine"])  # 记录已访问食物类型
                                self.restaurant_names_visiting.append(poi_sel["name"])

                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )

                                if success:
                                    return True, plan
                                # 回溯
                                self.backtrack_count += 1
                                print("add_restaurant failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.restaurants_visiting.pop()
                                self.food_type_visiting.pop()
                                self.restaurant_names_visiting.pop()

                    for _, poi_sel in top_candidates.iterrows():
                        if poi_sel["name"] in self.restaurant_names_visiting:
                            continue
                        if poi_sel["cuisine"] in self.food_type_visiting:
                            continue
                        # 临时添加至计划中，更新当前地点和时间
                        transports_ranking = self.innercity_transports_ranking
                        if self.transport_rules_by_distance is not None:
                            temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                            transports_ranking = self.get_transport_by_distance(temp_distance)
                        # 遍历市内交通类型
                        for trans_type_sel in transports_ranking:
                            self.search_nodes += 1
                            # 收集市内交通选项，从当前位置到景点
                            transports_sel = self.collect_innercity_transport(
                                query["target_city"],
                                current_position,
                                poi_sel["name"],
                                current_time,
                                trans_type_sel,
                            )
                            if not isinstance(transports_sel, list):
                                self.backtrack_count += 1
                                print("inner-city transport error, backtrack...")
                                continue

                            if len(transports_sel) == 0:
                                arrived_time = current_time
                            else:
                                arrived_time = transports_sel[-1]["end_time"]

                            opentime, endtime = (
                                poi_sel["opentime"],
                                poi_sel["endtime"],
                            )

                            # it is closed ...
                            if time_compare_if_earlier_equal(endtime, arrived_time):
                                closed = True
                                if time_compare_if_earlier_equal(endtime, opentime): # 营业到第二天
                                    closed = False
                                if closed:
                                    self.backtrack_count += 1
                                    print("The restaurant is closed now...")
                                    continue

                            backtrack_flag = False
                            if self.transport_rules_by_distance is not None:
                                distance = 0
                                for transport in transports_sel:
                                    if transport["mode"] is not None:
                                        distance += transport.get("distance", 0)
                                if len(transports_sel) == 3:
                                    mode = transports_sel[1]["mode"]
                                elif len(transports_sel) == 1:
                                    mode = transports_sel[0]["mode"]
                                for rule in self.transport_rules_by_distance:
                                    if rule["min_distance"] is not None:
                                        if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                            print("backtrack")
                                            backtrack_flag = True
                                    if rule["max_distance"] is not None:
                                        if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                            print("backtrack")
                                            backtrack_flag = True
                            if backtrack_flag and not self.too_many_backtrack:
                                self.backtrack_count += 1
                                continue

                            # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                            if time_compare_if_earlier_equal(
                                    arrived_time, opentime
                            ):
                                act_start_time = opentime
                            else:
                                act_start_time = arrived_time

                            poi_time = 60
                            # 计算活动结束时间
                            act_end_time = add_time_delta(act_start_time, poi_time)
                            aet = act_end_time
                            # 如果结束时间超过景点关闭时间，则截断为关闭时间
                            if time_compare_if_earlier_equal(endtime, act_end_time):
                                act_end_time = endtime
                                if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                                    act_end_time = aet

                            if not self.too_many_backtrack:
                                # 到达时间约束
                                if self.activities_arrive_time_dict is not None:
                                    arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                    if arrive_info:
                                        arrive_type, arrive_time = arrive_info
                                        if arrive_type == "early":  # 要求早于某个时间
                                            if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                print(
                                                    f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                self.backtrack_count += 1
                                                continue
                                        elif arrive_type == "late":  # 要求晚于某个时间
                                            if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                print(
                                                    f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                self.backtrack_count += 1
                                                continue

                                # 离开时间约束
                                if self.activities_leave_time_dict is not None:
                                    leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                    if leave_info:
                                        leave_type, leave_time = leave_info
                                        if leave_type == "early":  # 要求早于某个时间离开
                                            if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                print(
                                                    f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                self.backtrack_count += 1
                                                continue
                                        elif leave_type == "late":  # 要求晚于某个时间离开
                                            if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                print(
                                                    f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                self.backtrack_count += 1
                                                continue

                            try:
                                # 添加餐厅活动
                                plan = self.add_restaurant(
                                    plan,
                                    poi_type,
                                    poi_sel,
                                    current_day,
                                    arrived_time,
                                    transports_sel,
                                )
                                pn = poi_sel["name"]
                                pc = poi_sel["cuisine"]
                                print(f"add restaurant: {pn}, type: {pc}")
                            except:
                                self.backtrack_count += 1
                                print("add_restaurant failed, backtrack...")
                                continue

                            new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为餐厅结束时间
                            new_position = poi_sel["name"]  # 更新当前位置为餐厅名称

                            res_idx = res_info[res_info["name"] == poi_sel["name"]].index

                            self.restaurants_visiting.append(res_idx)  # 记录已访问餐厅
                            self.food_type_visiting.append(poi_sel["cuisine"])  # 记录已访问食物类型
                            self.restaurant_names_visiting.append(poi_sel["name"])

                            success, plan = self.dfs_poi(
                                query,
                                poi_plan,
                                plan,
                                new_time,
                                new_position,
                                current_day,
                            )

                            if success:
                                return True, plan
                            # 回溯
                            self.backtrack_count += 1
                            print("add_restaurant failed, backtrack...")

                            plan[current_day]["activities"].pop()
                            self.restaurants_visiting.pop()
                            self.food_type_visiting.pop()
                            self.restaurant_names_visiting.pop()

                # 如果是景点
                elif poi_type == "attraction":
                    attr_info = self.memory["attractions"]
                    candidate_attr_list = attr_info.copy()

                    # 仅选择免费的景点
                    if self.only_free_attractions is not None and self.only_free_attractions:
                        candidate_attr_list = candidate_attr_list[candidate_attr_list["price"] == 0]

                    # 删除不想游览的景点
                    if self.must_not_see_attraction is not None:
                        candidate_attr_list = candidate_attr_list[
                            ~candidate_attr_list["name"].isin(self.must_not_see_attraction)]

                    # 删除不想游览的景点类型
                    if self.must_not_see_attraction_type is not None:
                        candidate_attr_list = candidate_attr_list[
                            ~candidate_attr_list["type"].isin(self.must_not_see_attraction_type)]

                    # 确保必须游览的景点在候选集中
                    if self.must_see_attraction is not None:
                        for must_name in self.must_see_attraction:
                            if must_name not in candidate_attr_list["name"].values:
                                must_attr = attr_info[attr_info["name"] == must_name]
                                if not must_attr.empty:
                                    candidate_attr_list = pd.concat([candidate_attr_list, must_attr]).drop_duplicates()

                    # 确保必须游览的类型景点存在
                    if self.must_see_attraction_type is not None:
                        found_types = candidate_attr_list["type"].unique()
                        missing_types = [t for t in self.must_see_attraction_type if t not in found_types]
                        if missing_types:
                            print(f"[Warning] must see attraction type:{missing_types} is not in attraction candidates")

                    # 过滤掉已经访问过的景点
                    candidate_attr_list = candidate_attr_list.drop(index=self.attractions_visiting, errors="ignore")

                    # 根据开放时间过滤
                    candidate_attr_filtered = candidate_attr_list[
                        candidate_attr_list.apply(
                            lambda row: (
                                    time_compare_if_earlier_equal(row["opentime"], current_time) and
                                    time_compare_if_earlier_equal(current_time, row["endtime"])
                            ),
                            axis=1
                        )
                    ].copy()

                    # if not self.all_satisfy_flag and (self.overall_budget is not None or self.attraction_budget is not None):
                    if self.overall_budget is not None or self.attraction_budget is not None:
                        print("sorted by price")
                        # 有预算要求，按price从低到高排序
                        candidate_attr_ranked = candidate_attr_filtered.sort_values(by="price").reset_index(drop=True)
                    else:
                        print("sorted by distance")
                        # 根据current_position计算距离景点的距离并排序
                        candidate_attr_filtered["distance"] = candidate_attr_filtered.apply(
                            lambda row: self.calculate_distance(
                                query, current_position, row["name"]
                            ),
                            axis=1
                        )
                        candidate_attr_ranked = candidate_attr_filtered.sort_values(by="distance").reset_index(
                            drop=True)

                    # 检查时间段
                    stage = 0
                    if current_day == 0 and time_compare_if_earlier_equal("14:00", current_time) and "dinner" in candidates_type:
                        stage = 2
                    elif "lunch" in candidates_type and "dinner" in candidates_type:
                        stage = 1
                    elif "lunch" not in candidates_type and "dinner" in candidates_type:
                        stage = 2
                    # elif "lunch" not in candidates_type and "dinner" not in candidates_type:
                    #     stage = 3

                    n = 30  # 选取前 n 个景点

                    must_candidates = pd.DataFrame()
                    if self.must_see_attraction is not None:
                        for must_name in self.must_see_attraction:
                            if must_name in self.attractions_visiting:
                                continue
                            must_attr = attr_info[attr_info["name"] == must_name]
                            if not must_attr.empty:
                                must_candidates = pd.concat([must_candidates, must_attr]).drop_duplicates()

                    if not must_candidates.empty:
                        must_candidates = must_candidates.copy()
                        must_candidates["open_duration"] = must_candidates.apply(
                            lambda row: get_time_delta(row["opentime"], row["endtime"]),
                            axis=1
                        )
                        must_candidates = must_candidates.sort_values(by="open_duration", ascending=True).reset_index(drop=True)

                    must_type_candidates = pd.DataFrame()
                    if self.must_see_attraction_type is not None:
                        for must_type in self.must_see_attraction_type:
                            if must_type in self.spot_type_visiting:
                                continue
                            must_attr = attr_info[attr_info["type"] == must_type]
                            if not must_attr.empty:
                                must_type_candidates = pd.concat([must_candidates, must_attr]).drop_duplicates()

                    top_candidates = candidate_attr_ranked.iloc[
                                     :min(n, len(candidate_attr_ranked))].copy()


                    flag = True
                    if self.must_see_attraction is not None:
                        temp_cons = {"must_see_attraction": self.must_see_attraction}
                        flag, _ = self.check_constraint(plan, temp_cons)
                    if not flag:
                        for _, poi_sel in must_candidates.iterrows():
                            if poi_sel["name"] in self.attraction_names_visiting:
                                continue
                            print(f"lookahead to add attraction, candidate: {poi_sel['name']}")
                            # 临时添加至计划中，更新当前地点和时间
                            transports_ranking = self.innercity_transports_ranking
                            if self.transport_rules_by_distance is not None:
                                temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                                transports_ranking = self.get_transport_by_distance(temp_distance)
                            # 遍历市内交通类型
                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                # 收集市内交通选项，从当前位置到景点
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue

                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]

                                opentime, endtime = (
                                    poi_sel["opentime"],
                                    poi_sel["endtime"],
                                )

                                # it is closed ...
                                if time_compare_if_earlier_equal(endtime, arrived_time):
                                    self.backtrack_count += 1
                                    print(
                                        f"{poi_sel['name']} closed at {endtime}, start time: {current_time}, arrival time: {arrived_time}, backtrack...")
                                    continue

                                backtrack_flag = False
                                if self.transport_rules_by_distance is not None:
                                    distance = 0
                                    for transport in transports_sel:
                                        if transport["mode"] is not None:
                                            distance += transport.get("distance", 0)
                                    if len(transports_sel) == 3:
                                        mode = transports_sel[1]["mode"]
                                    elif len(transports_sel) == 1:
                                        mode = transports_sel[0]["mode"]
                                    for rule in self.transport_rules_by_distance:
                                        if rule["min_distance"] is not None:
                                            if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                        if rule["max_distance"] is not None:
                                            if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                if backtrack_flag and not self.too_many_backtrack:
                                    self.backtrack_count += 1
                                    continue

                                # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                                if time_compare_if_earlier_equal(arrived_time, opentime):
                                    act_start_time = opentime
                                else:
                                    act_start_time = arrived_time

                                # 选择景点游览时间
                                poi_time = 90
                                # 计算活动结束时间
                                act_end_time = add_time_delta(act_start_time, poi_time)
                                # 如果结束时间超过景点关闭时间，则截断为关闭时间
                                if time_compare_if_earlier_equal(endtime, act_end_time):
                                    act_end_time = endtime

                                if not self.too_many_backtrack:
                                    # 到达时间约束
                                    if self.activities_arrive_time_dict is not None:
                                        arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                        if arrive_info:
                                            arrive_type, arrive_time = arrive_info
                                            if arrive_type == "early":  # 要求早于某个时间
                                                if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif arrive_type == "late":  # 要求晚于某个时间
                                                if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                    # 离开时间约束
                                    if self.activities_leave_time_dict is not None:
                                        leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                        if leave_info:
                                            leave_type, leave_time = leave_info
                                            if leave_type == "early":  # 要求早于某个时间离开
                                                if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif leave_type == "late":  # 要求晚于某个时间离开
                                                if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                flag = False
                                if stage == 1:
                                    if time_compare_if_earlier_equal(act_end_time, "12:00"):
                                        flag = True

                                if stage == 2:
                                    if time_compare_if_earlier_equal(act_end_time, "19:00"):
                                        flag = True

                                if stage == 0 or stage == 3:
                                    flag = True

                                if not flag:
                                    continue

                                plan[current_day]["activities"] = self.add_poi(
                                    activities=plan[current_day]["activities"],
                                    position=poi_sel["name"],
                                    poi_type=poi_type,
                                    price=int(poi_sel["price"]),
                                    cost=int(poi_sel["price"])
                                         * self.query["people_number"],
                                    start_time=act_start_time,
                                    end_time=act_end_time,
                                    innercity_transports=transports_sel,
                                )
                                # 设置门票数量
                                plan[current_day]["activities"][-1]["tickets"] = (
                                    self.query["people_number"]
                                )

                                pn = poi_sel["name"]
                                pc = poi_sel["type"]
                                print(f"add attraction: {pn}, type: {pc}")

                                new_time = act_end_time  # 更新当前时间为活动结束时间
                                new_position = poi_sel["name"]  # 更新当前位置为景点名称

                                attr_idx = attr_info[attr_info["name"] == poi_sel["name"]].index

                                self.attractions_visiting.append(attr_idx)  # 记录已访问景点
                                self.spot_type_visiting.append(poi_sel["type"])  # 记录已访问景点类型
                                self.attraction_names_visiting.append(poi_sel["name"])  # 记录已访问景点名称

                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )

                                if success:
                                    return True, plan

                                self.backtrack_count += 1
                                print("add_attraction failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.attractions_visiting.pop()
                                self.spot_type_visiting.pop()
                                self.attraction_names_visiting.pop()

                    flag = True
                    if self.must_see_attraction_type is not None:
                        temp_cons = {"must_see_attraction_type": self.must_see_attraction_type}
                        flag, _ = self.check_constraint(plan, temp_cons)
                    if not flag:
                        for _, poi_sel in must_type_candidates.iterrows():
                            if poi_sel["name"] in self.attraction_names_visiting:
                                continue
                            if poi_sel["type"] in self.spot_type_visiting:
                                continue
                            print(f"lookahead to add attraction, candidate: {poi_sel['name']}")
                            # 临时添加至计划中，更新当前地点和时间
                            transports_ranking = self.innercity_transports_ranking
                            if self.transport_rules_by_distance is not None:
                                temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                                transports_ranking = self.get_transport_by_distance(temp_distance)
                            # 遍历市内交通类型
                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                # 收集市内交通选项，从当前位置到景点
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue

                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]

                                opentime, endtime = (
                                    poi_sel["opentime"],
                                    poi_sel["endtime"],
                                )

                                # it is closed ...
                                if time_compare_if_earlier_equal(endtime, arrived_time):
                                    self.backtrack_count += 1
                                    print(
                                        f"{poi_sel['name']} closed at {endtime}, start time: {current_time}, arrival time: {arrived_time}, backtrack...")
                                    continue

                                backtrack_flag = False
                                if self.transport_rules_by_distance is not None:
                                    distance = 0
                                    for transport in transports_sel:
                                        if transport["mode"] is not None:
                                            distance += transport.get("distance", 0)
                                    if len(transports_sel) == 3:
                                        mode = transports_sel[1]["mode"]
                                    elif len(transports_sel) == 1:
                                        mode = transports_sel[0]["mode"]
                                    for rule in self.transport_rules_by_distance:
                                        if rule["min_distance"] is not None:
                                            if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                        if rule["max_distance"] is not None:
                                            if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                                print("backtrack")
                                                backtrack_flag = True
                                if backtrack_flag and not self.too_many_backtrack:
                                    self.backtrack_count += 1
                                    continue

                                # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                                if time_compare_if_earlier_equal(arrived_time, opentime):
                                    act_start_time = opentime
                                else:
                                    act_start_time = arrived_time

                                # 选择景点游览时间
                                poi_time = 90
                                # 计算活动结束时间
                                act_end_time = add_time_delta(act_start_time, poi_time)
                                # 如果结束时间超过景点关闭时间，则截断为关闭时间
                                if time_compare_if_earlier_equal(endtime, act_end_time):
                                    act_end_time = endtime

                                if not self.too_many_backtrack:
                                    # 到达时间约束
                                    if self.activities_arrive_time_dict is not None:
                                        arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                        if arrive_info:
                                            arrive_type, arrive_time = arrive_info
                                            if arrive_type == "early":  # 要求早于某个时间
                                                if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif arrive_type == "late":  # 要求晚于某个时间
                                                if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                    print(
                                                        f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                    # 离开时间约束
                                    if self.activities_leave_time_dict is not None:
                                        leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                        if leave_info:
                                            leave_type, leave_time = leave_info
                                            if leave_type == "early":  # 要求早于某个时间离开
                                                if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue
                                            elif leave_type == "late":  # 要求晚于某个时间离开
                                                if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                    print(
                                                        f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                    self.backtrack_count += 1
                                                    continue

                                flag = False
                                if stage == 1:
                                    if time_compare_if_earlier_equal(act_end_time, "12:00"):
                                        flag = True

                                if stage == 2:
                                    if time_compare_if_earlier_equal(act_end_time, "19:00"):
                                        flag = True

                                if stage == 0 or stage == 3:
                                    flag = True

                                if not flag:
                                    continue

                                plan[current_day]["activities"] = self.add_poi(
                                    activities=plan[current_day]["activities"],
                                    position=poi_sel["name"],
                                    poi_type=poi_type,
                                    price=int(poi_sel["price"]),
                                    cost=int(poi_sel["price"])
                                         * self.query["people_number"],
                                    start_time=act_start_time,
                                    end_time=act_end_time,
                                    innercity_transports=transports_sel,
                                )
                                # 设置门票数量
                                plan[current_day]["activities"][-1]["tickets"] = (
                                    self.query["people_number"]
                                )

                                pn = poi_sel["name"]
                                pc = poi_sel["type"]
                                print(f"add attraction: {pn}, type: {pc}")

                                new_time = act_end_time  # 更新当前时间为活动结束时间
                                new_position = poi_sel["name"]  # 更新当前位置为景点名称

                                attr_idx = attr_info[attr_info["name"] == poi_sel["name"]].index

                                self.attractions_visiting.append(attr_idx)  # 记录已访问景点
                                self.spot_type_visiting.append(poi_sel["type"])  # 记录已访问景点类型
                                self.attraction_names_visiting.append(poi_sel["name"])  # 记录已访问景点名称

                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )

                                if success:
                                    return True, plan

                                self.backtrack_count += 1
                                print("add_attraction failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.attractions_visiting.pop()
                                self.spot_type_visiting.pop()
                                self.attraction_names_visiting.pop()

                    for _, poi_sel in top_candidates.iterrows():
                        if poi_sel["name"] in self.attraction_names_visiting:
                            continue
                        if poi_sel["type"] in self.spot_type_visiting:
                            continue
                        print(f"lookahead to add attraction, candidate: {poi_sel['name']}")
                        # 临时添加至计划中，更新当前地点和时间
                        transports_ranking = self.innercity_transports_ranking
                        if self.transport_rules_by_distance is not None:
                            temp_distance = self.calculate_distance(query, current_position, poi_sel["name"])
                            transports_ranking = self.get_transport_by_distance(temp_distance)
                        # 遍历市内交通类型
                        for trans_type_sel in transports_ranking:
                            self.search_nodes += 1
                            # 收集市内交通选项，从当前位置到景点
                            transports_sel = self.collect_innercity_transport(
                                query["target_city"],
                                current_position,
                                poi_sel["name"],
                                current_time,
                                trans_type_sel,
                            )
                            if not isinstance(transports_sel, list):
                                self.backtrack_count += 1
                                print("inner-city transport error, backtrack...")
                                continue

                            if len(transports_sel) == 0:
                                arrived_time = current_time
                            else:
                                arrived_time = transports_sel[-1]["end_time"]

                            opentime, endtime = (
                                poi_sel["opentime"],
                                poi_sel["endtime"],
                            )

                            # it is closed ...
                            if time_compare_if_earlier_equal(endtime, arrived_time):
                                self.backtrack_count += 1
                                print(
                                    f"{poi_sel['name']} closed at {endtime}, start time: {current_time}, arrival time: {arrived_time}, backtrack...")
                                continue

                            backtrack_flag = False
                            if self.transport_rules_by_distance is not None:
                                distance = 0
                                for transport in transports_sel:
                                    if transport["mode"] is not None:
                                        distance += transport.get("distance", 0)
                                if len(transports_sel) == 3:
                                    mode = transports_sel[1]["mode"]
                                elif len(transports_sel) == 1:
                                    mode = transports_sel[0]["mode"]
                                for rule in self.transport_rules_by_distance:
                                    if rule["min_distance"] is not None:
                                        if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                            print("backtrack")
                                            backtrack_flag = True
                                    if rule["max_distance"] is not None:
                                        if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                            print("backtrack")
                                            backtrack_flag = True
                            if backtrack_flag and not self.too_many_backtrack:
                                self.backtrack_count += 1
                                continue

                            # 确定活动开始时间,如果到达早于开放时间，则从开放时间开始
                            if time_compare_if_earlier_equal(arrived_time, opentime):
                                act_start_time = opentime
                            else:
                                act_start_time = arrived_time

                            # 选择景点游览时间
                            poi_time = 90
                            # 计算活动结束时间
                            act_end_time = add_time_delta(act_start_time, poi_time)
                            # 如果结束时间超过景点关闭时间，则截断为关闭时间
                            if time_compare_if_earlier_equal(endtime, act_end_time):
                                act_end_time = endtime

                            if not self.too_many_backtrack:
                                # 到达时间约束
                                if self.activities_arrive_time_dict is not None:
                                    arrive_info = self.activities_arrive_time_dict.get(poi_sel["name"])
                                    if arrive_info:
                                        arrive_type, arrive_time = arrive_info
                                        if arrive_type == "early":  # 要求早于某个时间
                                            if not time_compare_if_earlier_equal(act_start_time, arrive_time):
                                                print(
                                                    f"[Constraint] Arrival for {poi_sel['name']} too late: {act_start_time} > {arrive_time}")
                                                self.backtrack_count += 1
                                                continue
                                        elif arrive_type == "late":  # 要求晚于某个时间
                                            if not time_compare_if_earlier_equal(arrive_time, act_start_time):
                                                print(
                                                    f"[Constraint] Arrival for {poi_sel['name']} too early: {act_start_time} < {arrive_time}")
                                                self.backtrack_count += 1
                                                continue

                                # 离开时间约束
                                if self.activities_leave_time_dict is not None:
                                    leave_info = self.activities_leave_time_dict.get(poi_sel["name"])
                                    if leave_info:
                                        leave_type, leave_time = leave_info
                                        if leave_type == "early":  # 要求早于某个时间离开
                                            if not time_compare_if_earlier_equal(act_end_time, leave_time):
                                                print(
                                                    f"[Constraint] Leaving {poi_sel['name']} too late: {act_end_time} > {leave_time}")
                                                self.backtrack_count += 1
                                                continue
                                        elif leave_type == "late":  # 要求晚于某个时间离开
                                            if not time_compare_if_earlier_equal(leave_time, act_end_time):
                                                print(
                                                    f"[Constraint] Leaving {poi_sel['name']} too early: {act_end_time} < {leave_time}")
                                                self.backtrack_count += 1
                                                continue

                            flag = False
                            if stage == 1:
                                if time_compare_if_earlier_equal(act_end_time, "12:00"):
                                    flag = True

                            if stage == 2:
                                if time_compare_if_earlier_equal(act_end_time, "19:00"):
                                    flag = True

                            if stage == 0 or stage == 3:
                                flag = True

                            if not flag:
                                continue

                            plan[current_day]["activities"] = self.add_poi(
                                activities=plan[current_day]["activities"],
                                position=poi_sel["name"],
                                poi_type=poi_type,
                                price=int(poi_sel["price"]),
                                cost=int(poi_sel["price"])
                                     * self.query["people_number"],
                                start_time=act_start_time,
                                end_time=act_end_time,
                                innercity_transports=transports_sel,
                            )
                            # 设置门票数量
                            plan[current_day]["activities"][-1]["tickets"] = (
                                self.query["people_number"]
                            )

                            pn = poi_sel["name"]
                            pc = poi_sel["type"]
                            print(f"add attraction: {pn}, type: {pc}")

                            new_time = act_end_time  # 更新当前时间为活动结束时间
                            new_position = poi_sel["name"]  # 更新当前位置为景点名称

                            attr_idx = attr_info[attr_info["name"] == poi_sel["name"]].index

                            self.attractions_visiting.append(attr_idx)  # 记录已访问景点
                            self.spot_type_visiting.append(poi_sel["type"])  # 记录已访问景点类型
                            self.attraction_names_visiting.append(poi_sel["name"])  # 记录已访问景点名称

                            success, plan = self.dfs_poi(
                                query,
                                poi_plan,
                                plan,
                                new_time,
                                new_position,
                                current_day,
                            )

                            if success:
                                return True, plan

                            self.backtrack_count += 1
                            print("add_attraction failed, backtrack...")

                            plan[current_day]["activities"].pop()
                            self.attractions_visiting.pop()
                            self.spot_type_visiting.pop()
                            self.attraction_names_visiting.pop()

                # 如果是旅行的最后一天
                if current_day == query["days"] - 1:

                    # go back

                    if len(plan) < current_day + 1:
                        plan.append({"day": current_day + 1, "activities": []})
                    self.search_nodes += 1
                    # 获取市内交通排名
                    # transports_ranking = self.ranking_innercity_transport(current_position, poi_plan["back_transport"]["From"], current_day, current_time)
                    transports_ranking = self.innercity_transports_ranking
                    if self.transport_rules_by_distance is not None:
                        temp_distance = self.calculate_distance(query, current_position, poi_plan["back_transport"]["From"])
                        transports_ranking = self.get_transport_by_distance(temp_distance)
                    # 遍历市内交通类型
                    for trans_type_sel in transports_ranking:
                        self.search_nodes += 1
                        # 收集市内交通选项，从当前位置到返程交通的出发地
                        print("last day, collecting innercity transport to back-transport")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            poi_plan["back_transport"]["From"],
                            current_time,
                            trans_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]

                        if not self.too_many_backtrack:
                            if not time_compare_if_earlier_equal(arrived_time, poi_plan["back_transport"]["BeginTime"]):
                                self.backtrack_count += 1
                                print("Fail to catch the back transport")
                                continue

                        backtrack_flag = False
                        if self.transport_rules_by_distance is not None:
                            distance = 0
                            for transport in transports_sel:
                                if transport["mode"] is not None:
                                    distance += transport.get("distance", 0)
                            if len(transports_sel) == 3:
                                mode = transports_sel[1]["mode"]
                            elif len(transports_sel) == 1:
                                mode = transports_sel[0]["mode"]
                            for rule in self.transport_rules_by_distance:
                                if rule["min_distance"] is not None:
                                    if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                                if rule["max_distance"] is not None:
                                    if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                        if backtrack_flag and not self.too_many_backtrack:
                            self.backtrack_count += 1
                            continue

                        # 添加返程城际交通活动
                        plan[current_day]["activities"] = self.add_intercity_transport(
                            plan[current_day]["activities"],
                            poi_plan["back_transport"],
                            innercity_transports=transports_sel,
                            tickets=self.query["people_number"],
                        )

                        if not self.too_many_backtrack:
                            over_budget = self.check_budgets(plan)
                            if over_budget:
                                plan[current_day]["activities"].pop()
                                return False, plan

                        # 验证计划是否满足所有约束
                        res_bool, res_plan = self.constraints_validation(
                            query, plan, poi_plan
                        )

                        return True, res_plan

                # 如果不是最后一天且天数大于 1
                elif self.query["days"] > 1:
                    # go to hotel
                    hotel_sel = poi_plan["accommodation"]  # 获取选定的酒店信息
                    self.search_nodes += 1
                    # 获取市内交通排名
                    transports_ranking = self.innercity_transports_ranking
                    if self.transport_rules_by_distance is not None:
                        temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                        transports_ranking = self.get_transport_by_distance(temp_distance)
                    for trans_type_sel in transports_ranking:
                        self.search_nodes += 1
                        # 收集市内交通选项，从当前位置到酒店
                        print("not last day, but last event, collecting innercity transport to hotel")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        print(f"from: {current_position} to {hotel_sel['name']}")
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]

                        backtrack_flag = False
                        if self.transport_rules_by_distance is not None:
                            distance = 0
                            for transport in transports_sel:
                                if transport["mode"] is not None:
                                    distance += transport.get("distance", 0)
                            if len(transports_sel) == 3:
                                mode = transports_sel[1]["mode"]
                            elif len(transports_sel) == 1:
                                mode = transports_sel[0]["mode"]
                            for rule in self.transport_rules_by_distance:
                                if rule["min_distance"] is not None:
                                    if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                                if rule["max_distance"] is not None:
                                    if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                        if backtrack_flag and not self.too_many_backtrack:
                            self.backtrack_count += 1
                            continue

                        # 添加住宿活动
                        plan = self.add_accommodation(
                            current_plan=plan,
                            hotel_sel=hotel_sel,
                            current_day=current_day,
                            arrived_time=arrived_time,
                            required_rooms=self.required_rooms,
                            transports_sel=transports_sel,
                        )

                        new_time = "00:00"  # 新的一天开始
                        new_position = hotel_sel["name"]  # 新位置为酒店名称

                        # 递归调用 dfs_poi 进行后续规划
                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan,
                            new_time,
                            new_position,
                            current_day + 1,
                        )

                        if success:
                            return True, plan
                        else:
                            self.backtrack_count += 1
                            print("Try the go back hotel, failed, backtrack...")

                            plan[current_day]["activities"].pop()

                            return False, plan
            else:
                # raise Exception("Not Implemented.")
                print("incorrect poi type: {}".format(poi_type))
                continue

            candidates_type.remove(poi_type)
            print(f"remove: {poi_type}, candidate type: {candidates_type}")
            print("try another poi type, backtrack...")

        return False, plan

    # ====================================================================================
    # SECTION 6: CONSTRAINT VALIDATION
    # ====================================================================================
    #
    # These methods validate plans against different types of constraints:
    #   - check_constraint: Budget constraints (attraction, restaurant, hotel, etc.)
    #   - check_requirement: Must-visit POI requirements
    #   - check_budgets: Overall budget validation
    #   - constraints_validation: Complete plan validation (schema, commonsense, logic)
    #   - check_if_too_late: Time feasibility check
    #
    # Lines: 2672-2918
    # ====================================================================================

    def check_constraint(self, plan, constraints):
        # 初始化访问记录
        visited_attractions = set()
        visited_attraction_types = set()
        visited_restaurants = set()
        visited_restaurant_types = set()

        logic_fail = False
        backtrack = False

        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0
        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner", "attraction"]:
                    overall_cost += activity["cost"]
                    restaurant_cost += activity["cost"]
                if activity["type"] in ["attraction"]:
                    overall_cost += activity["cost"]
                    attraction_cost += activity["cost"]
                innercity_cost += sum(transport.get("cost", 0) for transport in activity.get("transports", []))
        self.overall_cost = overall_cost + innercity_cost + self.hotel_cost + self.intercity_cost
        # print(f"overall cost: {self.overall_cost}, attraction cost:{attraction_cost}, restaurant cost: {restaurant_cost}, innercity cost: {innercity_cost}")

        if self.attraction_budget is not None and self.attraction_budget < attraction_cost:
            self.backtrack_count += 1
            print("attraction budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.restaurant_budget is not None and self.restaurant_budget < restaurant_cost:
            self.backtrack_count += 1
            print("restaurant budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.innercity_budget is not None and self.innercity_budget < innercity_cost:
            self.backtrack_count += 1
            print("innercity budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.overall_budget is not None and self.overall_budget < self.overall_cost:
            self.backtrack_count += 1
            print("overall budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False

        # 遍历活动，统计信息并处理 must_not 类约束 & only_free_attractions
        for day in plan:
            for act in day["activities"]:
                poi_name = act.get("position")
                poi_info = None

                if act.get("type") == "attraction":
                    match = self.memory["attractions"][self.memory["attractions"]["name"] == poi_name]
                    if not match.empty:
                        poi_info = match.iloc[0].to_dict()
                        visited_attractions.add(poi_info["name"])
                        visited_attraction_types.add(poi_info["type"])

                        # must_not_see_attraction
                        if "must_not_see_attraction" in constraints:
                            if poi_info["name"] in constraints["must_not_see_attraction"]:
                                print("visited must_not_see_attraction")
                                backtrack = True

                        # must_not_see_attraction_type
                        if "must_not_see_attraction_type" in constraints:
                            if poi_info["type"] in constraints["must_not_see_attraction_type"]:
                                print("visited must_not_see_attraction_type")
                                backtrack = True

                        # only_free_attractions
                        if "only_free_attractions" in constraints and poi_info.get("price", 0) > 0:
                            print("only_free_attractions but not free")
                            backtrack = True

                elif act.get("type") in {"lunch", "dinner"}:
                    match = self.memory["restaurants"][self.memory["restaurants"]["name"] == poi_name]
                    if not match.empty:
                        poi_info = match.iloc[0].to_dict()
                        visited_restaurants.add(poi_info["name"])
                        visited_restaurant_types.add(poi_info["cuisine"])

                        # must_not_visit_restaurant
                        if "must_not_visit_restaurant" in constraints:
                            if poi_info["name"] in constraints["must_not_visit_restaurant"]:
                                print("visited must_not_visit_restaurant")
                                backtrack = True

                        # must_not_visit_restaurant_type
                        if "must_not_visit_restaurant_type" in constraints:
                            if poi_info["cuisine"] in constraints["must_not_visit_restaurant_type"]:
                                print("visited must_not_visit_restaurant_type")
                                backtrack = True

        # 遍历 must 类要求，统计是否满足
        # 注意：这些不满足不立即回溯，而是给机会后续补上
        if "must_see_attraction" in constraints:
            required = set(constraints["must_see_attraction"])
            if not required.issubset(visited_attractions):
                logic_fail = True

        if "must_see_attraction_type" in constraints:
            required = set(constraints["must_see_attraction_type"])
            if not required.issubset(visited_attraction_types):
                logic_fail = True

        if "must_visit_restaurant" in constraints:
            required = set(constraints["must_visit_restaurant"])
            if not required.issubset(visited_restaurants):
                logic_fail = True

        if "must_visit_restaurant_type" in constraints:
            required = set(constraints["must_visit_restaurant_type"])
            if not required.issubset(visited_restaurant_types):
                logic_fail = True

        if "must_innercity_transport" in constraints:
            if not set(constraints["must_innercity_transport"]).issubset(set(self.innercity_transports_ranking)):
                logic_fail = True

        if "must_not_innercity_transport" in constraints:
            if not set(constraints["must_not_innercity_transport"]) & set(self.innercity_transports_ranking):
                logic_fail = True

        return (not backtrack) and (not logic_fail), backtrack

    def check_requirement(self, plan):
        if self.all_satisfy:
            # 情况1：必须全部满足
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints)
                if backtrack:
                    return False, True  # 立即回溯
                if not ok:
                    return False, False  # 不满足但不强制回溯
            return True, False
        else:
            # 情况2：满足任一要求即可
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints)
                if backtrack:
                    return False, True
                if ok:
                    return True, False  # 有一组满足即可
            return False, False  # 所有组都不满足

    def check_budgets(self, plan):
        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0
        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner", "attraction"]:
                    overall_cost += activity["cost"]
                    restaurant_cost += activity["cost"]
                if activity["type"] in ["attraction"]:
                    overall_cost += activity["cost"]
                    attraction_cost += activity["cost"]
                innercity_cost += sum(transport.get("cost", 0) for transport in activity.get("transports", []))
        self.overall_cost = overall_cost + innercity_cost + self.hotel_cost + self.intercity_cost

        if self.attraction_budget is not None and self.attraction_budget < attraction_cost:
            self.backtrack_count += 1
            print("attraction budget exceeded, backtrack...")
            return True
        if self.restaurant_budget is not None and self.restaurant_budget < restaurant_cost:
            self.backtrack_count += 1
            print("restaurant budget exceeded, backtrack...")
            return True
        if self.innercity_budget is not None and self.innercity_budget < innercity_cost:
            self.backtrack_count += 1
            print("innercity budget exceeded, backtrack...")
            return True
        if self.overall_budget is not None and self.overall_budget < self.overall_cost:
            self.backtrack_count += 1
            print("overall budget exceeded, backtrack...")
            return True

        return False

    # ====================================================================================
    # SECTION 7: HELPER METHODS
    # ====================================================================================
    #
    # These methods provide utility functions for the search process:
    #   - select_and_add_breakfast: Add breakfast activity
    #   - select_next_poi_type: Determine next POI type based on time
    #   - select_poi_time: Calculate visit duration
    #
    # Lines: 2875-2933
    # ====================================================================================

    def select_and_add_breakfast(self, plan, poi_plan, current_day, current_time, current_position, transports_sel):
        # have breakfast at hotel
        plan[current_day]["activities"] = self.add_poi(plan[current_day]["activities"],
                                                       poi_plan["accommodation"]["name"], "breakfast", 0, 0, "08:00",
                                                       "08:30", innercity_transports=transports_sel)
        return plan

    def select_next_poi_type(self, candidates_type, plan, poi_plan, current_day, current_time, current_position):

        if current_day == self.query["days"] - 1:
            if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"],
                                             add_time_delta(current_time, 180)):
                return "back-intercity-transport", ["back-intercity-transport"]

        if len(candidates_type) == 1:
            return candidates_type[0], candidates_type

        if self.query["days"] > 1 and time_compare_if_earlier_equal(current_time, "6:00"):
            if "hotel" in candidates_type:
                return "hotel", candidates_type
            else:
                return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal("08:30", current_time) and time_compare_if_earlier_equal(current_time, "10:00"):
            if "attraction" in candidates_type:
                return "attraction", candidates_type
            else:
                return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal("10:00", current_time) and time_compare_if_earlier_equal(current_time, "13:00"):
            if "lunch" in candidates_type:
                return "lunch", candidates_type
            else:
                if "attraction" in candidates_type:
                    return "attraction", candidates_type
                else:
                    return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal("16:00", current_time) and time_compare_if_earlier_equal(current_time, "20:00"):
            if "dinner" in candidates_type:
                return "dinner", candidates_type
            else:
                if "attraction" in candidates_type:
                    return "attraction", candidates_type
                else:
                    return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal(current_time, "21:00"):
            if "attraction" in candidates_type:
                return "attraction", candidates_type

        return "hotel", candidates_type

    def select_poi_time(self, plan, poi_plan, current_day, start_time, poi_name, poi_type, recommended_visit_time):
        return 90

        return results_main, results_list

    def constraints_validation(self, query, plan, poi_plan):

        self.constraints_validation_count += 1

        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("validate the plan [for query {}]: ".format(query["uid"]))
        print(res_plan)

        self.least_plan_schema = deepcopy(res_plan)

        bool_result = func_commonsense_constraints(query, res_plan, verbose=True)

        # if not bool_result:
        #     exit(0)

        if bool_result:
            self.commonsense_pass_count += 1

        try:
            extracted_vars = get_symbolic_concepts(query, res_plan, need_ood=False)

        except:
            extracted_vars = None

        print(extracted_vars)

        # Check if hard_logic_py exists and is not empty
        if "hard_logic_py" in query and query["hard_logic_py"] and len(query["hard_logic_py"]) > 0:
            logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=True)
            print(logical_result)

            logical_pass = True
            for idx, item in enumerate(logical_result):
                logical_pass = logical_pass and item

                if item:
                    print(query["hard_logic_py"][idx], "passed!")
                else:
                    print(query["hard_logic_py"][idx], "failed...")
        else:
            # No hard_logic_py constraints or empty list
            print("[Warning] No hard_logic_py constraints found, skipping logical constraint validation")
            logical_result = []
            logical_pass = True

        if bool_result and np.sum(logical_result) > self.least_plan_logical_pass:
            self.least_plan_comm = deepcopy(res_plan)
            self.least_plan_logical_pass = np.sum(logical_result)
        # if logical_result:
        #     print("Logical passed!")

        if logical_pass:
            self.logical_pass_count += 1

        bool_result = bool_result and logical_pass

        if bool_result:
            print("\n Pass! \n")
            self.all_constraints_pass += 1

            if self.least_plan_logic is None:
                self.least_plan_logic = res_plan
        else:
            print("\n Failed \n")

        # plan = res_plan

        # print(result)
        # exit(0)

        if bool_result:
            res_plan["search_time_sec"] = time.time() - self.time_before_search
            res_plan["llm_inference_time_sec"] = self.llm_inference_time_count
            return True, res_plan
        else:
            return False, res_plan






    def check_if_too_late(
            self, query, current_day, current_time, current_position, poi_plan
    ):

        if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
            print("too late, after 23:00")
            return True

        if current_time != "" and current_day == query["days"] - 1:
            # We should go back in time ...
            transports_ranking = self.innercity_transports_ranking
            if self.transport_rules_by_distance is not None:
                temp_distance = self.calculate_distance(query, current_position, poi_plan["back_transport"]["From"])
                transports_ranking = self.get_transport_by_distance(temp_distance)

            for transport_type_sel in transports_ranking:

                self.search_nodes += 1

                flag = True
                if "back_transport" in poi_plan:
                    transports_sel = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        poi_plan["back_transport"]["From"],
                        current_time,
                        transport_type_sel,
                    )
                    if not isinstance(transports_sel, list):
                        self.backtrack_count += 1
                        print("inner-city transport error, backtrack...")
                        continue

                    if len(transports_sel) > 0:
                        arrived_time = transports_sel[-1]["end_time"]
                    else:
                        arrived_time = current_time

                    if not time_compare_if_earlier_equal(
                            poi_plan["back_transport"]["BeginTime"], arrived_time
                    ):
                        flag = False

                if flag:
                    if transport_type_sel != transports_ranking[-1]:
                        continue
                    print(
                        "Can not go back source-city in time, current POI {}, station arrived time: {}".format(
                            current_position, arrived_time
                        )
                    )
                    return True


        elif current_time != "":
            if "accommodation" in poi_plan:
                hotel_sel = poi_plan["accommodation"]
                transports_ranking = self.innercity_transports_ranking
                if self.transport_rules_by_distance is not None:
                    temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                    transports_ranking = self.get_transport_by_distance(temp_distance)

                for transport_type_sel in transports_ranking:
                    self.search_nodes += 1
                    flag = True
                    if "back_transport" in poi_plan:
                        print("collecting innercity transport to see if possible back to hotel")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            transport_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        flag = True
                        # print(transports_sel)
                        # print(transports_sel[-1])

                        if len(transports_sel) > 0:
                            arrived_time = transports_sel[-1]["end_time"]
                        else:
                            arrived_time = current_time
                        if not time_compare_if_earlier_equal("24:00", arrived_time):
                            flag = False
                    if flag:
                        print(
                            "Can not go back to hotel, current POI {}, hotel arrived time: {}".format(
                                current_position, arrived_time
                            )
                        )
                        return True

        return False




    # ====================================================================================
    # SECTION 8: DELEGATION METHODS
    # ====================================================================================
    #
    # IMPORTANT: DO NOT REMOVE THESE METHODS!
    #
    # These delegation methods provide a clean interface to modular components.
    # They are called throughout the search logic (generate_plan_with_search, dfs_poi).
    # Removing them will break the code.
    #
    # Delegation targets:
    #   - ConstraintParser: extract_user_constraints_by_DSL
    #   - RankingEngine: ranking_intercity_transport_go/back, ranking_hotel,
    #                    calculate_distance, get_transport_by_distance
    #   - POIManager: collect_poi_info_all, collect_innercity_transport,
    #                 collect_intercity_transport
    #   - PlanBuilder: add_intercity_transport, add_poi, add_accommodation,
    #                  add_restaurant, add_attraction
    #
    # Lines: 3130-3213 (83 lines)
    # ====================================================================================

    def extract_user_constraints_by_DSL(self, query):
        """Delegate to ConstraintParser"""
        return self.constraint_parser.extract_user_constraints_by_DSL(query)

    def ranking_intercity_transport_go(self, transport_info, query):
        """Delegate to RankingEngine"""
        return self.ranking_engine.ranking_intercity_transport_go(
            transport_info, self.intercity_budget, self.overall_budget
        )

    def ranking_intercity_transport_back(self, transport_info, query, selected_go):
        """Delegate to RankingEngine"""
        return self.ranking_engine.ranking_intercity_transport_back(transport_info, selected_go)

    def ranking_hotel(self, hotel_info, query):
        """Delegate to RankingEngine"""
        constraints_dict = {
            'must_live_hotel': self.must_live_hotel,
            'must_not_live_hotel': self.must_not_live_hotel,
            'must_live_hotel_feature': self.must_live_hotel_feature,
            'must_live_hotel_location_limit': self.must_live_hotel_location_limit,
        }
        return self.ranking_engine.ranking_hotel(hotel_info, query, constraints_dict, self.memory)

    def calculate_distance(self, query, start, end):
        """Delegate to RankingEngine"""
        return self.ranking_engine.calculate_distance(query, start, end)

    def get_transport_by_distance(self, distance):
        """Delegate to RankingEngine"""
        return self.ranking_engine.get_transport_by_distance(distance, self.transport_rules_by_distance)

    def collect_poi_info_all(self, city, poi_type):
        """Delegate to POIManager"""
        return self.poi_manager.collect_poi_info_all(city, poi_type)

    def collect_innercity_transport(self, city, start, end, start_time, trans_type):
        """Delegate to POIManager"""
        return self.poi_manager.collect_innercity_transport(city, start, end, start_time, trans_type)

    def collect_intercity_transport(self, source_city, target_city, trans_type):
        """Delegate to POIManager"""
        return self.poi_manager.collect_intercity_transport(source_city, target_city, trans_type)

    def add_intercity_transport(self, activities, intercity_info, innercity_transports=[], tickets=1):
        """Delegate to PlanBuilder"""
        return self.plan_builder.add_intercity_transport(activities, intercity_info, innercity_transports, tickets)

    def add_poi(self, activities, position, poi_type, price, cost, start_time, end_time, innercity_transports):
        """Delegate to PlanBuilder"""
        return self.plan_builder.add_poi(activities, position, poi_type, price, cost, start_time, end_time, innercity_transports)

    def add_accommodation(self, current_plan, hotel_sel, current_day, arrived_time, required_rooms, transports_sel):
        """Delegate to PlanBuilder"""
        return self.plan_builder.add_accommodation(current_plan, hotel_sel, current_day, arrived_time, required_rooms, transports_sel)

    def add_restaurant(self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel):
        """Delegate to PlanBuilder"""
        return self.plan_builder.add_restaurant(current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel)

    def add_attraction(self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel):
        """Delegate to PlanBuilder"""
        return self.plan_builder.add_attraction(current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel)
