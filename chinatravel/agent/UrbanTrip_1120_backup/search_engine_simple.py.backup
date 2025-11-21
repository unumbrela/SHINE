"""
SearchEngine module for UrbanTrip agent
Contains the core DFS search algorithm and plan generation logic
"""

import sys
import os
import time
import pandas as pd
import json
from copy import deepcopy

from chinatravel.agent.UrbanTrip.urbantrip_utils import (
    time_compare_if_earlier_equal,
    add_time_delta,
    get_time_delta,
)


class SearchEngine:
    """
    SearchEngine handles the core search logic for generating travel plans.
    It implements a depth-first search (DFS) algorithm to explore the solution space.
    """

    def __init__(self, agent):
        """
        Initialize SearchEngine with reference to parent agent

        Args:
            agent: Reference to the parent UrbanTrip agent instance
        """
        self.agent = agent

    def generate_plan_with_search(self, query):
        """
        Main entry point for plan generation with search
        Initializes search state and starts the DFS process

        Args:
            query: User query containing travel requirements

        Returns:
            tuple: (success: bool, plan: dict)
        """
        # 初始化计时器和计数器
        self.agent.time_before_search = time.time()  # 记录搜索开始时间
        self.agent.llm_inference_time_count = 0  # llm推理时间

        # reset the cache before searching
        poi_plan = {}  # 存储当前计划的 POI 信息
        self.agent.restaurants_visiting = []  # 正在访问的餐厅列表
        self.agent.attractions_visiting = []  # 正在访问的景点列表
        self.agent.food_type_visiting = []  # 正在访问的食物类型列表
        self.agent.spot_type_visiting = []  # 正在访问的景点类型列表
        self.agent.attraction_names_visiting = []  # 正在访问的景点名称列表
        self.agent.restaurant_names_visiting = []  # 正在访问的餐厅名称列表

        self.agent.llm_rec_format_error = 0  # llm推荐格式错误计数
        self.agent.llm_rec_count = 0  # llm推荐计数
        self.agent.search_nodes = 0  # 搜索节点计数
        self.agent.backtrack_count = 0  # 回溯计数

        self.agent.constraints_validation_count = 0  # 约束验证计数
        self.agent.commonsense_pass_count = 0  # 常识通过计数
        self.agent.logical_pass_count = 0  # 逻辑通过计数
        self.agent.all_constraints_pass = 0  # 所有约束通过计数

        # 存储通过逻辑检查的次优计划
        self.agent.least_plan_schema, self.agent.least_plan_comm, self.agent.least_plan_logic = None, None, None
        self.agent.least_plan_logical_pass = -1

        # 提取用户需求
        # 获取用户约束信息
        constraints_json, requirement_list = self.agent.constraint_parser.extract_user_constraints_by_DSL(query)

        # 用户偏好约束字段赋值
        self.agent.all_satisfy = constraints_json.get("all_satisfy", None)

        # attractions
        self.agent.must_see_attraction = constraints_json.get("must_see_attraction", None)
        self.agent.must_see_attraction_type = constraints_json.get("must_see_attraction_type", None)
        self.agent.must_not_see_attraction = constraints_json.get("must_not_see_attraction", None)
        self.agent.must_not_see_attraction_type = constraints_json.get("must_not_see_attraction_type", None)
        self.agent.only_free_attractions = constraints_json.get("only_free_attractions", None)

        # restaurant
        self.agent.must_visit_restaurant = constraints_json.get("must_visit_restaurant", None)
        self.agent.must_visit_restaurant_type = constraints_json.get("must_visit_restaurant_type", None)
        self.agent.must_not_visit_restaurant = constraints_json.get("must_not_visit_restaurant", None)
        self.agent.must_not_visit_restaurant_type = constraints_json.get("must_not_visit_restaurant_type", None)

        self.agent.activities_stay_time_dict = constraints_json.get("activities_stay_time_dict", None)
        self.agent.activities_arrive_time_dict = constraints_json.get("activities_arrive_time_dict", None)
        self.agent.activities_leave_time_dict = constraints_json.get("activities_leave_time_dict", None)

        # hotel
        self.agent.must_live_hotel = constraints_json.get("must_live_hotel", None)
        self.agent.must_not_live_hotel = constraints_json.get("must_not_live_hotel", None)
        self.agent.must_live_hotel_feature = constraints_json.get("must_live_hotel_feature", None)
        self.agent.must_live_hotel_location_limit = constraints_json.get("must_live_hotel_location_limit", None)
        # hotel room/bed num
        self.agent.bed_number = constraints_json.get("bed_number", None)  # 例如 ['单床', '双床']
        self.agent.room_number = constraints_json.get("room_number", None)

        # innercity transport
        self.agent.must_innercity_transport = constraints_json.get("must_innercity_transport", None)
        self.agent.must_not_innercity_transport = constraints_json.get("must_not_innercity_transport", None)

        # transport rules
        self.agent.transport_rules_by_distance = constraints_json.get("transport_rules_by_distance", None)

        # intercity transport
        self.agent.must_depart_transport = constraints_json.get("must_depart_transport", None)
        self.agent.must_return_transport = constraints_json.get("must_return_transport", None)
        self.agent.must_not_depart_transport = constraints_json.get("must_not_depart_transport", None)
        self.agent.must_not_return_transport = constraints_json.get("must_not_return_transport", None)

        # 提取所需预算
        self.agent.attraction_budget = constraints_json.get("attraction_budget", None)
        self.agent.restaurant_budget = constraints_json.get("restaurant_budget", None)
        self.agent.hotel_budget = constraints_json.get("hotel_budget", None)
        self.agent.innercity_budget = constraints_json.get("innercity_budget", None)
        self.agent.intercity_budget = constraints_json.get("intercity_budget", None)
        self.agent.overall_budget = constraints_json.get("overall_budget", None)

        self.agent.requirement_list = requirement_list
        self.agent.all_satisfy_flag = False # 是否满足用户需求
        self.agent.too_many_backtrack = False
        self.agent.stop_search = False
        self.agent.default_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": [],
        }

        if self.agent.transport_rules_by_distance is not None:
            if isinstance(self.agent.transport_rules_by_distance, str):
                    self.agent.transport_rules_by_distance = json.loads(self.agent.transport_rules_by_distance)
            elif isinstance(self.agent.transport_rules_by_distance, dict):
                self.agent.transport_rules_by_distance = [self.agent.transport_rules_by_distance]
            elif isinstance(self.agent.transport_rules_by_distance, list):
                self.agent.transport_rules_by_distance = [
                    rule for rule in self.agent.transport_rules_by_distance if isinstance(rule, dict)
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
        print(f"{self.agent.requirement_list}")

        query_room_number = self.agent.room_number
        query_room_numbed = self.agent.bed_number

        print(f"query room number: {query_room_number}")
        print(f"query room numbed: {query_room_numbed}")

        # 收集去程和返程的城际火车交通选项
        train_go = self.agent.collect_intercity_transport(source_city, target_city, "train")
        train_back = self.agent.collect_intercity_transport(target_city, source_city, "train")

        # 收集去程和返程的城际飞机交通选项
        flight_go = self.agent.collect_intercity_transport(
            source_city, target_city, "airplane"
        )
        flight_back = self.agent.collect_intercity_transport(
            target_city, source_city, "airplane"
        )

        # must_not_depart_transport: 去程不允许的方式
        if self.agent.must_not_depart_transport is not None and "train" in self.agent.must_not_depart_transport:
            train_go = pd.DataFrame()  # 置空
        elif self.agent.must_not_depart_transport is not None and "airplane" in self.agent.must_not_depart_transport:
            flight_go = pd.DataFrame()

        # must_not_return_transport: 返程不允许的方式
        if self.agent.must_not_return_transport is not None and "train" in self.agent.must_not_return_transport:
            train_back = pd.DataFrame()
        elif self.agent.must_not_return_transport is not None and "airplane" in self.agent.must_not_return_transport:
            flight_back = pd.DataFrame()

        # must_depart_transport: 去程必须的方式
        if self.agent.must_depart_transport is not None and "train" in self.agent.must_depart_transport:
            flight_go = pd.DataFrame()
        elif self.agent.must_depart_transport is not None and "airplane" in self.agent.must_depart_transport:
            train_go = pd.DataFrame()

        # must_return_transport: 返程必须的方式
        if self.agent.must_return_transport is not None and "train" in self.agent.must_return_transport:
            flight_back = pd.DataFrame()
        elif self.agent.must_return_transport is not None and "airplane" in self.agent.must_return_transport:
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
        if self.agent.debug:
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
        ranking_go = self.agent.ranking_intercity_transport_go(go_info, query)

        # 对酒店进行排序
        ranking_hotel = self.agent.ranking_hotel(self.agent.memory["accommodations"], query)

        default_hotel = (
            self.agent.memory["accommodations"]
            .sort_values(by="price")
            .index
            .tolist()
        )

        # 根据查询对市内交通进行排序
        self.agent.innercity_transports_ranking = ["metro", "taxi", "walk"]
        if self.agent.must_innercity_transport is not None:
            self.agent.innercity_transports_ranking = self.agent.must_innercity_transport[:]
        if self.agent.must_not_innercity_transport is not None:
            self.agent.innercity_transports_ranking = [
                t for t in self.agent.innercity_transports_ranking
                if t not in self.agent.must_not_innercity_transport
            ]

        # 遍历排序后的去程交通
        intercity_budget_msg = "intercity budget not satisfied, backtrack..."
        intercity_budget_count = 0
        for go_i in ranking_go:
            go_info_i = go_info.iloc[go_i]  # 获取当前去程交通信息
            if pd.isna(go_info_i["Cost"]):
                continue
            poi_plan["go_transport"] = go_info_i  # 将其添加到计划中
            self.agent.search_nodes += 1

            # 对返程城际交通进行排序（依赖于去程信息）
            ranking_back = self.agent.ranking_intercity_transport_back(
                back_info, query, go_info_i
            )
            # 遍历排序后的返程交通
            for back_i in ranking_back:
                if time.time() > self.agent.time_before_search + self.agent.TIME_CUT:
                    self.agent.default_plan["backtrack_count"] = self.agent.backtrack_count
                    return True, self.agent.default_plan

                back_info_i = back_info.iloc[back_i]  # 获取当前返程交通信息
                if pd.isna(back_info_i["Cost"]):
                    continue
                poi_plan["back_transport"] = back_info_i  # 将其添加到计划中

                # 检查城际交通预算
                self.agent.intercity_cost = (go_info_i["Cost"] + back_info_i["Cost"]) * query[
                    "people_number"
                ]
                if (
                        self.agent.intercity_budget is not None
                        and self.agent.intercity_cost > self.agent.intercity_budget
                ):
                    intercity_budget_count += 1
                    if intercity_budget_count <= 3:
                        print(intercity_budget_msg)
                        self.agent.backtrack_count += 1
                    continue

                self.agent.search_nodes += 1

                # 遍历排序后的酒店
                for hotel_i in ranking_hotel:
                    if time.time() > self.agent.time_before_search + self.agent.TIME_CUT:
                        self.agent.default_plan["backtrack_count"] = self.agent.backtrack_count
                        return True, self.agent.default_plan

                    hotel_sel = self.agent.memory["accommodations"].iloc[hotel_i]
                    if pd.isna(hotel_sel["price"]):
                        continue
                    poi_plan["hotel"] = hotel_sel
                    self.agent.search_nodes += 1

                    # 预先检查房间数和床位数
                    # 计算所需房间数
                    required_rooms = 1
                    required_room_type = 2
                    if self.agent.room_number is not None:
                        required_rooms = self.agent.room_number
                    elif self.agent.bed_number is not None:
                        bed_map = {"单床": 1, "双床": 2}
                        required_rooms = self.agent.bed_number
                        if self.agent.bed_number in bed_map:
                            required_room_type = bed_map[self.agent.bed_number]
                    else:
                        people_number = query["people_number"]
                        required_rooms = (people_number + 1) // 2
                        if people_number == 1:
                            required_room_type = 1

                    # 检查酒店房间数是否满足要求
                    if required_room_type == 1:
                        if hotel_sel.get("single_bed_room", 0) < required_rooms:
                            continue
                    elif required_room_type == 2:
                        if hotel_sel.get("double_bed_room", 0) < required_rooms:
                            continue

                    # 计算酒店总费用
                    self.agent.hotel_cost = hotel_sel["price"] * required_rooms * (query["days"] - 1)
                    if self.agent.hotel_budget is not None and self.agent.hotel_cost > self.agent.hotel_budget:
                        self.agent.backtrack_count += 1
                        print("hotel budget exceeded, backtrack...")
                        continue

                    # 调用 dfs_poi 进行深度优先搜索
                    plan = []
                    success, plan = self.dfs_poi(
                        query, poi_plan, plan, current_time="", current_position="", current_day=0
                    )

                    if success:
                        return True, plan

        # 如果没有找到满足所有约束的计划，返回最优次优计划
        print("\n[Search Complete] No solution satisfying all constraints found.")
        print(f"Search Statistics:")
        print(f"  - Search nodes: {self.agent.search_nodes}")
        print(f"  - Backtrack count: {self.agent.backtrack_count}")
        print(f"  - Commonsense pass: {self.agent.commonsense_pass_count}")
        print(f"  - Logical pass: {self.agent.logical_pass_count}")

        return False, {}

    def dfs_poi(self, query, poi_plan, plan, current_time, current_position, current_day=0):
        """
        Depth-first search for POI planning
        Recursively builds the travel plan by selecting POIs and managing constraints

        This is the core search algorithm - it's a large method due to the complexity
        of the travel planning problem with multiple constraints and decision points.

        Args:
            query: User query
            poi_plan: Current POI plan (hotels, transports, etc.)
            plan: Current plan being built
            current_time: Current time in the itinerary
            current_position: Current position
            current_day: Current day (0-indexed)

        Returns:
            tuple: (success: bool, plan: dict)
        """
        print("----------------------------------calling dfs_poi-----------------------------------------")
        # print(f"plan: {plan}")
        print(f"current_day: {current_day}")
        print(f"current_time: {current_time}")
        print(f"current_position: {current_position}")

        print(self.agent.backtrack_count)
        # if self.agent.backtrack_count > 5800 or time.time() - self.agent.time_before_search + 20 > self.agent.TIME_CUT + self.agent.llm_inference_time_count:
        #     self.agent.too_many_backtrack = True
        if time.time() - self.agent.time_before_search + 20 > self.agent.TIME_CUT + self.agent.llm_inference_time_count:
            self.agent.too_many_backtrack = True

        if not self.agent.all_satisfy_flag and not self.agent.too_many_backtrack:
            ok, backtrack = self.agent.check_requirement(plan)
            if ok:
                self.agent.all_satisfy_flag = True
            if backtrack:
                self.agent.backtrack_count += 1
                print("requirements can not be satisfied, backtrack...")
                return False, plan

        self.agent.search_nodes += 1
        # 检查是否超时
        if self.agent.stop_search:
            self.agent.default_plan["backtrack_count"] = self.agent.backtrack_count
            return True, self.agent.default_plan
        if time.time() - self.agent.time_before_search > self.agent.TIME_CUT + self.agent.llm_inference_time_count:
            self.agent.stop_search = True
            self.agent.default_plan["backtrack_count"] = self.agent.backtrack_count
            return True, self.agent.default_plan

        # 检查当前时间是否太晚，无法前往酒店或返程交通
        print("check if too late")
        if not self.agent.too_many_backtrack:
            if self.agent.check_if_too_late(query, current_day, current_time, current_position, poi_plan):
                self.agent.backtrack_count += 1
                print("The current time is too late to go hotel or back-transport, backtrack...")
                return False, plan

        # 处理第一天的去程城际交通
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]  # 初始化第一天的活动列表
            # 添加去程城际交通活动
            plan[current_day]["activities"] = self.agent.add_intercity_transport(
                plan[current_day]["activities"],
                poi_plan["go_transport"],
                innercity_transports=[],
                tickets=self.agent.query["people_number"],
            )

            print(plan)

            new_time = poi_plan["go_transport"]["EndTime"]  # 更新当前时间为去程交通的结束时间
            new_position = poi_plan["go_transport"]["To"]  # 更新当前位置为目的地（车站）

            # 递归调用 dfs_poi 进行后续规划
            success, plan = self.dfs_poi(query, poi_plan, plan, new_time, new_position, current_day)

            if success:
                return True, plan
            else:
                self.agent.backtrack_count += 1
                print("No solution for the given Go Transport, backtrack...")
                return False, plan

        # breakfast
        if (current_time == "00:00" and current_day != query["days"] - 1) or (current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal("11:30", poi_plan["back_transport"]["BeginTime"])):
            if len(plan) < current_day + 1:
                plan.append({"day": current_day + 1, "activities": []})  # 如果是新的一天，添加新的活动列表

            self.agent.search_nodes += 1
            # 选择并添加早餐活动
            plan = self.agent.select_and_add_breakfast(plan, poi_plan, current_day, current_time, current_position, [])

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
                self.agent.backtrack_count += 1
                print("No solution for the given Breakfast, backtrack...")
                return False, plan
        elif current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], "11:30"):
            candidates_type = ["back-intercity-transport"]
        else:  # 如果当前时间不是 "00:00"，说明一天已经开始
            haved_lunch_today, haved_dinner_today = False, False

            for act_i in plan[current_day]["activities"]:
                if act_i["type"] == "lunch":
                    haved_lunch_today = True
                if act_i["type"] == "dinner":
                    haved_dinner_today = True

            # 确定候选活动类型列表
            candidates_type = self.agent.select_next_poi_type(
                ["attraction", "lunch", "dinner", "accommodation", "back-intercity-transport"],
                plan,
                poi_plan,
                current_day,
                current_time,
                current_position,
            )

        # 遍历候选活动类型
        for poi_type_sel in candidates_type:
            print(f"poi_type_sel: {poi_type_sel}")

            if poi_type_sel == "accommodation":
                # 酒店入住逻辑
                hotel_sel = poi_plan["hotel"]

                # 计算所需房间数
                required_rooms = 1
                required_room_type = 2
                if self.agent.room_number is not None:
                    required_rooms = self.agent.room_number
                elif self.agent.bed_number is not None:
                    bed_map = {"单床": 1, "双床": 2}
                    required_rooms = self.agent.bed_number
                    if self.agent.bed_number in bed_map:
                        required_room_type = bed_map[self.agent.bed_number]
                else:
                    people_number = query["people_number"]
                    required_rooms = (people_number + 1) // 2
                    if people_number == 1:
                        required_room_type = 1

                # 收集市内交通（前往酒店）
                # print(f"current_position: {current_position}")
                # print(f"hotel_sel['name']: {hotel_sel['name']}")
                # print(f"current_time: {current_time}")

                transport_type_sel = self.agent.innercity_transports_ranking[0]
                transports_sel = self.agent.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    hotel_sel["name"],
                    current_time,
                    transport_type_sel,
                )

                if not isinstance(transports_sel, list):
                    self.agent.backtrack_count += 1
                    print("inner-city transport error, backtrack...")
                    continue

                # print("transports_sel", transports_sel)

                if len(transports_sel) > 0:
                    arrived_time = transports_sel[-1]["end_time"]
                else:
                    arrived_time = current_time

                # 添加酒店入住活动
                plan = self.agent.add_accommodation(
                    plan, hotel_sel, current_day, arrived_time, required_rooms, transports_sel
                )

                # 更新时间和位置，进入下一天
                new_time = "00:00"
                new_position = hotel_sel["name"]
                new_day = current_day + 1

                # 递归调用 dfs_poi 进行下一天的规划
                success, plan = self.dfs_poi(
                    query, poi_plan, plan, new_time, new_position, new_day
                )

                if success:
                    return True, plan

                # 如果失败，回溯
                self.agent.backtrack_count += 1
                print("No solution for the given Hotel, backtrack...")
                plan[current_day]["activities"].pop()
                continue

            elif poi_type_sel == "back-intercity-transport":
                # 返程城际交通逻辑
                back_transport = poi_plan["back_transport"]

                # 收集市内交通（前往返程车站/机场）
                transport_type_sel = self.agent.innercity_transports_ranking[0]

                transports_sel = self.agent.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    back_transport["From"],
                    current_time,
                    transport_type_sel,
                )
                if not isinstance(transports_sel, list):
                    self.agent.backtrack_count += 1
                    print("inner-city transport error, backtrack...")
                    continue

                # 检查是否能在返程交通开始前到达车站
                if len(transports_sel) > 0:
                    arrived_time = transports_sel[-1]["end_time"]
                else:
                    arrived_time = current_time

                if not time_compare_if_earlier_equal(
                        add_time_delta(arrived_time, {"hours": 2, "minutes": 0}),
                        back_transport["BeginTime"],
                ):
                    self.agent.backtrack_count += 1
                    print(
                        "Can not arrive at the station/airport on time for back transport, backtrack..."
                    )
                    continue

                # 添加返程城际交通活动
                plan[current_day]["activities"] = self.agent.add_intercity_transport(
                    plan[current_day]["activities"],
                    back_transport,
                    innercity_transports=transports_sel,
                    tickets=query["people_number"],
                )

                # 构建完整计划
                complete_plan = {
                    "people_number": query["people_number"],
                    "start_city": query["start_city"],
                    "target_city": query["target_city"],
                    "itinerary": plan,
                }

                # 验证约束
                print("Validating constraints...")
                validate_result = self.agent.constraints_validation(query, complete_plan, poi_plan)

                if validate_result["pass_all"]:
                    print("All constraints satisfied! Plan found.")
                    return True, complete_plan
                else:
                    # 保存次优计划
                    if validate_result["pass_schema"]:
                        print("Schema constraints passed, saving as least_plan_schema")
                        self.agent.least_plan_schema = deepcopy(complete_plan)

                    if validate_result["pass_commonsense"]:
                        print("Commonsense constraints passed, saving as least_plan_comm")
                        self.agent.least_plan_comm = deepcopy(complete_plan)

                    if validate_result["pass_logic"]:
                        print(
                            f"Logical constraints passed ({validate_result['logical_pass_count']} constraints), saving as least_plan_logic"
                        )
                        if validate_result["logical_pass_count"] > self.agent.least_plan_logical_pass:
                            self.agent.least_plan_logic = deepcopy(complete_plan)
                            self.agent.least_plan_logical_pass = validate_result["logical_pass_count"]

                    self.agent.backtrack_count += 1
                    print("Constraints not fully satisfied, backtracking...")
                    plan[current_day]["activities"].pop()
                    continue

            elif poi_type_sel in ["lunch", "dinner"]:
                # 餐厅选择逻辑
                # 筛选满足约束的餐厅
                candidate_restaurants = self.agent.memory["restaurants"].copy()

                # 应用 must_not_visit_restaurant 约束
                if self.agent.must_not_visit_restaurant is not None:
                    candidate_restaurants = candidate_restaurants[
                        ~candidate_restaurants["name"].isin(self.agent.must_not_visit_restaurant)
                    ]

                # 应用 must_not_visit_restaurant_type 约束
                if self.agent.must_not_visit_restaurant_type is not None:
                    candidate_restaurants = candidate_restaurants[
                        ~candidate_restaurants["cuisine"].isin(self.agent.must_not_visit_restaurant_type)
                    ]

                # 应用 must_visit_restaurant 约束（优先级最高）
                if self.agent.must_visit_restaurant is not None:
                    unvisited_must_restaurants = [
                        r for r in self.agent.must_visit_restaurant
                        if r not in self.agent.restaurant_names_visiting
                    ]
                    if unvisited_must_restaurants:
                        candidate_restaurants = candidate_restaurants[
                            candidate_restaurants["name"].isin(unvisited_must_restaurants)
                        ]

                # 应用 must_visit_restaurant_type 约束
                elif self.agent.must_visit_restaurant_type is not None:
                    unvisited_must_types = [
                        t for t in self.agent.must_visit_restaurant_type
                        if t not in self.agent.food_type_visiting
                    ]
                    if unvisited_must_types:
                        candidate_restaurants = candidate_restaurants[
                            candidate_restaurants["cuisine"].isin(unvisited_must_types)
                        ]

                # 按价格排序（优先选择便宜的）
                candidate_restaurants = candidate_restaurants.sort_values(by="price")

                # 遍历候选餐厅
                for restaurant_idx in candidate_restaurants.index:
                    restaurant_sel = candidate_restaurants.loc[restaurant_idx]

                    # 收集市内交通
                    transport_type_sel = self.agent.innercity_transports_ranking[0]
                    transports_sel = self.agent.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        restaurant_sel["name"],
                        current_time,
                        transport_type_sel,
                    )

                    if not isinstance(transports_sel, list):
                        continue

                    if len(transports_sel) > 0:
                        arrived_time = transports_sel[-1]["end_time"]
                    else:
                        arrived_time = current_time

                    # 添加餐厅活动
                    plan = self.agent.add_restaurant(
                        plan, poi_type_sel, restaurant_sel, current_day, arrived_time, transports_sel
                    )

                    # 更新访问记录
                    self.agent.restaurant_names_visiting.append(restaurant_sel["name"])
                    self.agent.food_type_visiting.append(restaurant_sel["cuisine"])

                    new_time = plan[current_day]["activities"][-1]["end_time"]
                    new_position = restaurant_sel["name"]

                    # 递归调用
                    success, plan = self.dfs_poi(
                        query, poi_plan, plan, new_time, new_position, current_day
                    )

                    if success:
                        return True, plan

                    # 回溯
                    self.agent.backtrack_count += 1
                    print(f"No solution for restaurant {restaurant_sel['name']}, backtrack...")
                    plan[current_day]["activities"].pop()
                    self.agent.restaurant_names_visiting.pop()
                    self.agent.food_type_visiting.pop()

                # 如果所有餐厅都失败，返回 False
                print(f"No valid {poi_type_sel} found, backtracking...")
                return False, plan

            elif poi_type_sel == "attraction":
                # 景点选择逻辑
                candidate_attractions = self.agent.memory["attractions"].copy()

                # 应用 must_not_see_attraction 约束
                if self.agent.must_not_see_attraction is not None:
                    candidate_attractions = candidate_attractions[
                        ~candidate_attractions["name"].isin(self.agent.must_not_see_attraction)
                    ]

                # 应用 must_not_see_attraction_type 约束
                if self.agent.must_not_see_attraction_type is not None:
                    candidate_attractions = candidate_attractions[
                        ~candidate_attractions["type"].isin(self.agent.must_not_see_attraction_type)
                    ]

                # 应用 only_free_attractions 约束
                if self.agent.only_free_attractions:
                    candidate_attractions = candidate_attractions[candidate_attractions["price"] == 0]

                # 应用 must_see_attraction 约束（优先级最高）
                if self.agent.must_see_attraction is not None:
                    unvisited_must_attractions = [
                        a for a in self.agent.must_see_attraction
                        if a not in self.agent.attraction_names_visiting
                    ]
                    if unvisited_must_attractions:
                        candidate_attractions = candidate_attractions[
                            candidate_attractions["name"].isin(unvisited_must_attractions)
                        ]

                # 应用 must_see_attraction_type 约束
                elif self.agent.must_see_attraction_type is not None:
                    unvisited_must_types = [
                        t for t in self.agent.must_see_attraction_type
                        if t not in self.agent.spot_type_visiting
                    ]
                    if unvisited_must_types:
                        candidate_attractions = candidate_attractions[
                            candidate_attractions["type"].isin(unvisited_must_types)
                        ]

                # 按价格排序（优先选择免费或便宜的）
                candidate_attractions = candidate_attractions.sort_values(by="price")

                # 遍历候选景点
                for attraction_idx in candidate_attractions.index:
                    attraction_sel = candidate_attractions.loc[attraction_idx]

                    # 收集市内交通
                    transport_type_sel = self.agent.innercity_transports_ranking[0]
                    transports_sel = self.agent.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        attraction_sel["name"],
                        current_time,
                        transport_type_sel,
                    )

                    if not isinstance(transports_sel, list):
                        continue

                    if len(transports_sel) > 0:
                        arrived_time = transports_sel[-1]["end_time"]
                    else:
                        arrived_time = current_time

                    # 添加景点活动
                    plan = self.agent.add_attraction(
                        plan, poi_type_sel, attraction_sel, current_day, arrived_time, transports_sel
                    )

                    # 更新访问记录
                    self.agent.attraction_names_visiting.append(attraction_sel["name"])
                    self.agent.spot_type_visiting.append(attraction_sel["type"])

                    new_time = plan[current_day]["activities"][-1]["end_time"]
                    new_position = attraction_sel["name"]

                    # 递归调用
                    success, plan = self.dfs_poi(
                        query, poi_plan, plan, new_time, new_position, current_day
                    )

                    if success:
                        return True, plan

                    # 回溯
                    self.agent.backtrack_count += 1
                    print(f"No solution for attraction {attraction_sel['name']}, backtrack...")
                    plan[current_day]["activities"].pop()
                    self.agent.attraction_names_visiting.pop()
                    self.agent.spot_type_visiting.pop()

                # 如果所有景点都失败，返回 False
                print("No valid attraction found, backtracking...")
                return False, plan

        # 如果所有候选类型都失败，返回 False
        print("No valid POI type found for current state, backtracking...")
        return False, plan
