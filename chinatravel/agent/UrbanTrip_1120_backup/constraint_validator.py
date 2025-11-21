"""Constraint Validator Module for UrbanTrip Agent

This module handles validation of travel plans against various constraints
including budget, must-see/must-not-see requirements, and symbolic constraints.
"""

from chinatravel.symbol_verification.commonsense_constraint import func_commonsense_constraints
from chinatravel.symbol_verification.hard_constraint import get_symbolic_concepts, evaluate_constraints_py


class ConstraintValidator:
    """Validates travel plans against user constraints"""

    def __init__(self, memory, query):
        """
        Args:
            memory: Dictionary containing attractions and restaurants info
            query: Query dictionary
        """
        self.memory = memory
        self.query = query

        # Tracking variables
        self.backtrack_count = 0
        self.constraints_validation_count = 0
        self.all_satisfy_flag = True

        # Cost tracking
        self.overall_cost = 0
        self.hotel_cost = 0
        self.intercity_cost = 0

        # Constraint fields (will be set from parsed constraints)
        self.attraction_budget = None
        self.restaurant_budget = None
        self.innercity_budget = None
        self.overall_budget = None

        # Fallback plans
        self.least_plan_logic = None
        self.least_plan_comm = None
        self.least_plan_schema = None

        # Requirement list
        self.all_satisfy = True
        self.requirement_list = []

    def set_constraints(self, constraints_dict, requirement_list, all_satisfy):
        """Set constraint values from parsed DSL

        Args:
            constraints_dict: Main constraints dictionary
            requirement_list: List of constraint groups
            all_satisfy: Boolean indicating AND vs OR logic
        """
        self.all_satisfy = all_satisfy
        self.requirement_list = requirement_list

        # Set budget constraints
        self.attraction_budget = constraints_dict.get("attraction_budget")
        self.restaurant_budget = constraints_dict.get("restaurant_budget")
        self.innercity_budget = constraints_dict.get("innercity_budget")
        self.overall_budget = constraints_dict.get("overall_budget")

    def set_costs(self, hotel_cost, intercity_cost):
        """Set fixed costs for hotel and intercity transport

        Args:
            hotel_cost: Total hotel cost
            intercity_cost: Total intercity transport cost
        """
        self.hotel_cost = hotel_cost
        self.intercity_cost = intercity_cost

    def check_constraint(self, plan, constraints, innercity_transports_ranking):
        """Check if plan satisfies a single constraint group

        Args:
            plan: Travel plan (list of days with activities)
            constraints: Single constraint dictionary
            innercity_transports_ranking: List of allowed transport modes

        Returns:
            tuple: (ok: bool, backtrack: bool)
                ok: True if constraint is satisfied
                backtrack: True if should immediately backtrack
        """
        # Initialize visit tracking
        visited_attractions = set()
        visited_attraction_types = set()
        visited_restaurants = set()
        visited_restaurant_types = set()

        logic_fail = False
        backtrack = False

        # Calculate costs
        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0

        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner"]:
                    overall_cost += activity["cost"]
                    restaurant_cost += activity["cost"]
                if activity["type"] in ["attraction"]:
                    overall_cost += activity["cost"]
                    attraction_cost += activity["cost"]
                innercity_cost += sum(transport.get("cost", 0) for transport in activity.get("transports", []))

        self.overall_cost = overall_cost + innercity_cost + self.hotel_cost + self.intercity_cost

        # Check budget constraints
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

        # Process activities and check must_not constraints
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

        # Check must-see requirements (don't backtrack immediately)
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
            if not set(constraints["must_innercity_transport"]).issubset(set(innercity_transports_ranking)):
                logic_fail = True

        if "must_not_innercity_transport" in constraints:
            if set(constraints["must_not_innercity_transport"]) & set(innercity_transports_ranking):
                logic_fail = True

        return (not backtrack) and (not logic_fail), backtrack

    def check_requirement(self, plan, innercity_transports_ranking):
        """Check if plan satisfies all requirements (based on all_satisfy flag)

        Args:
            plan: Travel plan
            innercity_transports_ranking: List of allowed transport modes

        Returns:
            tuple: (ok: bool, backtrack: bool)
        """
        if self.all_satisfy:
            # Must satisfy all constraint groups (AND logic)
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints, innercity_transports_ranking)
                if backtrack:
                    return False, True  # Immediate backtrack
                if not ok:
                    return False, False  # Not satisfied but don't force backtrack
            return True, False
        else:
            # Satisfy any constraint group (OR logic)
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints, innercity_transports_ranking)
                if backtrack:
                    return False, True
                if ok:
                    return True, False  # One group satisfied
            return False, False  # No group satisfied

    def check_budgets(self, plan):
        """Quick budget validation check

        Args:
            plan: Travel plan

        Returns:
            bool: True if budget exceeded (should backtrack), False otherwise
        """
        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0

        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner"]:
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

    def constraints_validation(self, query, plan, poi_plan, time_before_search, llm_inference_time_count):
        """Comprehensive final validation with symbolic constraint checking

        Args:
            query: Query dictionary
            plan: Complete travel plan (itinerary format)
            poi_plan: POI plan dictionary
            time_before_search: Start time of search
            llm_inference_time_count: LLM inference time

        Returns:
            dict: Validation results with keys:
                - pass_all: bool - All constraints passed
                - pass_schema: bool - Schema validation passed
                - pass_commonsense: bool - Commonsense constraints passed
                - pass_logic: bool - Logical constraints passed
                - logical_pass_count: int - Number of logical constraints passed
                - plan: dict - The validated plan
        """
        import time
        import numpy as np
        from copy import deepcopy

        self.constraints_validation_count += 1

        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("validate the plan [for query {}]: ".format(query["uid"]))
        print(res_plan)

        # Track as least_plan_schema (base level - valid schema)
        pass_schema = True
        self.least_plan_schema = deepcopy(res_plan)

        # Check commonsense constraints
        bool_result = func_commonsense_constraints(query, res_plan, verbose=True)
        pass_commonsense = bool_result

        if bool_result:
            self.commonsense_pass_count += 1

        # Extract symbolic concepts
        try:
            extracted_vars = get_symbolic_concepts(query, res_plan, need_ood=False)
        except:
            extracted_vars = None

        print(extracted_vars)

        # Check logical constraints (hard_logic_py)
        pass_logic = True
        logical_pass_count = 0

        if "hard_logic_py" in query and query["hard_logic_py"] and len(query["hard_logic_py"]) > 0:
            logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=True)
            print(logical_result)

            for idx, item in enumerate(logical_result):
                if item:
                    print(query["hard_logic_py"][idx], "passed!")
                    logical_pass_count += 1
                else:
                    print(query["hard_logic_py"][idx], "failed...")
                    pass_logic = False
        else:
            # No hard_logic_py constraints or empty list
            print("[Warning] No hard_logic_py constraints found, skipping logical constraint validation")
            logical_result = []

        # Update least_plan_comm if commonsense passed and logical constraints improved
        if bool_result and logical_pass_count > self.least_plan_logical_pass:
            self.least_plan_comm = deepcopy(res_plan)
            self.least_plan_logical_pass = logical_pass_count

        # Update logical_pass_count if all logical constraints passed
        if pass_logic:
            self.logical_pass_count += 1

        # Overall result: commonsense AND all logical constraints
        pass_all = bool_result and pass_logic

        if pass_all:
            print("\n Pass! \n")
            self.all_constraints_pass += 1

            if self.least_plan_logic is None:
                self.least_plan_logic = res_plan
        else:
            print("\n Failed \n")

        # Add timing information if all constraints passed
        if pass_all:
            res_plan["search_time_sec"] = time.time() - time_before_search
            res_plan["llm_inference_time_sec"] = llm_inference_time_count

        return {
            "pass_all": pass_all,
            "pass_schema": pass_schema,
            "pass_commonsense": pass_commonsense,
            "pass_logic": pass_logic,
            "logical_pass_count": logical_pass_count,
            "plan": res_plan
        }
