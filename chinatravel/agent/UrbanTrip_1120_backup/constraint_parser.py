"""Constraint Parser Module for UrbanTrip Agent

This module handles parsing of DSL (Domain-Specific Language) constraints
into structured Python dictionaries.
"""

import re


class ConstraintParser:
    """Parses DSL constraints from hard_logic_py into structured constraints"""

    def __init__(self, memory):
        """
        Args:
            memory: Dictionary containing attractions and restaurants info
        """
        self.memory = memory

    def extract_user_constraints_by_DSL(self, query):
        """Extract and parse user constraints from DSL expressions

        Args:
            query: Query dictionary containing hard_logic_py field

        Returns:
            tuple: (constraints_dict, requirement_list)
        """
        # Convert DSL to string format
        dsl = query.get("hard_logic_py", "")
        if isinstance(dsl, (list, tuple)):
            dsl = "\n".join(str(item) for item in dsl)
        elif not isinstance(dsl, str):
            dsl = str(dsl) if dsl is not None else ""

        def extract_list(s):
            """Extract quoted strings from DSL expression"""
            return re.findall(r"[\"']([^\"']+)[\"']", s) or [s.strip()]

        def parse_single_dsl(dsl_str, query):
            """Parse a single DSL expression into constraints dictionary"""
            res = {}

            # all_satisfy logic (AND vs OR)
            res["all_satisfy"] = False if re.search(
                r"result_list\s*=\s*\[\]\s*.*\s*result\s*=\s*False\s*.*\s*result\s*=\s*result\s*or\s*r",
                dsl_str,
                flags=re.S
            ) else True

            # Attraction constraints
            patterns = {
                "must_see_attraction": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)attraction_name_set',
                "must_see_attraction_type": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)attraction_type_set',
                "must_not_see_attraction": r'result\s*=\s*not\s*\(\{([^}]*)\}(?:&|<=)attraction_name_set',
                "must_not_see_attraction_type": r'result\s*=\s*not\s*\(\{([^}]*)\}(?:&|<=)attraction_type_set'
            }
            for key, pat in patterns.items():
                m = re.search(pat, dsl_str)
                res[key] = extract_list(m.group(1)) if m else None

            # only_free_attractions
            m = re.search(
                r"attraction_cost\s*\+=\s*activity_cost\(activity\).*?attraction_cost\s*<=\s*0",
                dsl_str, flags=re.S
            )
            if m:
                res["only_free_attractions"] = True
            else:
                res.pop("only_free_attractions", None)

            # Activity time constraints
            matches = re.findall(
                r"if\s+activity_position\(activity\)\s*==\s*'([^']+)'.*?activity_time\(activity\)\s*>=\s*([0-9]+)",
                dsl_str, flags=re.S
            )
            res["activities_stay_time_dict"] = {name: int(time) for name, time in matches} if matches else None

            matches = re.findall(
                r"if\s+activity_position\(activity\)\s*==\s*'([^']+)'.*?activity_start_time\(activity\)\s*<=\s*'([^']+)'",
                dsl_str, flags=re.S
            )
            res["activities_arrive_time_dict"] = {name: ["early", t] for name, t in matches} if matches else None

            matches = re.findall(
                r"if\s+activity_position\(activity\)\s*==\s*'([^']+)'.*?activity_end_time\(activity\)\s*>=\s*'([^']+)'",
                dsl_str, flags=re.S
            )
            res["activities_leave_time_dict"] = {name: ["late", t] for name, t in matches} if matches else None

            # Restaurant constraints
            patterns = {
                "must_visit_restaurant": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)restaurant_name_set',
                "must_visit_restaurant_type": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)restaurant_type_set',
                "must_not_visit_restaurant": r'result\s*=\s*not\s*\(\{([^}]*)\}(?:&|<=)restaurant_name_set',
                "must_not_visit_restaurant_type": r'result\s*=\s*not\s*\(\{([^}]*)\}(?:&|<=)restaurant_type_set'
            }
            for key, pat in patterns.items():
                m = re.search(pat, dsl_str)
                res[key] = extract_list(m.group(1)) if m else None

            # Hotel constraints
            patterns = {
                "must_live_hotel": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)accommodation_name_set',
                "must_not_live_hotel": r'result\s*=\s*not\s*\(\{([^}]*)\}(?:&|<=)accommodation_name_set',
                "must_live_hotel_feature": r'result\s*=\s*\(\{([^}]*)\}(?:&|<=)accommodation_type_set'
            }
            for key, pat in patterns.items():
                m = re.search(pat, dsl_str)
                res[key] = extract_list(m.group(1)) if m else None

            m = re.search(
                r"poi_distance\(target_city\(plan\)\s*,\s*'([^']+)'\s*,\s*accommodation_position\)\s*<=\s*([0-9\.]+)",
                dsl_str)
            res["must_live_hotel_location_limit"] = [{m.group(1): float(m.group(2))}] if m else None

            m = re.search(r"room_type\(activity\)\s*!=\s*([0-9]+)", dsl_str)
            res["bed_number"] = int(m.group(1)) if m else None

            m = re.search(r"room_count\(activity\)\s*!=\s*([0-9]+)", dsl_str)
            res["room_number"] = int(m.group(1)) if m else None

            # Innercity transport constraints
            res["must_innercity_transport"] = None
            if res["must_innercity_transport"] is None:
                m = re.search(r'result=\(\s*\{(.*?)\}\s*&\s*inner_city_transportation_set', dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None
            if res["must_innercity_transport"] is None:
                m = re.search(r"result=\(innercity_transport_set<=\s*\{([^}]*)\}\s*\)", dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None
            if res["must_innercity_transport"] is None:
                m = re.search(r"result=\(\s*\{([^}]*)\}\s*<=innercity_transport_set\)", dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None

            m = re.search(r'result\s*=\s*not\s*\(\s*\{(.*?)\}\s*&\s*inner_city_transportation_set', dsl_str)
            res["must_not_innercity_transport"] = extract_list(m.group(1)) if m else None

            # Transport rules by distance
            m = re.search(
                r"innercity_transport_type\(activity_transports\(activity\)\)\s*!=\s*'([^']+)'.*?innercity_transport_distance\(activity_transports\(activity\)\)\s*>\s*([0-9\.]+)",
                dsl_str, flags=re.S)
            res["transport_rules_by_distance"] = [{"min_distance": float(m.group(2)), "max_distance": None,
                                                   "transport_type": [m.group(1)]}] if m else None

            # Intercity transport constraints
            m = re.search(r'allactivities\(plan\)\[0\]\[\'type\'\]\s*==\s*["\']([^"\']+)["\']', dsl_str)
            res["must_depart_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[0\]\[\'type\'\]\s*!=\s*["\']([^"\']+)["\']', dsl_str)
            res["must_not_depart_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[-1\]\[\'type\'\]\s*==\s*["\']([^"\']+)["\']', dsl_str)
            res["must_return_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[-1\]\[\'type\'\]\s*!=\s*["\']([^"\']+)["\']', dsl_str)
            res["must_not_return_transport"] = extract_list(m.group(1)) if m else None

            if res["must_depart_transport"] is None and res["must_not_depart_transport"] is None and res[
                "must_return_transport"] is None and res["must_not_return_transport"] is None:
                m = re.search(r'result=\(\{([^}]*)\}==intercity_transport_set\)', dsl_str)
                res["intercity transport"] = extract_list(m.group(1)) if m else None
                if res.get("intercity transport"):
                    res["must_depart_transport"] = res["intercity transport"]
                    res["must_return_transport"] = res["intercity transport"]

            # Budget constraints
            budget_patterns = {
                "attraction_budget": r"attraction_cost\s*<=\s*([0-9]+)",
                "restaurant_budget": r"restaurant_cost\s*<=\s*([0-9]+)",
                "hotel_budget": r"accommodation_cost\s*<=\s*([0-9]+)",
                "innercity_budget": r"inner_city_transportation_cost\s*<=\s*([0-9]+)",
                "intercity_budget": r"inter_city_transportation_cost\s*<=\s*([0-9]+)",
                "overall_budget": r"total_cost\s*<=\s*([0-9]+)"
            }
            for key, pat in budget_patterns.items():
                m = re.search(pat, dsl_str)
                res[key] = float(m.group(1)) if m else None

            m = re.search(r"result=\(hotel_cost/people_count\(plan\)/\(day_count\(plan\)-1\)<=([0-9\.]+)\)", dsl_str)
            hb = float(m.group(1)) if m else None
            if hb != None:
                res["hotel_budget"] = query["days"] * query["people_number"] * hb

            m = re.search(r"result=\(food_cost/food_count/people_count\(plan\)<=([0-9\.]+)\)", dsl_str)
            res["restaurant_budget_per_meal"] = float(m.group(1)) if m else None

            attr_info = self.memory["attractions"]
            res_info = self.memory["restaurants"]

            # Ensure must_see_attraction / must_visit_restaurant exist as lists
            res.setdefault("must_see_attraction", [])
            res.setdefault("must_visit_restaurant", [])

            # Process time constraint dictionaries
            dict_keys = [
                "activities_stay_time_dict",
                "activities_arrive_time_dict",
                "activities_leave_time_dict"
            ]

            # Initialize as lists if not already
            if not isinstance(res.get("must_see_attraction"), list):
                res["must_see_attraction"] = []
            if not isinstance(res.get("must_visit_restaurant"), list):
                res["must_visit_restaurant"] = []

            for key in dict_keys:
                if res.get(key) is not None:
                    for name in res[key].keys():
                        if name in attr_info["name"].values:
                            if name not in res["must_see_attraction"]:
                                res["must_see_attraction"].append(name)
                        elif name in res_info["name"].values:
                            if name not in res["must_visit_restaurant"]:
                                res["must_visit_restaurant"].append(name)

            res_filtered = {k: v for k, v in res.items() if v is not None}

            return res_filtered

        # Parse complete DSL
        results_main = parse_single_dsl(dsl, query)

        # If all_satisfy=False, split DSL into multiple constraint groups
        results_list = []
        if not results_main["all_satisfy"]:
            sub_dsls = [s.strip() for s in dsl.split("result_list.append(result)") if s.strip()]
            for sub_dsl in sub_dsls:
                results_list.append(parse_single_dsl(sub_dsl, query))
        else:
            results_list = [results_main]

        return results_main, results_list
