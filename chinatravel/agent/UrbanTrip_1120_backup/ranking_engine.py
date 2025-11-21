"""Ranking Engine Module for UrbanTrip Agent

This module handles ranking and filtering of transport options, hotels,
and POI candidates based on user preferences and constraints.
"""

import numpy as np
from geopy.distance import geodesic


class RankingEngine:
    """Handles ranking of transport, hotels, and POIs"""

    def __init__(self, poi_search):
        """
        Args:
            poi_search: POI search service for coordinate lookups
        """
        self.poi_search = poi_search

    def ranking_intercity_transport_go(self, transport_info, intercity_budget, overall_budget):
        """Rank outbound intercity transport options

        Args:
            transport_info: DataFrame of transport options
            intercity_budget: Intercity budget constraint
            overall_budget: Overall budget constraint

        Returns:
            list: Indices in ranked order
        """
        time_list = transport_info["BeginTime"].tolist()
        price_list = transport_info["Cost"].tolist()

        # Sort by time first
        sorted_indices = np.argsort(time_list)

        # Filter by budget if constraint exists
        if intercity_budget is not None:
            budget_threshold = intercity_budget * 0.6
            filtered_indices = [idx for idx in sorted_indices if price_list[idx] <= budget_threshold]
            if filtered_indices:
                sorted_indices = filtered_indices

        # If overall budget exists, use combined time+price ranking
        if overall_budget is not None:
            # Calculate time ranking
            time_ranking = np.zeros(len(time_list), dtype=int)
            for i, idx in enumerate(np.argsort(time_list)):
                time_ranking[idx] = i + 1
            # Calculate price ranking
            price_ranking = np.zeros(len(price_list), dtype=int)
            for i, idx in enumerate(np.argsort(price_list)):
                price_ranking[idx] = i + 1
            # Combined ranking
            combined_ranking = time_ranking + price_ranking
            sorted_indices = list(np.argsort(combined_ranking))

        return sorted_indices

    def ranking_intercity_transport_back(self, transport_info, selected_go):
        """Rank return intercity transport options

        Args:
            transport_info: DataFrame of transport options
            selected_go: Selected outbound transport info

        Returns:
            list: Indices in ranked order (latest departure first)
        """
        time_list = transport_info["BeginTime"].tolist()
        sorted_lst = sorted(enumerate(time_list), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, value in sorted_lst]
        time_ranking = np.zeros_like(sorted_indices)
        for i, idx in enumerate(sorted_indices):
            time_ranking[idx] = i + 1

        ranking_idx = np.argsort(time_ranking)
        return ranking_idx

    def ranking_hotel(self, hotel_info, query, constraints_dict, memory):
        """Filter and rank hotels by constraints

        Args:
            hotel_info: DataFrame of hotel options
            query: Query dictionary
            constraints_dict: Parsed constraints
            memory: Memory containing POI info

        Returns:
            list: Indices of valid hotels sorted by price
        """
        candidate_idx = set(range(len(hotel_info)))

        # Must-live hotel constraint
        must_live_hotel = constraints_dict.get("must_live_hotel")
        if must_live_hotel is not None:
            must_idx = set(
                hotel_info[hotel_info["name"].isin(must_live_hotel)].index.tolist()
            )
            if must_idx:
                candidate_idx &= must_idx
            else:
                return []

        # Must-not-live hotel constraint
        must_not_live_hotel = constraints_dict.get("must_not_live_hotel")
        if must_not_live_hotel is not None:
            not_idx = set(
                hotel_info[hotel_info["name"].isin(must_not_live_hotel)].index.tolist()
            )
            candidate_idx -= not_idx

        # Must-have hotel features
        must_live_hotel_feature = constraints_dict.get("must_live_hotel_feature")
        if must_live_hotel_feature is not None:
            for feature in must_live_hotel_feature:
                feature_idx = set(
                    hotel_info[hotel_info["featurehoteltype"].str.contains(feature, na=False)].index.tolist()
                )
                candidate_idx &= feature_idx

        # Location distance limit
        must_live_hotel_location_limit = constraints_dict.get("must_live_hotel_location_limit")
        if must_live_hotel_location_limit is not None:
            for limit_dict in must_live_hotel_location_limit:
                for poi_name, max_distance in limit_dict.items():
                    valid_hotels = []
                    for idx in candidate_idx:
                        hotel_name = hotel_info.iloc[idx]["name"]
                        distance = self.calculate_distance(query, hotel_name, poi_name)
                        if distance is not None and distance <= max_distance:
                            valid_hotels.append(idx)
                    candidate_idx = set(valid_hotels)

        if not candidate_idx:
            return []

        # Sort by price
        candidate_list = list(candidate_idx)
        prices = [hotel_info.iloc[idx]["price"] for idx in candidate_list]
        sorted_pairs = sorted(zip(candidate_list, prices), key=lambda x: x[1])
        return [idx for idx, _ in sorted_pairs]

    def calculate_distance(self, query, start, end):
        """Calculate geodesic distance between two POIs

        Args:
            query: Query dictionary
            start: Start POI name
            end: End POI name

        Returns:
            float: Distance in kilometers, or None if lookup fails
        """
        city = query["target_city"]

        coordinate_A = self.poi_search.search(city, start)
        coordinate_B = self.poi_search.search(city, end)

        if not coordinate_A or not coordinate_B:
            return None

        distance = geodesic(coordinate_A, coordinate_B).kilometers
        return distance

    def get_transport_by_distance(self, distance, transport_rules_by_distance):
        """Select transport modes based on distance rules

        Args:
            distance: Distance in kilometers
            transport_rules_by_distance: List of distance-based transport rules

        Returns:
            list: Applicable transport modes
        """
        selected_modes = []

        if transport_rules_by_distance is None:
            return ["metro", "taxi", "walk"]

        for rule in transport_rules_by_distance:
            min_d = rule.get("min_distance", 0)
            max_d = rule.get("max_distance", float("inf"))
            transport = rule.get("transport_type")

            if min_d is None and max_d is not None:
                if distance <= max_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])
            if max_d is None and (min_d is not None and min_d != 0):
                if distance >= min_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])
            if min_d is not None and max_d is not None:
                if min_d <= distance <= max_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])

        if not selected_modes:
            selected_modes = ["metro", "taxi", "walk"]

        return selected_modes
