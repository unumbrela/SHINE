"""POI Manager Module for UrbanTrip Agent

This module handles collection and caching of POI information from
the environment, including attractions, restaurants, hotels, and transport.
"""

import pandas as pd


class POIManager:
    """Manages POI data collection and caching"""

    def __init__(self, env, query):
        """
        Args:
            env: Environment interface for querying POI data
            query: Query dictionary with people_number
        """
        self.env = env
        self.query = query

    def collect_poi_info_all(self, city, poi_type):
        """Fetch all POIs of a specific type for a city

        Args:
            city: City name
            poi_type: One of "accommodation", "attraction", "restaurant"

        Returns:
            DataFrame: POI information
        """
        if poi_type == "accommodation":
            func_name = "accommodations_select"
        elif poi_type == "attraction":
            func_name = "attractions_select"
        elif poi_type == "restaurant":
            func_name = "restaurants_select"
        else:
            raise NotImplementedError

        poi_info = self.env(
            "{func}('{city}', 'name', lambda x: True)".format(func=func_name, city=city)
        )["data"]

        # Fetch all pages
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                poi_info = pd.concat([poi_info, info_i], axis=0, ignore_index=True)

        return poi_info

    def collect_innercity_transport(self, city, start, end, start_time, trans_type):
        """Fetch innercity transport options

        Args:
            city: City name
            start: Start location
            end: End location
            start_time: Departure time
            trans_type: Transport type ("metro", "taxi", "walk")

        Returns:
            list: Transport options, or "No solution" if error
        """
        if start == end:
            return []

        call_str = (
            'goto("{city}", "{start}", "{end}", "{start_time}", "{trans_type}")'.format(
                city=city,
                start=start,
                end=end,
                start_time=start_time,
                trans_type=trans_type,
            )
        )

        info = self.env(call_str)["data"]

        if not isinstance(info, list):
            return "No solution"

        # Adjust costs based on people_number
        if len(info) == 3:  # Metro with walk segments
            info[1]["price"] = info[1]["cost"]
            info[1]["tickets"] = self.query["people_number"]
            info[1]["cost"] = info[1]["price"] * info[1]["tickets"]

            info[0]["price"] = info[0]["cost"]
            info[2]["price"] = info[2]["cost"]
        elif info[0]["mode"] == "taxi":
            info[0]["price"] = info[0]["cost"]
            info[0]["cars"] = int((self.query["people_number"] - 1) / 4) + 1
            info[0]["cost"] = info[0]["price"] * info[0]["cars"]
        elif info[0]["mode"] == "walk":
            info[0]["price"] = info[0]["cost"]

        return info

    def collect_intercity_transport(self, source_city, target_city, trans_type):
        """Fetch intercity transport options

        Args:
            source_city: Source city
            target_city: Target city
            trans_type: Transport type ("train" or "airplane")

        Returns:
            DataFrame: Transport options
        """
        info_return = self.env(
            "intercity_transport_select('{source_city}', '{target_city}', '{trans_type}')".format(
                source_city=source_city, target_city=target_city, trans_type=trans_type
            )
        )

        if not info_return["success"]:
            return pd.DataFrame([])

        trans_info = info_return["data"]

        # Fetch all pages
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                trans_info = pd.concat([trans_info, info_i], axis=0, ignore_index=True)

        return trans_info
