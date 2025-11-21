"""Plan Builder Module for UrbanTrip Agent

This module handles construction of activity records and adding them
to the travel itinerary.
"""

import pandas as pd
from copy import deepcopy
from chinatravel.agent.UrbanTrip.urbantrip_utils import (
    time_compare_if_earlier_equal,
    add_time_delta,
)


class PlanBuilder:
    """Builds and adds activities to travel plans"""

    def __init__(self, query):
        """
        Args:
            query: Query dictionary with people_number
        """
        self.query = query

    def add_intercity_transport(self, activities, intercity_info, innercity_transports=[], tickets=1):
        """Add intercity transport activity to plan

        Args:
            activities: List of activities
            intercity_info: Transport information dictionary
            innercity_transports: Associated innercity transports
            tickets: Number of tickets

        Returns:
            list: Updated activities list
        """
        activity_i = {
            "start_time": intercity_info["BeginTime"],
            "end_time": intercity_info["EndTime"],
            "start": intercity_info["From"],
            "end": intercity_info["To"],
            "price": intercity_info["Cost"],
            "cost": intercity_info["Cost"] * tickets,
            "tickets": tickets,
            "transports": innercity_transports,
        }

        if "TrainID" in intercity_info and not pd.isna(intercity_info["TrainID"]):
            activity_i["TrainID"] = intercity_info["TrainID"]
            activity_i["type"] = "train"
        elif "FlightID" in intercity_info and not pd.isna(intercity_info["FlightID"]):
            activity_i["FlightID"] = intercity_info["FlightID"]
            activity_i["type"] = "airplane"

        activities.append(activity_i)
        return activities

    def add_poi(self, activities, position, poi_type, price, cost, start_time, end_time, innercity_transports):
        """Add generic POI activity to plan

        Args:
            activities: List of activities
            position: POI name
            poi_type: Activity type
            price: Price per person
            cost: Total cost
            start_time: Activity start time
            end_time: Activity end time
            innercity_transports: Associated transport

        Returns:
            list: Updated activities list
        """
        activity_i = {
            "position": position,
            "type": poi_type,
            "price": price,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time,
            "transports": innercity_transports,
        }

        activities.append(activity_i)
        return activities

    def add_accommodation(self, current_plan, hotel_sel, current_day, arrived_time, required_rooms, transports_sel):
        """Add hotel accommodation activity

        Args:
            current_plan: Current travel plan
            hotel_sel: Selected hotel info
            current_day: Day index
            arrived_time: Arrival time
            required_rooms: Number of rooms needed
            transports_sel: Associated transport

        Returns:
            dict: Updated plan
        """
        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=hotel_sel["name"],
            poi_type="accommodation",
            price=int(hotel_sel["price"]),
            cost=int(hotel_sel["price"]) * required_rooms,
            start_time=arrived_time,
            end_time="24:00",
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["room_type"] = hotel_sel["numbed"]
        current_plan[current_day]["activities"][-1]["rooms"] = required_rooms

        return current_plan

    def add_restaurant(self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel):
        """Add restaurant activity (breakfast/lunch/dinner)

        Args:
            current_plan: Current travel plan
            poi_type: Meal type ("lunch" or "dinner")
            poi_sel: Selected restaurant info
            current_day: Day index
            arrived_time: Arrival time
            transports_sel: Associated transport

        Returns:
            dict: Updated plan

        Raises:
            Exception: If restaurant timing constraints are violated
        """
        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]

        # Adjust start time to opening time if arriving too early
        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        # Lunch timing constraints
        if poi_type == "lunch":
            if time_compare_if_earlier_equal(act_start_time, "11:00"):
                act_start_time = "11:00"
            if time_compare_if_earlier_equal(endtime, "11:00"):
                raise Exception("ERROR: restaurant closed before 11:00")
            if time_compare_if_earlier_equal("13:00", act_start_time):
                raise Exception("ERROR: lunch begins after 13:00")

        # Dinner timing constraints
        if poi_type == "dinner":
            if time_compare_if_earlier_equal(act_start_time, "17:00"):
                act_start_time = "17:00"
            if time_compare_if_earlier_equal(endtime, "17:00"):
                if not time_compare_if_earlier_equal(endtime, opentime):
                    raise Exception("ERROR: restaurant closed before 17:00")
            if time_compare_if_earlier_equal("20:00", act_start_time):
                raise Exception("ERROR: dinner begins after 20:00")

        # Calculate end time (60 minutes)
        poi_time = 60
        act_end_time = add_time_delta(act_start_time, poi_time)
        aet = act_end_time

        # Truncate if exceeds closing time
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime
            if time_compare_if_earlier_equal(endtime, opentime):  # Open overnight
                act_end_time = aet

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        return tmp_plan

    def add_attraction(self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel):
        """Add attraction activity

        Args:
            current_plan: Current travel plan
            poi_type: Activity type (always "attraction")
            poi_sel: Selected attraction info
            current_day: Day index
            arrived_time: Arrival time
            transports_sel: Associated transport

        Returns:
            dict: Updated plan

        Raises:
            Exception: If attraction is closed at arrival time
        """
        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]

        # Check if closed
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")

        # Adjust start time to opening time if arriving too early
        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        # Calculate end time (90 minutes)
        poi_time = 90
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        tmp_plan[current_day]["activities"][-1]["tickets"] = self.query["people_number"]

        return tmp_plan

    def select_and_add_breakfast(self, plan, poi_plan, current_day, current_time, current_position, transports_sel):
        """Add breakfast activity at hotel location

        Args:
            plan: Current plan
            poi_plan: POI plan dictionary
            current_day: Day index
            current_time: Current time (should be "00:00")
            current_position: Current position (hotel)
            transports_sel: Transport to breakfast location

        Returns:
            dict: Updated plan
        """
        # Breakfast is at hotel, fixed time 08:00-08:30
        return plan
