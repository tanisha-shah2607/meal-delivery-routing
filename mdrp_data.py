import os
import sys
import math
import pandas as pd


class Instance:
    """One Grubhub MDRP instance (orders, restaurants, couriers, parameters)."""

    def __init__(self, path):
        self.path = path
        rd = lambda name: pd.read_table(os.path.join(path, name))
        self.orders = rd("orders.txt").set_index("order")
        self.restaurants = rd("restaurants.txt").set_index("restaurant")
        self.couriers = rd("couriers.txt").set_index("courier")
        p = rd("instance_parameters.txt").iloc[0]
        self.speed = float(p["meters_per_minute"])          # meters per minute
        self.pickup_service = float(p["pickup service minutes"])
        self.dropoff_service = float(p["dropoff service minutes"])
        self.target_c2d = float(p["target click-to-door"])
        self.max_c2d = float(p["maximum click-to-door"])
        # id -> (x, y) for every order, restaurant and courier start point
        self.loc = {}
        for df in (self.orders, self.restaurants, self.couriers):
            for i, row in df.iterrows():
                self.loc[i] = (float(row["x"]), float(row["y"]))

    def travel(self, a, b):
        """Travel minutes between two node ids; same rule as the official evaluator."""
        (x1, y1), (x2, y2) = self.loc[a], self.loc[b]
        return math.ceil(math.hypot(x2 - x1, y2 - y1) / self.speed)


if __name__ == "__main__":
    inst = Instance(sys.argv[1])
    print("orders:", len(inst.orders), "| restaurants:", len(inst.restaurants),
          "| couriers:", len(inst.couriers))
    print("speed m/min:", inst.speed, "| service min (pick/drop):",
          inst.pickup_service, inst.dropoff_service)
    o = inst.orders.iloc[0]
    print("first order:", inst.orders.index[0], dict(o))
    print("travel restaurant r1 -> order o1:", inst.travel("r1", "o1"), "min")