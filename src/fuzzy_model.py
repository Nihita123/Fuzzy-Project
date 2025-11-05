"""
Fuzzy Inference System (FIS) for Health Risk Assessment based on Fluoride, Age, and Ingestion Rate.
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyHealthRisk:
    def __init__(self):
        # Input variables
        self.fluoride = ctrl.Antecedent(np.arange(0, 4.1, 0.1), "fluoride")
        self.age = ctrl.Antecedent(np.arange(0, 81, 1), "age")
        self.ingestion = ctrl.Antecedent(np.arange(0.5, 5.1, 0.1), "ingestion")

        # Output variables
        self.health_risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), "health_risk")

        self.define_memberships()
        self.define_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def define_memberships(self):
        self.fluoride["low"] = fuzz.trimf(self.fluoride.universe, [0, 0, 1.5])
        self.fluoride["medium"] = fuzz.trimf(self.fluoride.universe, [1.0, 2.0, 2.5])
        self.fluoride["high"] = fuzz.trimf(self.fluoride.universe, [2.0, 3.5, 4.0])

        self.age["young"] = fuzz.trimf(self.age.universe, [0, 15, 30])
        self.age["adult"] = fuzz.trimf(self.age.universe, [20, 40, 60])
        self.age["old"] = fuzz.trimf(self.age.universe, [50, 70, 80])

        self.ingestion["low"] = fuzz.trimf(self.ingestion.universe, [0.5, 1.5, 2.0])
        self.ingestion["medium"] = fuzz.trimf(self.ingestion.universe, [2.0, 3.0, 4.0])
        self.ingestion["high"] = fuzz.trimf(self.ingestion.universe, [3.5, 4.5, 5.0])

        self.health_risk["low"] = fuzz.trimf(self.health_risk.universe, [0, 0.2, 0.4])
        self.health_risk["medium"] = fuzz.trimf(self.health_risk.universe, [0.3, 0.5, 0.7])
        self.health_risk["high"] = fuzz.trimf(self.health_risk.universe, [0.6, 0.8, 1.0])

    def define_rules(self):
        r1 = ctrl.Rule(self.fluoride["high"] & self.ingestion["high"], self.health_risk["high"])
        r2 = ctrl.Rule(self.fluoride["medium"] & self.ingestion["medium"], self.health_risk["medium"])
        r3 = ctrl.Rule(self.fluoride["low"], self.health_risk["low"])
        r4 = ctrl.Rule(self.fluoride["medium"] & self.age["adult"], self.health_risk["medium"])
        r5 = ctrl.Rule(self.fluoride["high"] & self.age["old"], self.health_risk["high"])
        self.rules = [r1, r2, r3, r4, r5]

    def compute(self, fluoride_val, age_val, ingestion_val):
        self.sim.input["fluoride"] = fluoride_val
        self.sim.input["age"] = age_val
        self.sim.input["ingestion"] = ingestion_val
        self.sim.compute()
        return self.sim.output["health_risk"]

    def batch_evaluate(self, df, fluoride_col="F_raw"):
        df = df.copy()
        if "age" not in df.columns:
            df["age"] = [20] * len(df)
        if "ingestion" not in df.columns:
            df["ingestion"] = [2.5] * len(df)

        risks = []
        for _, row in df.iterrows():
            risk = self.compute(row[fluoride_col], row["age"], row["ingestion"])
            risks.append(risk)
        df["health_risk"] = risks
        return df
