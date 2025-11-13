import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def fuzzy_classify_fluoride(fluoride_value):
    """
    Classifies fluoride level using fuzzy logic:
    Input: fluoride_value (mg/L)
    Output: (Label, Risk_Score)
    """

    # Input variable
    fluoride = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'fluoride')

    # Output variable
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Membership functions
    fluoride['low'] = fuzz.trimf(fluoride.universe, [0, 0, 0.6])
    fluoride['normal'] = fuzz.trimf(fluoride.universe, [0.4, 0.8, 1.2])
    fluoride['high'] = fuzz.trimf(fluoride.universe, [1.0, 3.0, 3.0])

    risk['safe'] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['moderate'] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk['danger'] = fuzz.trimf(risk.universe, [60, 100, 100])

    # Fuzzy rules
    rule1 = ctrl.Rule(fluoride['low'], risk['safe'])
    rule2 = ctrl.Rule(fluoride['normal'], risk['moderate'])
    rule3 = ctrl.Rule(fluoride['high'], risk['danger'])

    # Control system
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
    risk_sim.input['fluoride'] = fluoride_value
    risk_sim.compute()

    risk_score = risk_sim.output['risk']

    # Convert to category
    if risk_score < 40:
        label = "Low (Safe)"
    elif risk_score < 65:
        label = "Normal (Moderate)"
    else:
        label = "High (Dangerous)"

    return label, risk_score
