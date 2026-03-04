class DecisionEngine:
    def __init__(self, high_risk_threshold=0.7, low_risk_threshold=0.3):
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold

    def get_action(self, risk_score, uncertainty):
        """
        Determine action based on risk score and uncertainty.
        """
        # If uncertainty is very high, flag for manual review
        if uncertainty > 0.15: # Arbitrary threshold for this example
            return "Manual Review Required"
            
        if risk_score > self.high_risk_threshold:
            return "Block"
        elif risk_score > self.low_risk_threshold:
            return "Warning"
        else:
            return "Safe"

    def process(self, risk_score, uncertainty):
        action = self.get_action(risk_score, uncertainty)
        prediction = "phishing" if risk_score > 0.5 else "legitimate"
        
        return {
            "prediction": prediction,
            "risk_score": round(risk_score, 4),
            "uncertainty": round(uncertainty, 4),
            "action": action
        }
