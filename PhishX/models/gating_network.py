import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareGating(nn.Module):
    """
    SOTA Research Component:
    Adversarially-Aware Uncertainty-Driven Gating (AAUDG).
    
    This network uses Bayesian Uncertainty (via MC Dropout) to decide
    which engine to trust during an adversarial attack.
    """
    def __init__(self, feature_dim=8):
        super(UncertaintyAwareGating, self).__init__()
        # Input: [Mean_T, Var_T, Mean_C, Var_C, Numeric Features]
        # Dim: 1 + 1 + 1 + 1 + feature_dim = 12
        input_dim = 4 + feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, mean_t, var_t, mean_c, var_c, features):
        """
        Calculates dynamic weight Alpha based on Bayesian confidence.
        """
        combined = torch.cat([mean_t, var_t, mean_c, var_c, features], dim=1)
        alpha = self.mlp(combined)
        
        # Risk-Aware Fusion
        p_final = alpha * mean_t + (1 - alpha) * mean_c
        
        # Uncertainty propagation: Combined variance
        # (Simplified empirical estimate for the combined system)
        systemic_uncertainty = alpha * var_t + (1 - alpha) * var_c
        
        return p_final, alpha, systemic_uncertainty
