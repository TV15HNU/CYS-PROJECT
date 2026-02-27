import torch

from .gating_network import UncertaintyAwareGating

class PhishXEnsemble:
    def __init__(self, transformer_model, cnn_model, gating_network=None, transformer_weight=0.7, cnn_weight=0.3):
        self.transformer = transformer_model
        self.cnn = cnn_model
        self.gating = gating_network # If None, use static weights
        self.t_weight = transformer_weight
        self.c_weight = cnn_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transformer.to(self.device)
        self.cnn.to(self.device)
        if self.gating:
            self.gating.to(self.device)
            
        self.transformer.eval()
        self.cnn.eval()

    def predict(self, transformer_inputs, cnn_inputs, numeric_features=None, num_passes=10):
        """
        Enables Monte Carlo Dropout Inference to estimate model uncertainty.
        """
        self.transformer.train() # KEEP DROPOUT ACTIVE
        self.cnn.train()         # KEEP DROPOUT ACTIVE
        
        t_probs, c_probs = [], []
        
        with torch.no_grad():
            for _ in range(num_passes):
                # Stochastic pass for Transformer
                t_id = transformer_inputs['input_ids'].to(self.device)
                t_mk = transformer_inputs['attention_mask'].to(self.device)
                t_probs.append(torch.sigmoid(self.transformer(t_id, t_mk)))
                
                # Stochastic pass for CNN
                c_in = cnn_inputs.to(self.device)
                c_probs.append(torch.sigmoid(self.cnn(c_in)))
        
        # Calculate Bayesian stats
        t_stack = torch.stack(t_probs)
        mean_t, var_t = t_stack.mean(dim=0), t_stack.var(dim=0)
        
        c_stack = torch.stack(c_probs)
        mean_c, var_c = c_stack.mean(dim=0), c_stack.var(dim=0)
        
        # Dynamic Fusion
        if self.gating and numeric_features is not None:
            feat_tensor = torch.tensor([list(numeric_features.values())], dtype=torch.float).to(self.device)
            p_final, alpha, uncertainty = self.gating(mean_t, var_t, mean_c, var_c, feat_tensor)
            
            return {
                "p_final": p_final.item(),
                "alpha": alpha.item(),
                "uncertainty": uncertainty.item(),
                "p_t": mean_t.item(),
                "p_c": mean_c.item(),
                "var_t": var_t.item(),
                "var_c": var_c.item()
            }
        else:
            # Fallback
            p_final = (mean_t.item() * self.t_weight) + (mean_c.item() * self.c_weight)
            return {
                "p_final": p_final, 
                "uncertainty": (var_t.item() + var_c.item())/2, 
                "alpha": self.t_weight,
                "p_t": mean_t.item(),
                "p_c": mean_c.item()
            }
