import shutil
import os

source_dir = r"c:\Users\vishn\OneDrive\Desktop\cyber security\PhishX\models"
target_dir = r"c:\Users\vishn\OneDrive\Desktop\cyber security\PhishX_v2\core\original_models"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

files = ["transformer_model.py", "char_cnn_model.py", "ensemble.py", "gating_network.py"]
for f in files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))

# Copy utils/feature_extraction.py
shutil.copy(r"c:\Users\vishn\OneDrive\Desktop\cyber security\PhishX\utils\feature_extraction.py", 
            r"c:\Users\vishn\OneDrive\Desktop\cyber security\PhishX_v2\core\feature_extraction.py")
print("Files copied successfully")
