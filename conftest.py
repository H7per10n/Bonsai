import sys
import os

# Make the repo root importable so `import Bonsai` resolves correctly
# when running pytest from any directory.
sys.path.insert(0, os.path.dirname(__file__))
