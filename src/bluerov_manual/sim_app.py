import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from marinegym import init_simulation_app

def init_app(cfg):
    # MarineGym already wraps SimulationApp creation correctly
    return init_simulation_app(cfg)
