import sys
import os
import numpy as np
import copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeeDNNLoader import BeeDNNLoader

def test_distributed():
    print("Testing Distributed Training API...")
    
    # Create Server
    server = BeeDNNLoader(2)
    server.add_layer("Dense", [2.0], "Zeros") # 2 neurons
    server.add_layer("Dense", [1.0], "Zeros") # 1 neurons
    server.create_trainer() # Initialize optimizers
    # Set optimizer to SGD with lr=1 for easy math
    # fit() usually sets params, we can set them manually via trainer
    # But BeeDNNLoader doesn't fully expose trainer config setters yet in __init__
    # We will use .fit() to init everything, then manual step.
    
    # Initialize weights
    print("Server initialized.")
    initial_weights = server.get_weights()
    print("Initial weights (should be 0):", initial_weights)
    
    # Create Worker and sync
    worker = BeeDNNLoader(2)
    worker.add_layer("Dense", [2.0], "Zeros")
    worker.add_layer("Dense", [1.0], "Zeros")
    worker.set_weights(initial_weights)
    print("Worker synced.")
    
    # Simulate Worker Training
    # Let's say worker weights changed by +1.0
    worker_weights = initial_weights + 1.0
    worker.set_weights(worker_weights)
    print("Worker trained (weights +1.0).")
    
    # Distributed Update on Server
    # 1. Accumulate Diff
    # Diff = Server - Worker = 0 - 1 = -1
    # Adding to Gradient: Gradient += -1
    server.accumulate_weight_diff(worker_weights)
    
    # 2. Distributed Step
    # Using SGD, lr=0.1 (default might be 0.01, let's assume unknown)
    # To verify, we'll see if weights move.
    # Optimizer step: W = W - lr * Grad
    # W = 0 - lr * (-1) = +lr
    server.distributed_step(num_workers=1.0)
    
    new_server_weights = server.get_weights()
    print("New Server Weights:", new_server_weights)
    
    if np.all(new_server_weights > initial_weights):
        print("PASS: Server weights updated in correct direction.")
    else:
        print("FAIL: Server weights did not update or went wrong direction.")
        
    print("Diff:", new_server_weights - initial_weights)
    
    # Test Mixing
    print("\nTesting Weight Mixing (Theta)...")
    # Reset server to 0
    server.set_weights(initial_weights)
    
    # Target weights = 10.0
    target_weights = initial_weights + 10.0
    theta = 0.5
    
    # Mix: 0.5 * 0 + 0.5 * 10 = 5.0
    server.mix_weights(target_weights, theta)
    mixed_weights = server.get_weights()
    print("Mixed Weights (Target 5.0):", mixed_weights)
    
    if np.allclose(mixed_weights, 5.0):
        print("PASS: Weights mixed correctly.")
    else:
        print("FAIL: Weight mixing failed.")


if __name__ == "__main__":
    try:
        test_distributed()
    except Exception as e:
        print("Test failed with exception:", e)
        import traceback
        traceback.print_exc()
