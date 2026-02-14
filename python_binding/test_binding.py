import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeeDNNLoader import BeeDNNLoader
import numpy as np

def test():
    print("Testing BeeDNN Python Binding...")
    try:
        # Note: This requires the shared library to be built and present.
        # Ensure 'BeeDNNLib.dll' (or .so) is in the current directory or path.
        
        dnn = BeeDNNLoader(2) # Input size 2
        print("Net created.")
        
        # Test adding layers with floats
        dnn.add_layer("Dense", [4.0], "GlorotUniform")
        dnn.add_layer("Tanh")
        dnn.add_layer("Dense", [1.0], "GlorotUniform")
        print("Layers added.")
        
        dnn.set_train_mode(True)
        print("Train mode set.")
        
        # Test predict
        net_in = np.array([[0,0],[0,1]], dtype=np.float32)
        net_out = np.zeros((2,1), dtype=np.float32)
        dnn.predict(net_in, net_out)
        print("Prediction run successfully.")
        print("Output shape:", net_out.shape)
        
        # Test Training API (New)
        trainer = dnn.create_trainer()
        print("Trainer created.")
        
        samples = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
        truth = np.array([[0],[1],[1],[0]], dtype=np.float32)
        
        dnn.fit(samples, truth, batch_size=4, epochs=1)
        print("Training loop run successfully (1 epoch).")
        
        loss = dnn.get_train_loss()
        print(f"Loss history size: {len(loss)}")
        
        print("Test passed!")
    except Exception as e:
        print("Test failed:", e)
        # Check if library load failed
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
