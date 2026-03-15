from symphon.irrep_backend import IrRepsIrrep
import numpy as np

# Mock data
class MockAtoms:
    def __init__(self):
        self.cell = np.eye(3)
        self.scaled_positions = np.array([[0,0,0]])
        self.numbers = [1]
        self.masses = [1.0]

atoms = MockAtoms()
q = [0,0,0]
freqs = np.array([0,0,0])
eigvecs = np.eye(3)

irr = IrRepsIrrep(atoms, q, freqs, eigvecs)
try:
    print("Checking HAS_SPGREP...")
    # We call a method that uses HAS_SPGREP or just check it directly if we can
    import symphon.irrep_backend as ib
    print(f"HAS_SPGREP in ib: {getattr(ib, 'HAS_SPGREP', 'NOT FOUND')}")
except Exception as e:
    print(f"Caught error: {e}")
