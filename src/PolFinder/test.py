import numpy as np
from ase import Atoms
from ase.io import write

atoms = np.loadtxt('test.txt')

uc = atoms[:9, :]

symbols = ['Sr' for _ in range(9)]

cell = [20, 20, 20]
uc = Atoms(symbols=symbols, positions=uc, cell=cell, pbc=True)
write('Poscar_test', uc, format = 'vasp')
