import os
import shutil
from .BakSneppen_Simple import BakSneppen2D
from .BS_simple_avalanche import BakSneppen2D_A
from .BakSneppen_PopulationDynamics import BakSneppen2D_PD
from .BakSneppen_PopulationDynamics_Conserved import BakSneppen2D_PDC

def test_BakSneppen2D_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')

def test_BakSneppen2D_A_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D_A(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')

def test_BakSneppen2D_PD_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D_PD(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')

def test_BakSneppen2D_PDC_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D_PDC(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')
    
