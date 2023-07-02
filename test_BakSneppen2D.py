import os
import shutil
from Bak_Sneppen import BakSneppen2D, BakSneppen2D_A, BakSneppen2D_ConservedPopulation

def test_BakSneppen2D_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5, print("Test Passed")

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
    assert len(model.min_fitness) == 5, print("Test Passed")

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')

def test_BakSneppen2D_ConservedPopulation_simulation():
    # The test folder is 'test_folder'
    if not os.path.exists('test_folder'):
        os.makedirs('test_folder')

    # Pass 'test_folder' to the model
    model = BakSneppen2D_ConservedPopulation(5, 'test_folder')
    model.simulate(5)

    # Check whether the simulation steps matches with expected.
    assert len(model.min_fitness) == 5, print("Test Passed")

    # Clean up by removing the test folder
    shutil.rmtree('test_folder')