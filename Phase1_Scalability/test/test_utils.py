import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *

class TestUtils(unittest.TestCase):
    def test_generate_samples(self):
        domain = [(-1, 1), (-1, 1)]
        num_samples = 100
        samples = generate_samples(domain, num_samples)
        
        # Add assertions to validate the generated samples
        assert len(samples) == num_samples
        assert all(domain[0][0] <= sample[0] <= domain[0][1] for sample in samples)
        assert all(domain[1][0] <= sample[1] <= domain[1][1] for sample in samples)
        
        # Add assertions to validate the visualization output
        # For example, you can check if the plot is displayed correctly


