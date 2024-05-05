import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
from Scripts.BaBBasic import QBasic, BaBBasic
import heapq

class BranchAndBound:
    def __init__(self, initial_network_status, set_U):
        self.best_solution = None
        self.best_cost = float('inf')
        self.initial_network_status = initial_network_status
        self.set_U = set_U

    def solve(self):
        # Initialize the priority queue with the initial network status
        Q = [(self.initial_network_status.calculate_cost(), self.initial_network_status)]

        while Q:
            # Dequeue the network status with the lowest cost
            cost, network_status = heapq.heappop(Q)

            # If this is a valid solution and its cost is less than the best cost found so far
            if network_status.is_valid() and cost < self.best_cost:
                # Update the best solution and the best cost
                self.best_solution = network_status
                self.best_cost = cost

            # For each neuron in set_U
            for neuron in self.set_U:
                # Generate new network statuses by setting the neuron to active, inactive, and zero
                for state in [NeuronStatus.ACTIVE, NeuronStatus.INACTIVE, NeuronStatus.ZERO]:
                    new_status = network_status.copy()
                    new_status.set_neuron_state(neuron, state)
                    new_cost = new_status.calculate_cost()

                    # If the cost of the new status is less than the best cost found so far
                    if new_cost < self.best_cost:
                        # Add the new status to the priority queue
                        heapq.heappush(Q, (new_cost, new_status))