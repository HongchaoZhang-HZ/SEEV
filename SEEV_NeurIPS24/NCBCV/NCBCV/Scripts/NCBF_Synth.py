from Modules.utils import *
from Modules.NCBF import NCBF
from Cases import Case

class NCBFSynth(NCBF):
    def __init__(self, case: Case, NCBF: NCBF):
        super(NCBFSynth, self).__init__()
        self.case = case
        self.NCBF = NCBF
        self.DIM = self.case.DIM
        self.N = self.NCBF.N
        self.DOMAIN = self.NCBF.DOMAIN
        self.ctrldom = self.case.CTRLDOM
        self.discrete = self.case.discrete

    def h_x(self, x):
        return torch.sin(x[:, 0]) + torch.cos(x[:, 1])

    def generate_input(self, shape=[100,100]):
        state_space = self.DOMAIN
        noise = 1e-2 * torch.rand(shape)
        cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
        nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
        ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
        vxo, vyo = torch.meshgrid(nx, ny)
        vx = vxo + noise
        vy = vyo + noise
        data = torch.stack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])], dim=2)
        data = data.reshape(shape[0] * shape[1], 2)
        return data

    def generate_data(self, size: int = 100) -> torch.Tensor:
        state_space = self.DOMAIN
        shape = []
        for _ in range(self.DIM):
            shape.append(size)
        noise = 1e-2 * torch.rand(shape)
        cell_length = (state_space[0][1] - state_space[0][0]) / size
        raw_data = []
        for i in range(self.DIM):
            data_element = torch.linspace(state_space[i][0] + cell_length/2, state_space[i][1] - cell_length/2, shape[0])
            raw_data.append(data_element)
        raw_data_grid = torch.meshgrid(raw_data)
        noisy_data = []
        for i in range(self.DIM):
            noisy_data_item = raw_data_grid[i] + noise
            noisy_data_item = noisy_data_item.reshape([torch.prod(torch.Tensor(shape),dtype=int), 1])
            noisy_data.append(noisy_data_item)
        data = torch.cat([torch.Tensor(item) for item in noisy_data], dim=1)
        return data

    def correctness(self, ref_output, model_output, l_co=1):
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        violations = torch.sigmoid((-ref_output + norm_model_output).reshape([1, length]))
        loss = l_co * torch.sum(violations)
        return loss

    def warm_start(self, ref_output, model_output):
        loss = nn.MSELoss()
        loss_fcn = loss(model_output, ref_output)
        return loss_fcn

    def def_loss(self, *loss):
        total_loss = 0
        for l in loss:
            total_loss += l
        return total_loss

    def train(self, num_epoch):
        optimizer = optim.SGD(self.parameters(), lr=0.001)

        for epoch in range(num_epoch):
            shape = [100,100]
            rdm_input = self.generate_input(shape)

            running_loss = 0.0
            for i, data in enumerate(rdm_input, 0):

                optimizer.zero_grad()

                model_output = self.forward(rdm_input)
                ref_output = torch.tanh(self.h_x(rdm_input).reshape([shape[0]*shape[1], 1]))

                warm_start_loss = self.warm_start(ref_output, model_output)
                correctness_loss = self.correctness(ref_output, model_output, 1)
                loss = self.def_loss(warm_start_loss + correctness_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


# Create an instance of NCBF with default architecture and domain
ncbf = NCBF(architecture=None, domain=None)

# Generate input data with default shape
input_data = ncbf.generate_input()

# Generate data with default size
data = ncbf.generate_data()

# Calculate correctness loss with default reference output and model output
ref_output = torch.randn(data.shape[0], 1)
model_output = torch.randn(data.shape[0], 1)
correctness_loss = ncbf.correctness(ref_output, model_output)

# Calculate warm start loss with default reference output and model output
warm_start_loss = ncbf.warm_start(ref_output, model_output)

# Calculate total loss with default loss components
total_loss = ncbf.def_loss(warm_start_loss, correctness_loss)

# Train the model for a specified number of epochs
num_epochs = 10
ncbf.train(num_epochs)