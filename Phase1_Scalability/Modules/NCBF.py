from util import *
from NNet import NNet

class NCBF(NNet):
    def __init__(self, architecture, domain, h_x):
        super(NCBF, self).__init__(architecture)
        self.DOMAIN = domain
        self.h_x = h_x
        

    def generate_input(self, shape=[100,100]):
        state_space = self.DOMAIN
        noise = 1e-2 * torch.rand(shape)
        cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
        nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
        ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
        vxo, vyo = torch.meshgrid(nx, ny)
        vx = vxo + noise
        vy = vyo + noise
        data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
        data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
        return data

    def generate_data(self, size: int = 100) -> torch.Tensor:
        '''
        Generate data for training or plotting
        :param size: the number of samples on each dimension
        :return: a mesh grid torch data
        '''
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
            # noisy_data_item = np.expand_dims(noisy_data_item, axis=self.DIM)
            noisy_data_item = noisy_data_item.reshape([torch.prod(torch.Tensor(shape),dtype=int), 1])
            noisy_data.append(noisy_data_item)
        data = torch.hstack([torch.Tensor(item) for item in noisy_data])

        return data

    def correctness(self, ref_output, model_output, l_co=1):
        '''
        Correctness loss function
        :param ref_output: h(x) output
        :param model_output: nn output
        :param l_co: lagrangian coefficient
        :return: number of correctness violation
        '''
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        violations = torch.sigmoid((-ref_output + norm_model_output).reshape([1, length]))
        loss = l_co * torch.sum(violations)
        return loss

    def warm_start(self, ref_output, model_output):
        '''
        MSE loss between ref and model
        :param ref_output: h(x) output
        :param model_output: nn output
        :return: MSE
        '''
        # loss = -torch.sum(torch.tanh(output))
        loss = nn.MSELoss()
        loss_fcn = loss(model_output, ref_output)
        return loss_fcn

    def def_loss(self, *loss):
        '''
        Define loss function by adding all loss
        :param loss: *loss allows multiple inputs
        :return: total loss
        '''
        total_loss = 0
        for l in loss:
            total_loss += l
        return total_loss

    def train(self, num_epoch):
        # Default training
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        for epoch in range(num_epoch):
            # Generate data
            shape = [100,100]
            vx, vy, rdm_input = self.generate_input(shape)

            running_loss = 0.0
            for i, data in enumerate(rdm_input, 0):

                optimizer.zero_grad()

                model_output = self.forward(rdm_input)
                ref_output = torch.tanh(self.h_x(rdm_input.transpose(0, 1)).reshape([shape[0]*shape[1], 1]))

                warm_start_loss = self.warm_start(ref_output, model_output)
                correctness_loss = self.correctness(ref_output, model_output, 1)
                loss = self.def_loss(warm_start_loss + correctness_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0



