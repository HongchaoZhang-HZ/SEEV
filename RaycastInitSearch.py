from hmac import new
from numpy import sign
from torch import ne
from Verifier.VeriUtil import *
from Scripts.Status import NeuronStatus, NetworkStatus

''' Raycast Initialial Search

'''



# Raycast Initialial Search:
class Ray:
    def __init__(self, model, case, 
                 Origin:np.array, Direction:np.array, 
                 safe_desire=True):
        self.Origin = Origin
        self.Direction = Direction
        self.rayspeed = 1.0
        self.partical_dense = 10
        self.reflect_penalty = 10
        self.reflect_limit = 5
        self.dying_limit = 30
        
        self.safe_desire = safe_desire
        self.NStatus = NetworkStatus(model)
        self.speed_step = self.rayspeed / self.partical_dense
        self.is_spacebound = False
        self.is_rayintersect = False
        self.is_raytransparency = False
        self.model = model
        self.num_layers = len(self.model.layers)/2
        self.case = case
        self.list_intersection = []
        self.list_activation_intersections = []
        
    def orthogonal_vector(self, W_o):
        # Ensure W_o is a numpy array
        W_o = np.asarray(W_o).squeeze()
        # Choose the first standard basis vector e1
        e1 = np.zeros_like(W_o).squeeze()
        e1[0] = 1
        # Project e1 onto W_o
        projection = np.dot(e1, W_o) / np.dot(W_o, W_o) * W_o
        # Subtract the projection from e1 to get a vector orthogonal to W_o
        orthogonal_vec = e1 - projection
        # Normalize the orthogonal vector to have unit length
        orthogonal_vec /= np.linalg.norm(orthogonal_vec)
        return orthogonal_vec    
    
    
    def compute_boundary_angle(self, list_flag_exceed):
        # bound_hit = list(self.intersect == spacebound)
        new_direction = self.Direction
        for i in range(len(list_flag_exceed)):
            if list_flag_exceed[i]:
                new_direction[i] = -new_direction[i]
        return new_direction
    
    def compute_exit_angle_from_model(self, model, intersect):
        # compute new_direction from the model
        # Given input direction and the intersection point with the model, we first compute the orthogonal vector to the model. Then we compute the exit angle. 
        self.NStatus.get_netstatus_from_input(torch.tensor(intersect).float())
        S = self.NStatus.network_status_values
        W_B, r_B, W_o, r_o = LinearExp(model, S)
        # The idea is that the hyperplane can be represented by the gain computed from the model. The orthogonal vector can be computed by the computed gain.
        orth_vec = self.orthogonal_vector(W_o[self.num_layers-1])
        
        new_direction = self.update_direction(self.Direction, orth_vec)
        return new_direction
    
    def update_direction(self, Direction, orth_vec):
        # Normalize the orthogonal vector v
        v_normalized = orth_vec / np.linalg.norm(orth_vec)
        
        # Normalize the vector a to make it a unit vector
        a_normalized = Direction / np.linalg.norm(Direction)
        # Calculate the dot product of a and v
        dot_product_a_v = np.dot(a_normalized, v_normalized)
        
        # Calculate the orthogonal component of a_normalized with respect to v_normalized
        orthogonal_component = a_normalized - dot_product_a_v * v_normalized
        orthogonal_component /= np.linalg.norm(orthogonal_component)
        
        # Construct vector b by combining the parallel and orthogonal components
        angle_cosine = np.dot(a_normalized, v_normalized)
        new_direction = angle_cosine * v_normalized + np.sqrt(1 - angle_cosine**2) * orthogonal_component
        return new_direction
    
    def rayspread(self):
        sign_origin = np.sign(self.model.forward(torch.tensor(self.Origin).float()).detach().numpy())
        while self.dying_limit > 0 and self.reflect_limit >= 0:
            for step in range(self.partical_dense):
                # raystep in angle with desired speed
                target = self.Origin + self.Direction * self.speed_step * step
                self.dying_limit -= self.speed_step  
                
                # check if the ray bounce on the space boundary
                if self.if_spacebound(target, self.case.DOMAIN):
                    self.dying_limit -= 10
                    break
                
                # check if the ray intersect with the model
                if self.if_rayintersect(target, sign_origin):
                    self.reflect_limit -= 1
                    break
    
    def binarysearch(self, p_safe, p_unsafe, iter_lim=100):
        flag = False
        S = None
        x = None
        for iter in range(iter_lim):
            mid_point = (p_safe + p_unsafe) / 2.0
            self.NStatus.get_netstatus_from_input(mid_point)
            # print(p_safe, mid_point, p_unsafe)
            S = self.NStatus.network_status_values
            res = solver_lp(self.model, S, SSpace=self.case.SSpace)
            x = res.get_x_val()
            if res.is_success():
                flag = True
                return flag, S, x
            # elif self.model.forward(mid_point) > 0:
            elif torch.sign(self.model.forward(mid_point)) * torch.sign(self.model.forward(p_safe)) < 0:
                p_unsafe = mid_point
            else:
                p_safe = mid_point
        return flag, S, x
    
    def if_rayintersect(self, target, sign_origin):
        # first we check if the sign flips
        sign_target = np.sign(self.model.forward(torch.tensor(target).float()).detach().numpy())
        if sign_origin != sign_target:
            # second we check if the ray intersect with the model
            # Implement the ray intersection using binary search
            previous_point = torch.tensor(target - self.speed_step * self.Direction)
            if (not self.case.reverse_flag and sign_origin == 1) or (self.case.reverse_flag and sign_origin == -1):
                safe_point = previous_point.float()
                unsafe_point = torch.tensor(target).float()
            else:
                safe_point = torch.tensor(target).float()
                unsafe_point = previous_point.float()
            succ_flag, S, x = self.binarysearch(safe_point, unsafe_point)
            if succ_flag:
                self.list_intersection.append(x)
                self.list_activation_intersections.append(S)
                self.Origin = x
                self.Direction = self.compute_exit_angle_from_model(self.model, x)
            if self.if_raytransparency(sign_origin):
                self.Origin = target
                self.reflect_limit += 1
            
            print("The ray has hit the space boundary")
            print("The new direction is: ", self.Direction)
            print("The new origin is: ", self.Origin)
            return True
        else:
            return False
            
            
    
    def if_raytransparency(self, sign_origin):
        if self.safe_desire:
            if sign_origin == 1:
                return False
            else:
                return True
        else:
            if sign_origin == -1:
                return False
            else:
                return True
    
    def if_spacebound(self, target, spacebound=None):
        # TODO: debug this one
        # if the intersect is on the space boundary, return True
        # compare if the vector target exceeds the space boundary
        if spacebound is None:
            spacebound = self.DOMAIN
        # space bondary is represented by a list of list. Each inner list is a pair of lower and upper bound of the space. 
        list_flag_exceed = [target[i] < spacebound[i][0] or target[i] > spacebound[i][1] for i in range(len(target))]
        
        if any(list_flag_exceed):
            self.is_spacebound = True
            self.dying_limit -= self.reflect_penalty
            self.Direction = self.compute_boundary_angle(list_flag_exceed)
            # Calculate new origin
            new_origin = np.copy(target)
            for I, exceed in enumerate(list_flag_exceed):
                if exceed:
                    if target[I] < spacebound[I][0]:
                        boundary = spacebound[I][0]
                    else:
                        boundary = spacebound[I][1]
                    
                    # Calculate the intersection point
                    t = (boundary - target[I]) / (self.Direction[I] * self.speed_step)
                    intersection_point = target + t * (-self.Direction * self.speed_step)
                    
                    # Update new_origin with the intersection point for the exceeded dimension
                    new_origin[I] = intersection_point[I]
            
            self.Origin = new_origin
            # visualize
            print("The ray has hit the space boundary")
            print("The new direction is: ", self.Direction)
            print("The new origin is: ", self.Origin)
            
            return True
        else:
            return False

class RaycastSearch:
    def __init__(self, model, case, 
                 same_origin=None, same_direction=None):
        self.model = model
        self.case = case
        self.num_origin = 1
        self.num_rays = 4
        self.list_rays = []
        self.list_intersection = []
        self.list_activation_intersections = []
        if same_origin is None:
            self.same_origin = True
        else:
            self.same_origin = same_origin
        if same_direction is None:
            self.same_direction = False
        else:
            self.same_direction = same_direction
        self.mode_selection()
        
    def mode_selection(self):
        if self.same_origin and self.same_direction:
            self.mode = 0 # same origin and same direction
        elif self.same_origin and not self.same_direction:
            self.mode = 1 # same origin and rdm direction
        elif not self.same_origin and self.same_direction:
            self.mode = 2 # rdm origin and same direction
        else:
            self.mode = 3 # rdm origin and rdm direction
    
    def rdm_direction(self):
        # uniformly distrbuted random direction in the space
        theta_list = [np.random.uniform(0, 2*np.pi) for i in range(self.case.DIM)]
        return np.array(theta_list)
    
    def rdm_origin(self):
        # uniformly distrbuted random origin in the space
        origin_list = [np.random.uniform(self.DOM[i][0], self.DOM[i][1]) for i in range(self.case.DIM)]
        return np.array(origin_list)
    
    def init_ray(self, Origin=None, Direction=None):
        if Origin is None:
            Origin = self.rdm_origin()
        if Direction is None:
            Direction = self.rdm_direction()
        return Ray(self.model, self.case, Origin, Direction)
    
    def raycast(self, list_Origin:list=None, list_Direction:list=None):
        if self.mode == 0 or self.mode == 1:
            if list_Origin is None:
                raise ValueError("Origin must be specified")
            origin = list_Origin
        else:
            origin = [self.rdm_origin() for i in range(self.num_origin)]
        if self.mode == 0 or self.mode == 2:
            if list_Direction is None:
                raise ValueError("Direction must be specified")
            direction = list_Direction
            self.num_rays = len(list_Direction)
        else:
            direction = [self.rdm_direction() for i in range(self.num_rays)]
        for i in range(self.num_origin):
            print(i)
            for j in range(self.num_rays):
                ray = Ray(self.model, self.case, origin[i], direction[j])
                ray.rayspread()
                self.list_rays.append(ray)
                self.list_intersection += ray.list_intersection
                self.list_activation_intersections += ray.list_activation_intersections
        print(f'Raycast identified {len(self.list_intersection)} intersections with {len(self.list_rays)} rays')

if __name__ == "__main__":
    from Cases.LinearSatellite import LinearSat
    from Modules.NNet import NeuralNetwork as NNet
    
    case = LinearSat()
    n = 16
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('relu', n), ('linear', 1)]
    model = NNet(architecture)
    # trained_state_dict = torch.load(f"Phase2_Verification/models/linear_satellite_hidden_32_epoch_50_reg_0.05.pt")
    trained_state_dict = torch.load(f"models/linear_satellite_layer_3_hidden_16_epoch_50_reg_0.pt")
    model.load_state_dict_from_sequential(trained_state_dict)
    
    Origin = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    Direction = [np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
    # ray = Ray(model, case, Origin[0], Direction[0])
    # ray.rayspread()
    # print(len(ray.list_intersection))
    
    RCSearch = RaycastSearch(model, case, same_origin=True, same_direction=False)
    RCSearch.num_rays = 10
    RCSearch.raycast(Origin, Direction)
    