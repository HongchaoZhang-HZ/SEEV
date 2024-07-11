from hmac import new
from torch import ne
from Verifier.VeriUtil import *
from Scripts.Status import NeuronStatus, NetworkStatus

''' Raycast Initialial Search

'''



# Raycast Initialial Search:
class Ray:
    def __init__(self, model, Origin, Direction:np.array):
        self.Origin = Origin
        self.Direction = Direction
        self.rayspeed = 1.0
        self.partical_dense = 10
        self.reflect_limit = 5
        self.dying_limit = 20
        self.NStatus = NetworkStatus(model)
        
    def orthogonal_vector(W_o):
        # Ensure W_o is a numpy array
        W_o = np.asarray(W_o)
        # Choose the first standard basis vector e1
        e1 = np.zeros_like(W_o)
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
        self.NStatus.get_netstatus_from_input(torch.tensor(intersect))
        S = self.NStatus.network_status_values
        W_B, r_B, W_o, r_o = LinearExp(model, S)
        # The idea is that the hyperplane can be represented by the gain computed from the model. The orthogonal vector can be computed by the computed gain.
        orth_vec = self.orthogonal_vector(W_o)
        
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
    
    def rayspread(self, model, case, angle=None, speed=None):
        if speed is None:
            speed = self.rayspeed
        if angle is None:
            angle = self.Direction
        while self.dying_limit > 0:
            if 
        self.dying_limit -= speed
        if self.if_spacebound(case.DOM):
            self.dying_limit -= 10
        # TODO: raystep(angle, speed)
        target = self.Origin + angle * speed
        list_flag_exceed = not(self.if_spacebound(target, case.DOM))
        if any(list_flag_exceed):
            self.dying_limit -= 10
            self.Direction = self.compute_boundary_angle(list_flag_exceed, case.DOM)
            # TODO: Compute new origin
            self.Origin = target
            return False, None
        
        # TODO: ray spread with angle and speed by sampling a few points in the given direction and steplength
        list_samples = [self.Origin + angle * speed/self.partical_dense * i for i in range(self.partical_dense)]
        sign_origin = np.sign(model.forward(torch.tensor(self.Origin)))
        for sample in list_samples:
            sign_sample = np.sign(model.forward(torch.tensor(sample)))
            if sign_origin != sign_sample:
                # TODO: call binary search
                intersection = None
                return True, intersection
    
    def if_rayintersect(self, model):
        # TODO: Implement the ray intersection using binary search
        
        pass
    
    def if_raytransparency(self):
        # TODO: if intersect and goes to desired side, return True
        pass
    
    def if_spacebound(self, spacebound):
        # TODO: if the intersect is on the space boundary, return True
        pass
    
    def raycast(self, model, case):
        list_intersect = []
        while self.dying_limit and self.reflect_limit:
            flag, intersect = self.rayspread(case, self.Direction, self.rayspeed)
            if flag:
                self.reflect_limit -= 1
                list_intersect.append(intersect)
                if self.if_raytransparency():
                    self.reflect_limit += 1
                    # self.Direction remains unchanged
                else:
                    # reflect the ray
                    self.Direction = self.compute_exit_angle_from_model(model, self.intersect)
        return list_intersect


class RaycastInitSearch:
    def __init__(self, model, case, Origin=None):
        self.model = model
        if Origin is None:
            Origin = self.rdm_origin()
        else:        
            self.Origin = Origin
        self.case = case
        self.ray = Ray(Origin, None)
        self.dim = case.DIM
        self.DOM = case.DOM
        
    def rdm_direction(self):
        # uniformly distrbuted random direction in the space
        theta_list = [np.random.uniform(0, 2*np.pi) for i in range(self.dim)-1]
        return np.array(theta_list)
    
    def rdm_origin(self):
        # uniformly distrbuted random origin in the space
        origin_list = [np.random.uniform(self.DOM[i][0], self.DOM[i][1]) for i in range(self.dim)]
        return np.array(origin_list)
    
    def reflect_direction(self, ray):
        # TODO: compute exit angle from the surface normal
        pass
    