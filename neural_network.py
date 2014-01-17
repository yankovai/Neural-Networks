import numpy as np
from scipy.optimize import minimize

class NeuralNetwork(object):
    
    def __init__(self, nfeatures, noutputs, nhidden_layers):
        """
        """
        
        self.nfeatures = nfeatures
        self.noutputs = noutputs
        self.nhidden_layers = nhidden_layers   
        
        units_per_layer = np.ceil(nfeatures*1.5)
        
#       Initialize thetas
        np.random.seed(9999)
        thetas = [np.random.rand(units_per_layer, nfeatures + 1)]
        thetas.append(np.random.rand(units_per_layer, units_per_layer + 1)*(nhidden_layers - 1))
        thetas.append(np.random.rand(noutputs, units_per_layer + 1))
        self.thetas = thetas
        self.nthetas = len(thetas)
        self.thetas_shapes = map(np.shape, thetas)
                
    def _sigmoid(self, z):
        """
        Evaluates the sigmoid function at z.
        """
        
        return np.power(1. + np.exp(-z), -1)      
        
    def _forward_prop(self, x, thetas = None):
        """
        Returns all activation unit values.
        """
        
        if thetas == None:
            thetas = self.thetas
            
        a = [np.append(1., x)]
        for theta, i in zip(thetas, range(self.nthetas)):
            z = np.dot(theta, a[i])
            sigmoid_z = self._sigmoid(z)
            a.append(np.append(1., sigmoid_z))
        
        # Remove bias unit from last layer
        a[-1] = a[-1][1::]
        
        return a
            
    def _back_prop(self, a, y):
        """
        Perform back propagation to obtain the error in each layer's units. 
        """
        
        delta = a[-1] - y 
        deltas = [delta]
        for theta, ai in zip(reversed(self.thetas), a[-2:0:-1]):
            delta = np.dot(theta.transpose(), delta)
            delta *= ai*(1. - ai)
            delta = delta[1::]
            deltas.append(delta)
        
        return deltas
    
    def _update_Deltas(self, a, deltas, Deltas):
        """
        Add the derivative contribution of each training example. 
        """
        
        updated_Deltas = []
        a = a[-2::-1]    
        for Delta, delta, ai in zip(reversed(Deltas), deltas, a):
            updated_Deltas.insert(0, Delta + np.outer(delta, ai))
        
        return updated_Deltas
        
    def cost(self, h_theta, y):
        """
        Calculates the cost function for one training example. 
        """
        
        return -y*np.log(h_theta) - (1. - y)*np.log(1. - h_theta)
    
    def total_cost(self, X, Y, thetas = None):
        """
        Calculates the total cost function using every example in the training
        set.
        """
        
        if thetas == None:
            thetas = self.thetas
            
        J = 0.0
        m = X.shape[0]
        for x, true_indx in zip(X, Y):
            y = np.zeros(self.noutputs)
            y[true_indx] = 1.
            h_theta = self._forward_prop(x, thetas)[-1]
            J += self.cost(h_theta, y)
                
        return np.sum(J)/m
        
    def get_gradients(self, X, Y):
        """
        Gets the derivative of the cost function with respect to each regression
        parameter using back propagation.
        """
        
        # Initialize Deltas
        Deltas = []
        for theta in self.thetas:
            Deltas.append(np.zeros_like(theta))        
        
        m_inv = 1./X.shape[0]
        for x, true_indx in zip(X, Y):
            y = np.zeros(self.noutputs)
            y[true_indx] = 1.
            a = self._forward_prop(x)
            deltas = self._back_prop(a, y)
            Deltas = self._update_Deltas(a, deltas, Deltas)
        
        Deltas = map(lambda x: x*m_inv, Deltas)
        
        return Deltas
        
    def _objective_function(self, thetas, X, Y):
        """
        Function to be minimzed, put in a form that scipy's minimization
        function can work with.
        """
        
        # Convert thetas vector to form total_cost can understand
        thetas = self.reshape_thetas(thetas, 'list')
        self.thetas = thetas
        
        # Get cost function value
        fval = self.total_cost(X, Y, thetas)
        
        # Get derivatives using back propagation
        Deltas = self.get_gradients(X, Y)
        dfval = self.reshape_thetas(Deltas, 'vector')
        
        return fval, dfval
        
    def learn_thetas(self, X, Y):
        """
        Use scipy's conjugate gradient optimizer to find the thetas that
        minimize the cost function.
        """
        
        thetas0 = self.reshape_thetas(self.thetas, 'vector')
        res = minimize(self._objective_function, thetas0, args=(X, Y), method='Newton-CG', jac=True,
                       options = {'disp': False, 'maxiter': 400})
        
        self.thetas = self.reshape_thetas(res.x, 'list')
        
    def reshape_thetas(self, thetas, transform_type):
        """
        If 'thetas' is a vector then set 'transform_type' to 'list' and a list
        of appropriately shaped arrays for the neural network will be returned. 
        Alternately, if 'thetas' is a list of such arrays then set 
        'transform_type' to 'vector' and a vector of all the theta values will
        be returned.
        """
        
        if transform_type == 'vector':
            thetas_unrolled = thetas[0].ravel()
            for theta in thetas[1::]:
                thetas_unrolled = np.append(thetas_unrolled, theta.ravel())
                
            return thetas_unrolled
            
        elif transform_type == 'list':
            thetas_rolled = []
            num_elements = 0
            for theta_shape in self.thetas_shapes:
                elements = thetas[num_elements: num_elements + np.prod(theta_shape)]
                num_elements += np.prod(theta_shape)
                thetas_rolled.append(np.reshape(elements, theta_shape)) 
            
            return thetas_rolled
    
    def predict(self, X):
        """
        Uses the current theta parameters to predict the classification for
        input data X.
        """
        
        npredictions = X.shape[0]
        predictions = np.zeros(npredictions)
        for x, i in zip(X, xrange(npredictions)):
            predictions[i] = np.argmax(self._forward_prop(x)[-1])
        
        return predictions
        

 

                    
                    

        
    






