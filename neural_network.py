import numpy as np

class Neural_Network(object):
    
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
        
#       Initialize Deltas
        Deltas = []
        for theta in thetas:
            Deltas.append(np.zeros_like(theta))
            
        self.Deltas = Deltas
          
    
    def _sigmoid(self, z):
        """
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
    
    def _update_Deltas(self, a, deltas):
        """
        Add the derivative contribution of each training example. 
        """
        
        a = a[-2::-1]        
        for Delta, delta, ai in zip(reversed(self.Deltas), deltas, a):
            Delta += np.outer(delta, ai)
        
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
        
        Deltas = self.Deltas
        m_inv = 1./X.shape[0]    
        for x, true_indx in zip(X, Y):
            y = np.zeros(self.noutputs)
            y[true_indx] = 1.
            a = self._forward_prop(x)
            deltas = self._back_prop(a, y)
            self._update_Deltas(a, deltas)
        
        Deltas = map(lambda x: x*m_inv, Deltas)
        
        return Deltas
    
    def gradient_check(self, X, Y, eps = 1e-4):
        """
        Compares the derivatives of the cost function with respect to each 
        regression parameter, as determined using back propagation and central
        differencing.
        """
        
        back_prop_grads = self.get_gradients(X, Y)
        
        eps_normalization = 0.5/eps        
        for theta, grad in zip(self.thetas, back_prop_grads):
            m, n = theta.shape
            for mi in range(m):
                for ni in range(n):
                    theta_original_value = theta[mi, ni]
                    theta[mi, ni] += eps
                    J_p_eps = self.total_cost(X, Y, self.thetas)
                    theta[mi, ni] -= 2.*eps
                    J_m_eps = self.total_cost(X, Y, self.thetas)
                    theta[mi, ni] = theta_original_value
                    print (J_p_eps - J_m_eps)*eps_normalization, grad[mi, ni]
                    
                    
# Use iris data set
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target
        
    
nn = Neural_Network(4, 3, 2)





