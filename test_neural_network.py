from neural_network import *
import unittest

class TestNeuralNetwork(unittest.TestCase):
    """
    Use the iris data set to test functions in the NeuralNetwork class. 
    """

    def setUp(self):
        """
        """
        
        # Use iris data set
        from sklearn import datasets
        
        iris = datasets.load_iris()
        self.X = iris.data
        self.Y = iris.target
        
    def test_gradients(self):
        """
        Compares the derivatives of the cost function with respect to each 
        regression parameter, as determined using back propagation and central
        differencing.
        """
        
        eps = 1e-4
        nn = NeuralNetwork(4, 3, 2)
        X = self.X
        Y = self.Y
        
        back_prop_grads = nn.get_gradients(X, Y)
        
        eps_normalization = 0.5/eps        
        for theta, grad in zip(nn.thetas, back_prop_grads):
            m, n = theta.shape
            for mi in range(m):
                for ni in range(n):
                    theta_original_value = theta[mi, ni]
                    theta[mi, ni] += eps
                    J_p_eps = nn.total_cost(X, Y, nn.thetas)
                    theta[mi, ni] -= 2.*eps
                    J_m_eps = nn.total_cost(X, Y, nn.thetas)
                    theta[mi, ni] = theta_original_value
                    central_difference = (J_p_eps - J_m_eps)*eps_normalization
                    
                    # Assert that the difference between back prop. gradient and
                    # central difference gradient is within 10E-6 of each other
                    self.assertGreater(1e-6, central_difference - grad[mi, ni])
          

if __name__ == '__main__':
    unittest.main()

