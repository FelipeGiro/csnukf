import numpy as np
from scipy.stats import norm, multivariate_normal

class ClosedSkewNormal:
    """
    Closed Skewed Normal
    ====================
    
    Input the bivariate normal parameters (mu and Sigma) OR directly the the parameters of the distribution
    (mu_z, Sigma_z, Gamma_z, nu_z, Delta_z).
    
    """
    def __init__(self, mu=None, Sigma=None, mu_z=None, Sigma_z=None, Gamma_z=None, nu_z=None, Delta_z=None):
        
        bivariate = np.all((mu==None) & (Sigma==None))
        distribution = (mu_z==None) & (Sigma_z==None) & (Gamma_z==None) & (nu_z==None) & (Delta_z==None)
        
        if ~bivariate & distribution:
            self.mu = np.atleast_2d(mu)
            self.Sigma = np.atleast_2d(Sigma)
            
            self._bivariate2z()
            
        elif bivariate & ~distribution:
            self.mu_z = np.atleast_2d(mu_z)
            self.Sigma_z = np.atleast_2d(Sigma_z)
            self.Gamma_z = np.atleast_2d(Gamma_z)
            self.nu_z = np.atleast_2d(nu_z)
            self.Delta_z = np.atleast_2d(Delta_z)
            
            self._z2bivariate()
        else:
            print("No value were correctly inputed.")
            
    def _bivariate2z(self):
        mu = self.mu
        Sigma = self.Sigma
        
        mu_x = mu[0]
        mu_y = mu[1]
        Sigma_x = Sigma[0, 0]
        Sigma_y = Sigma[1, 1]
        Gamma_xy = Sigma[0, 1]
        Gamma_yx = Sigma[1, 0]
        
        self.mu_z = np.atleast_2d(mu_x)
        self.Sigma_z = np.atleast_2d(Sigma_x)
        self.Gamma_z = np.atleast_2d(Gamma_yx/Sigma_x)
        self.nu_z = np.atleast_2d(-mu_y)
        self.Delta_z = np.atleast_2d(Sigma_y - Gamma_yx*Sigma_x**(-1)*Gamma_xy)
    
    def _z2bivariate(self):
        mu_z = float(self.mu_z)
        Sigma_z = float(self.Sigma_z)
        Gamma_z = float(self.Gamma_z)
        nu_z = float(self.nu_z)
        Delta_z = float(self.Delta_z)
    
        self.mu = np.array([
            [ mu_z],
            [-nu_z]
        ])
        self.Sigma = np.array([
            [Sigma_z        , Sigma_z*Gamma_z                  ],
            [Gamma_z*Sigma_z, Delta_z + Gamma_z*Sigma_z*Gamma_z]
        ])
    
    def pdf_bivariate(self, pos):
        """
        Get PDF of bivariate normal
        ===========================
        
        Get the underlying bi-variate normal of the CSN distribution.
        Similar to scipy.multivariate_normal.pdf.
        
        Parameters:
        -----------
        pos : array
        
        Return
        pdf : array
        
        """
        
        nu = self.mu
        Sigma = self.Sigma

        rv = multivariate_normal(nu.flatten(), Sigma)
        
        return rv.pdf(pos)
    
    def pdf_z(self, z):
        """
        Get PDF in z space
        ==================
        
        Parameters:
        -----------
        z : array
        
        Return
        pdf : array
        
        """
        
        mu_z = self.mu_z
        Sigma_z = self.Sigma_z
        Gamma_z = self.Gamma_z
        nu_z = self.nu_z
        Delta_z = self.Delta_z
        
        term1 = norm.cdf(0, nu_z, np.sqrt(Delta_z + Gamma_z*Sigma_z*Gamma_z.T))
        term2 = norm.cdf(Gamma_z*(z - mu_z), nu_z, np.sqrt(Delta_z))
        term3 = norm.pdf(z, mu_z, np.sqrt(Sigma_z))
        
        return (term1**(-1)*term2*term3).flatten()
    
    def get_bivariate_parameters(self):
        # get mean and covatiance of underlying bi-variate normal
        return self.mu, self.Sigma
    
    def get_distribution_parameters(self):
        # get paramters of CSN distribution
        return self.mu_z, self.Sigma_z, self.Gamma_z, self.nu_z, self.Delta_z
    
    def __str__(self):
        text = 'Closed Skewed Normal\n====================\n'
        text += 'Bivariate parameters:\n'
        text += f'  mu:\n{self.mu}\n'
        text += f'  Sigma:\n{self.Sigma}\n'
        text += 'Distribution parameters:\n'
        text += f'  mu_z: {self.mu_z}\n'
        text += f'  Sigma_z : {self.Sigma_z}\n'
        text += f'  Gamma_z : {self.Gamma_z}\n'
        text += f'  nu_z : {self.nu_z}\n'
        text += f'  Delta_z : {self.Delta_z}\n'
        return text