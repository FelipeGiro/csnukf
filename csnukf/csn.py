import numpy as np
from scipy.stats import norm, multivariate_normal

class ClosedSkewNormal:
    """
    Closed Skewed Normal
    ====================
    
    Input the bivariate normal parameters (mu and Sigma) OR directly the the parameters of the distribution
    (mu_z, Sigma_z, Gamma_z, nu_z, Delta_z).
    
    """
    def __init__(
            self, 
            mu = None, Sigma = None, n = None, q = None,
            mu_z = None, Sigma_z = None, Gamma_z = None, nu_z = None, Delta_z = None,
            mu_x = None, mu_y = None, Sigma_x = None, Sigma_y = None, Gamma_xy = None, # Gamma_yx = None,
            ):

        # check variables availability
        var_aval = (mu is not None) & (Sigma is not None) & (n is not None) & (q is not None)
        var_aval_z = (mu_z is not None) & (Sigma_z is not None) & (Gamma_z is not None) & (nu_z is not None) & (Delta_z is not None)
        var_aval_xy = (mu_x is not None) & (mu_y is not None) & (Sigma_x is not None) & (Sigma_y is not None) & (Gamma_xy is not None)

        if var_aval & ~var_aval_z & ~var_aval_xy:
            self.mu = np.atleast_1d(mu).flatten()
            self.Sigma = np.atleast_2d(Sigma)
            self.n = n
            self.q = q
            
            self._check_dims()

            self._bivariate2z()
            self._bivariate2xy()
            
        elif ~var_aval & var_aval_z & ~var_aval_xy:

            self.mu_z = np.atleast_1d(mu_z).flatten()
            self.Sigma_z = np.atleast_2d(Sigma_z)
            self.Gamma_z = np.atleast_2d(Gamma_z)
            self.nu_z = np.atleast_1d(nu_z).flatten()
            self.Delta_z = np.atleast_2d(Delta_z)

            self.n = len(self.mu_z)
            self.q = len(self.nu_z)

            self._check_dims_z()

            self._z2bivariate()
            self._z2xy()

        elif ~var_aval & ~var_aval_z & var_aval_xy:
            
            Gamma_yx = Gamma_xy

            self.mu_x = np.atleast_1d(mu_x)
            self.mu_y = np.atleast_1d(mu_y)
            self.Sigma_x = np.atleast_2d(Sigma_x)
            self.Sigma_y = np.atleast_2d(Sigma_y)
            self.Gamma_xy = np.atleast_2d(Gamma_xy)
            self.Gamma_yx = np.atleast_2d(Gamma_yx)

            self.n = len(self.mu_x)
            self.q = len(self.mu_y)

            self._check_dims_xy()
            self._xy2z()
            self._xy2bivariate()

        else:
            raise AttributeError("Wrong value inputed")
    
    def _check_dims(self):
        n_plus_q = self.n + self.q
        
        if self.mu.shape != (n_plus_q, ):
            raise AttributeError(
                "Shape of mu {} is different of n+q {}".format(self.mu.shape, (n_plus_q, ))
                )
        if self.Sigma.shape != (n_plus_q, n_plus_q):
            raise AttributeError(
                "Shape of Sigma {} is different of n+q,n+q {}".format(self.Sigma.shape, (n_plus_q, n_plus_q))
                )

    def _check_dims_z(self):
        n_z, q_z = self.n, self.q

        if self.mu_z.shape != (n_z, ):
            raise AttributeError(
                "Shape of mu_z {} is different of n {}".format(self.mu_z.shape, (n_z, ))
                )
        if self.Sigma_z.shape != (n_z, n_z):
            raise AttributeError(
                "Shape of Sigma_z {} is different of n x n {}".format(self.Sigma_z.shape, (n_z, n_z))
                )
        if self.Gamma_z.shape != (q_z, n_z):
            raise AttributeError(
                "Shape of Gamma_z_z {} is different of q x n {}".format(self.Gamma_z.shape, (q_z, n_z))
                )
        if self.nu_z.shape != (q_z, ):
            raise AttributeError(
                "Shape of nu_z {} is different of q {}".format(self.nu_z.shape, (q_z, ))
                )
        if self.Delta_z.shape != (q_z, q_z):
            raise AttributeError(
                "Shape of Delta_z {} is different of q x q {}".format(self.Delta_z.shape, (q_z, q_z))
                )

    def _check_dims_xy(self):
        n_plus_q = self.n + self.q
        
        if self.mu.shape != (n_plus_q, ):
            raise AttributeError(
                "Shape of mu {} is different of n+q {}".format(self.mu.shape, (n_plus_q, ))
                )
        if self.Sigma.shape != (n_plus_q, n_plus_q):
            raise AttributeError(
                "Shape of Sigma {} is different of n+q,n+q {}".format(self.Sigma.shape, (n_plus_q, n_plus_q))
                )

    def _bivariate2z(self):
        mu = self.mu
        Sigma = self.Sigma
        n, q = self.n, self.q
        
        mu_x = mu[:n]
        mu_y = mu[n:]
        Sigma_x = Sigma[:n, :n]
        Sigma_y = Sigma[n:, n:]
        Gamma_xy = Sigma[:n, n:]
        Gamma_yx = Sigma[n:, :n]
        
        self.mu_z = np.atleast_1d(mu_x).flatten()
        self.Sigma_z = np.atleast_2d(Sigma_x)
        self.Gamma_z = np.atleast_2d(np.matmul(Gamma_yx, np.linalg.inv(Sigma_x)))
        self.nu_z = np.atleast_1d(-mu_y).flatten()
        self.Delta_z = np.atleast_2d(
            Sigma_y - np.matmul(
                np.matmul(
                    Gamma_yx, np.linalg.inv(Sigma_x)
                    ), Gamma_xy
                )
                )
        
    def _bivariate2xy(self):
        pass
    
    def _z2bivariate(self):
        mu_z = self.mu_z
        Sigma_z = self.Sigma_z
        Gamma_z = self.Gamma_z
        nu_z = self.nu_z
        Delta_z = self.Delta_z
    
        self.mu = np.vstack([mu_z,-nu_z]).flatten()
        self.Sigma = np.block([
            [Sigma_z        , Sigma_z*Gamma_z                  ],
            [Gamma_z*Sigma_z, Delta_z + Gamma_z*Sigma_z*Gamma_z]
        ])

    def _z2xy(self):
        pass

    def _xy2bivariate(self):
        pass

    def _xy2z(self):
        pass
    
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
        return self.mu, self.Sigma, self.n, self.q
    
    def get_distribution_parameters(self):
        # get paramters of CSN distribution
        return self.mu_z, self.Sigma_z, self.Gamma_z, self.nu_z, self.Delta_z
    
    def __str__(self):
        text = 'Closed Skewed Normal\n====================\n'
        text += f'  n, q:\n({self.n}, {self.q})\n'
        text += 'Bivariate parameters:\n'
        text += f'  mu:\n{self.mu}\n'
        text += f'  Sigma:\n{self.Sigma}\n'
        text += 'Distribution parameters:\n'
        text += f'  mu_z:\n{self.mu_z}\n'
        text += f'  Sigma_z:\n{self.Sigma_z}\n'
        text += f'  Gamma_z:\n{self.Gamma_z}\n'
        text += f'  nu_z:\n{self.nu_z}\n'
        text += f'  Delta_z:\n{self.Delta_z}\n'
        return text

if __name__ == "__main__":
    lambda_l = 0.2

    mu_0 = np.array([30, 2])*1e4 # altitude and velocity
    Delta_0 = np.eye(2)*(1 - lambda_l**2)
    Sigma_x = np.diag([1e3, 4e2])

    obj = ClosedSkewNormal(
        mu_z=mu_0, 
        Sigma_z=Sigma_x, 
        Gamma_z=lambda_l*Sigma_x**(1/2), 
        nu_z=np.zeros(2), 
        Delta_z=Delta_0,
    )
    print(obj)