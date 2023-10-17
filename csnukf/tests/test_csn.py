import unittest

import numpy as np
from csnukf import ClosedSkewNormal

class TestCSN1n1q(unittest.TestCase):
    
    def setUp(self):

        # bi-variate
        self.csn1D_obj = ClosedSkewNormal(
            mu_z = np.array([[ 3.0]]),
            nu_z = np.array([[ 4.0]]),
            Sigma_z = np.array([[ 2.0]]),
            Gamma_z = np.array([[-5.0]]),
            Delta_z = np.array([[ 3.0]])
        )

        self.params = {
            "mu_z" : np.array([[ 3.0]]),
            "nu_z" : np.array([[ 4.0]]),
            "Sigma_z" : np.array([[ 2.0]]),
            "Gamma_z" : np.array([[-5.0]]),
            "Delta_z" : np.array([[ 3.0]]),
        }

    def test_conversion_1n1q(self):
        
        mu_z_0 = np.array([[ 3.0]])
        nu_z_0 = np.array([[ 4.0]])
        Sigma_z_0 = np.array([[ 2.0]])
        Gamma_z_0 = np.array([[-5.0]])
        Delta_z_0 = np.array([[ 3.0]])

        # bi-variate
        test_CSN = ClosedSkewNormal(**self.params)

        mu, Sigma, n, q = test_CSN.get_bivariate_parameters()

        # to z
        test_CSN = ClosedSkewNormal(
            mu = mu,
            Sigma = Sigma,
            n = n,
            q = q
        )
        mu_z_1, Sigma_z_1, Gamma_z_1, nu_z_1, Delta_z_1 = test_CSN.get_distribution_parameters()

        self.assertEqual(mu_z_0, mu_z_1)
        self.assertEqual(Sigma_z_0, Sigma_z_1)
        self.assertEqual(Gamma_z_0, Gamma_z_1)
        self.assertEqual(nu_z_0, nu_z_1)
        self.assertEqual(Delta_z_0, Delta_z_1)

        self.csn1D_obj = test_CSN

    def test_pdf_z(self):

        z = np.linspace(-4, 4, num=250)
        csn_pdf_arr = self.csn1D_obj.pdf_z(z)

        self.assertEqual(len(csn_pdf_arr.shape), 1)
        self.assertEqual(csn_pdf_arr.shape, z.shape)

    def test_bi_pdf(self):

        x_biv = np.linspace(-1, 2, num = 250)
        y_biv = np.linspace(-8, 8, num = 250)
        X, Y = np.meshgrid(x_biv, y_biv)
        pos = np.dstack((X, Y))

        z = np.linspace(-4, 4, num=250)
        csn_pdf_arr = self.csn1D_obj.pdf_bivariate(pos)

        self.assertEqual(len(csn_pdf_arr.shape), 2)

    def test_rvs(self, n_samples = int(1e6)):
        Y = self.csn1D_obj.rvs(n_samples)
        likelyhood_mean = self.csn1D_obj.pdf_z(Y).mean()

        self.assertLess(likelyhood_mean, .4)

class test_CSN2n2q(unittest.TestCase):

    def setUp(self) -> None:

        lambda_l = 0.2

        mu_0 = np.array([30, 2])*1e4 # altitude and velocity
        Delta_0 = np.eye(2)*(1 - lambda_l**2)
        Sigma_x = np.diag([1e3, 4e2])

        self.params = {
            "mu_z"    : mu_0,
            "Sigma_z" : np.diag([1e3, 4e2]),
            "Gamma_z" : lambda_l*Sigma_x**(1/2), 
            "nu_z"    : np.zeros(2),
            "Delta_z" : Delta_0,
        }

    def test_conversion_2n2q(self):

        # bi-variate
        test_CSN = ClosedSkewNormal(**self.params)

        mu, Sigma, n, q = test_CSN.get_bivariate_parameters()

        # to z
        test_CSN = ClosedSkewNormal(
            mu = mu,
            Sigma = Sigma,
            n = n,
            q = q
        )
        mu_z_1, Sigma_z_1, Gamma_z_1, nu_z_1, Delta_z_1 = test_CSN.get_distribution_parameters()

        self.assertSequenceEqual(
            self.params["mu_z"].flatten().round(8).tolist(), 
            mu_z_1.flatten().round(8).tolist()
            )
        self.assertSequenceEqual(
            self.params["Sigma_z"].flatten().round(8).tolist(), 
            Sigma_z_1.flatten().round(8).tolist()
            )
        self.assertSequenceEqual(
            self.params["Gamma_z"].flatten().round(8).tolist(), 
            Gamma_z_1.flatten().round(8).tolist()
            )
        self.assertSequenceEqual(
            self.params["nu_z"].flatten().round(8).tolist(), 
            nu_z_1.flatten().round(8).tolist()
            )
        self.assertSequenceEqual(
            self.params["Delta_z"].flatten().round(8).tolist(), 
            Delta_z_1.flatten().round(8).tolist()
            )

if __name__ == "__main__":
    unittest.main()