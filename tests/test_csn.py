import unittest

import numpy as np
from csnukf import ClosedSkewNormal

class TestCSN1D(unittest.TestCase):
    
    def setUp(self):

        # bi-variate
        self.csn1D_obj = ClosedSkewNormal(
            mu_z = np.array([[ 3.0]]),
            nu_z = np.array([[ 4.0]]),
            Sigma_z = np.array([[ 2.0]]),
            Gamma_z = np.array([[-5.0]]),
            Delta_z = np.array([[ 3.0]])
        )

    def test_conversion_00(self):
        
        mu_z_0 = np.array([[ 3.0]])
        nu_z_0 = np.array([[ 4.0]])
        Sigma_z_0 = np.array([[ 2.0]])
        Gamma_z_0 = np.array([[-5.0]])
        Delta_z_0 = np.array([[ 3.0]])

        # bi-variate
        test_CSN = ClosedSkewNormal(
            mu_z = mu_z_0,
            nu_z = nu_z_0,
            Sigma_z = Sigma_z_0,
            Gamma_z = Gamma_z_0,
            Delta_z = Delta_z_0
        )

        mu, Sigma = test_CSN.get_bivariate_parameters()

        # to z
        test_CSN = ClosedSkewNormal(
            mu = mu,
            Sigma = Sigma
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

if __name__ == "__main__":
    unittest.main()