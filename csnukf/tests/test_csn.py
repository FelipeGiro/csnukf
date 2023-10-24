import unittest

import numpy as np
from csnukf import ClosedSkewNormal

from scipy.stats import multivariate_normal

class Test_CSN(unittest.TestCase):
    def setUp(self):
        self.z = np.atleast_2d(np.linspace(-100,100, 201))
    
    def test_csn_n1q0(self):
        params_ref = {
            "mu" : np.array([[ 3.0]]),
            "Sigma" : np.array([[ 4.0]]),
            "n" : 1,
            "q" : 0
        }
        csn_obj = ClosedSkewNormal(**params_ref)

        params_mvn = csn_obj.get_parameters("mvn", "dict")
        params_xy = csn_obj.get_parameters("xy", "dict")
        params_z = csn_obj.get_parameters("z", "dict")

        csn_from_mvn = ClosedSkewNormal(**params_mvn)
        csn_from_xy = ClosedSkewNormal(**params_xy)
        csn_from_z = ClosedSkewNormal(**params_z)

        self.assertTrue(csn_from_mvn == csn_from_xy, "CSN(mvn) != CSN(xy)")
        self.assertTrue(csn_from_xy == csn_from_z, "CSN(xy) != CSN(z)")
        self.assertTrue(csn_from_z == csn_from_mvn, "CSN(z) != CSN(mvn)")

        self.assertTrue(csn_from_mvn == csn_from_mvn, "CSN(mvn) != CSN(mvn)")
        self.assertTrue(csn_from_xy == csn_from_xy, "CSN(xy) != CSN(xy)")
        self.assertTrue(csn_from_z == csn_from_z, "CSN(z) != CSN(z)")

        z = np.repeat(self.z, repeats=csn_obj.n, axis=0).T
        self.assertListEqual(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")

    def test_csn_n1q1(self):

        params_ref = {
            "mu_z" : np.array([[ 3.0]]),
            "Sigma_z" : np.array([[ 2.0]]),
            "nu_z" : np.array([[ 4.0]]),
            "Gamma_z" : np.array([[-5.0]]),
            "Delta_z" : np.array([[ 3.0]])
        }

        csn_obj = ClosedSkewNormal(**params_ref)

        params_mvn = csn_obj.get_parameters("mvn", "dict")
        params_xy = csn_obj.get_parameters("xy", "dict")
        params_z = csn_obj.get_parameters("z", "dict")

        csn_from_mvn = ClosedSkewNormal(**params_mvn)
        csn_from_xy = ClosedSkewNormal(**params_xy)
        csn_from_z = ClosedSkewNormal(**params_z)

        self.assertTrue(csn_from_mvn == csn_from_xy, "CSN(mvn) != CSN(xy)")
        self.assertTrue(csn_from_xy == csn_from_z, "CSN(xy) != CSN(z)")
        self.assertTrue(csn_from_z == csn_from_mvn, "CSN(z) != CSN(mvn)")

        self.assertTrue(csn_from_mvn == csn_from_mvn, "CSN(mvn) != CSN(mvn)")
        self.assertTrue(csn_from_xy == csn_from_xy, "CSN(xy) != CSN(xy)")
        self.assertTrue(csn_from_z == csn_from_z, "CSN(z) != CSN(z)")

        z = np.repeat(self.z, repeats=csn_obj.n, axis=0).T
        self.assertListEqual(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
    
    def test_csn_n1q2(self):

        params_ref = {
            
            "mu" : np.array([[ 3.0], [1.2], [9]]),
            "Sigma" : np.array(
                [
                    [ 12.0, 5.5, .9],
                    [ 5.5, 6, 1.1],
                    [ .9, 1.1, 1.6]
                    ]
                ),
            "n" : 1,
            "q" : 2
        }

        csn_obj = ClosedSkewNormal(**params_ref)

        params_mvn = csn_obj.get_parameters("mvn", "dict")
        params_xy = csn_obj.get_parameters("xy", "dict")
        params_z = csn_obj.get_parameters("z", "dict")

        csn_from_mvn = ClosedSkewNormal(**params_mvn)
        csn_from_xy = ClosedSkewNormal(**params_xy)
        csn_from_z = ClosedSkewNormal(**params_z)

        self.assertTrue(csn_from_mvn == csn_from_xy, "CSN(mvn) != CSN(xy)")
        self.assertTrue(csn_from_xy == csn_from_z, "CSN(xy) != CSN(z)")
        self.assertTrue(csn_from_z == csn_from_mvn, "CSN(z) != CSN(mvn)")

        self.assertTrue(csn_from_mvn == csn_from_mvn, "CSN(mvn) != CSN(mvn)")
        self.assertTrue(csn_from_xy == csn_from_xy, "CSN(xy) != CSN(xy)")
        self.assertTrue(csn_from_z == csn_from_z, "CSN(z) != CSN(z)")

        z = np.repeat(self.z, repeats=csn_obj.n, axis=0).T
        self.assertListEqual(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")
        self.assertListEqual(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), "pdf(z): CSN(mvn) != CSN(xy)")

    def test_csn_n2q1(self):

        params_ref = {
            "mu" : np.array([[ 3.0], [1.2], [9]]),
            "Sigma" : np.array(
                [
                    [ 2.0, 5.5, .9],
                    [ 5.5, 6, 1.1],
                    [ .9, 1.1, 4.6]
                    ]
                ),
            "n" : 2,
            "q" : 1
        }

        csn_obj = ClosedSkewNormal(**params_ref)

        params_mvn = csn_obj.get_parameters("mvn", "dict")
        params_xy = csn_obj.get_parameters("xy", "dict")
        params_z = csn_obj.get_parameters("z", "dict")

        csn_from_mvn = ClosedSkewNormal(**params_mvn)
        csn_from_xy = ClosedSkewNormal(**params_xy)
        csn_from_z = ClosedSkewNormal(**params_z)

        self.assertTrue(csn_from_mvn == csn_from_xy, "CSN(mvn) != CSN(xy)")
        self.assertTrue(csn_from_xy == csn_from_z, "CSN(xy) != CSN(z)")
        self.assertTrue(csn_from_z == csn_from_mvn, "CSN(z) != CSN(mvn)")

        self.assertTrue(csn_from_mvn == csn_from_mvn, "CSN(mvn) != CSN(mvn)")
        self.assertTrue(csn_from_xy == csn_from_xy, "CSN(xy) != CSN(xy)")
        self.assertTrue(csn_from_z == csn_from_z, "CSN(z) != CSN(z)")
    
    def test_csn_n2q2(self):

        params_ref = {
            "mu_z" : np.array([[ 3.0, 4.0]]),
            "Sigma_z" : np.array(
                [
                    [ 2.0, 5.5],
                    [ 5.5, 6]
                    ]
                ),
            "nu_z" : np.array([[ -3.0, 4.0]]),
            "Gamma_z" : np.array(
                [
                    [ 4.0, 2.2],
                    [ 2.2, 3]
                    ]
                ),
            "Delta_z" : np.array(
                [
                    [ 3.0, -1],
                    [ -1, 9]
                    ]
                ),
        }

        csn_obj = ClosedSkewNormal(**params_ref)

        params_mvn = csn_obj.get_parameters("mvn", "dict")
        params_xy = csn_obj.get_parameters("xy", "dict")
        params_z = csn_obj.get_parameters("z", "dict")

        csn_from_mvn = ClosedSkewNormal(**params_mvn)
        csn_from_xy = ClosedSkewNormal(**params_xy)
        csn_from_z = ClosedSkewNormal(**params_z)

        self.assertTrue(csn_from_mvn == csn_from_xy, "CSN(mvn) != CSN(xy)")
        self.assertTrue(csn_from_xy == csn_from_z, "CSN(xy) != CSN(z)")
        self.assertTrue(csn_from_z == csn_from_mvn, "CSN(z) != CSN(mvn)")

        self.assertTrue(csn_from_mvn == csn_from_mvn, "CSN(mvn) != CSN(mvn)")
        self.assertTrue(csn_from_xy == csn_from_xy, "CSN(xy) != CSN(xy)")
        self.assertTrue(csn_from_z == csn_from_z, "CSN(z) != CSN(z)")

class test_operations(unittest.TestCase):
    def setUp(self) -> None:
        # closed skew normal obsjects
        self.csn1_1n1q = ClosedSkewNormal(
            mu_z = np.array([[ 0.0]]),
            nu_z = np.array([[ 5.0]]),
            Sigma_z = np.array([[ 8.0]]),
            Gamma_z = np.array([[ 5.0]]),
            Delta_z = np.array([[ 3.0]])
        )

        self.csn2_1n1q = ClosedSkewNormal(
            mu_z = np.array([[-1.8]]),
            nu_z = np.array([[0.5]]),
            Sigma_z = np.array([[ 1.5]]),
            Gamma_z = np.array([[-2.0]]),
            Delta_z = np.array([[1.0]])
        )

        self.csn3_1n1q = ClosedSkewNormal(
            mu_z = np.array([[ 0.0]]),
            nu_z = np.array([[ 0.0]]),
            Sigma_z = np.array([[ 1.0]]),
            Gamma_z = np.array([[ -2.0]]),
            Delta_z = np.array([[ 3.0]])
        )

        self.csn4_2n1q = ClosedSkewNormal(
            mu_z = np.array([[0.0], [1.0]]),
            Sigma_z = np.array([[ 1.0, .1], [.1, .8]]),
            nu_z = np.array([[ 0.0]]),
            Gamma_z = np.array([[ -2.0, -.5]]),
            Delta_z = np.array([[ 3.0]])
        )

    def test_sumCSNs_1(self):
        csn_result = self.csn1_1n1q + self.csn2_1n1q

        self.assertEqual(csn_result.q, self.csn1_1n1q.q + self.csn2_1n1q.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, self.csn1_1n1q.n, "n_result != n_1")
        self.assertEqual(csn_result.n, self.csn2_1n1q.n, "n_result != n_2")

    def test_sumCSNs_2(self):
        pass

    def test_sumCSNs_3(self):
        pass

    def test_sumCSNs_4(self):
        pass

    def test_sumCSNs_5(self):
        pass

if __name__ == "__main__":
    unittest.main()