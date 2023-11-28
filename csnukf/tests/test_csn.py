import unittest

import numpy as np
from csnukf import ClosedSkewNormal
from csnukf.examples import get_all_example_pdf

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
        np.testing.assert_array_almost_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_almost_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_almost_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")

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
        np.testing.assert_array_almost_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_almost_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_almost_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")

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
        np.testing.assert_array_almost_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_almost_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_almost_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")

    def test_csn_n2q1(self):

        params_ref = {
            "mu" : np.array([[ 3.0], [1.2], [9]]),
            "Sigma" : np.array(
                [
                    [ 4.0, 2.1, .9],
                    [ 2.1, 6, 1.1],
                    [ .9, 1.1, 4.6]
                ]
                ),
            "n" : 2,
            "q" : 1
        }

        x, y = np.mgrid[-100:100:1, -100:100:1]
        z = np.dstack((x, y))

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

        np.testing.assert_array_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")


        # TODO test random variables sampling
        # samples = [insert right values here]
        np.random.seed(15000422)
        # np.testing.assert_array_almost_equal(csn_obj.rvs(10), samples, err_msg="Error in random variable sampling")
    
    
        # TODO test random variables sampling
        # samples = [insert right values here]
        np.random.seed(15000422)
        # np.testing.assert_array_almost_equal(csn_obj.rvs(10), samples, err_msg="Error in random variable sampling")
    
    def test_csn_n2q2(self):

        params_ref = {
            "mu_z" : np.array([[ 3.0, 4.0]]),
            "Sigma_z" : np.array(
                [
                    [ 8, 5.5],
                    [ 5.5, 6]
                    ]
                ),
            "nu_z" : np.array([[ -3.0, 4.0]]),
            "Gamma_z" : np.array(
                [
                    [ 4.0, 1.2],
                    [ 1.2, 3]
                    ]
                ),
            "Delta_z" : np.array(
                [
                    [ 3.0, -1],
                    [ -1, 9]
                    ]
                ),
        }

        x, y = np.mgrid[-100:100:1, -100:100:1]
        z = np.dstack((x, y))

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

        np.testing.assert_array_almost_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_almost_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_almost_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")

    def test_csn_n1q2(self):

        params_ref = {
            "mu" : np.array([[ 3.0], [1.2], [9]]),
            "Sigma" : np.array(
                [
                    [ 4.0, 2.1, .9],
                    [ 2.1, 6, 1.1],
                    [ .9, 1.1, 4.6]
                ]
                ),
            "n" : 1,
            "q" : 2
        }

        z = np.linspace(-100,100,201)

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

        np.testing.assert_array_almost_equal(csn_from_mvn.pdf(z).tolist(), csn_from_xy.pdf(z).tolist(), err_msg="pdf(z): CSN(mvn) != CSN(xy)")
        np.testing.assert_array_almost_equal(csn_from_xy.pdf(z).tolist(), csn_from_z.pdf(z).tolist(), err_msg="pdf(z): CSN(xy) != CSN(z)")
        np.testing.assert_array_almost_equal(csn_from_z.pdf(z).tolist(), csn_from_mvn.pdf(z).tolist(), err_msg="pdf(z): CSN(z) != CSN(mvn)")

class Test_CSN_errors(unittest.TestCase):
    def test_insuficient_paramters(self):
        with self.assertRaises(AttributeError):
            ClosedSkewNormal(
                mu_z = np.array([[ 3.0, 4.0]]),
                Delta_z = np.array(
                    [
                        [ 3.0, -1],
                        [ -1, 9]
                    ]
                )
            
            )

    def test_inconsistent_paramters_mvn(self):
        with self.assertRaises(AttributeError):
            ClosedSkewNormal(
                mu = np.array([[ 3.0, 4.0]]),
                Sigma = np.array(
                    [
                        [ 3.0, -1],
                        [ -1, 9]
                    ]
                ),
                n = 1,
                q = 1,
                mu_z = np.array(([[3.0]]))
            )

    def test_inconsistent_paramters_xy(self):
        with self.assertRaises(AttributeError):
            ClosedSkewNormal(
                mu_x = np.eye(1),
                Sigma_x = np.eye(1),
                mu_y = np.eye(1),
                Sigma_y = np.eye(1),
                Gamma_xy = np.eye(1),
                Gamma_yx = np.eye(1),
                mu = np.ones(2)
            )

    def test_inconsistent_paramters_z(self):
        with self.assertRaises(AttributeError):
            ClosedSkewNormal(
                mu_z = np.array([[-1.8]]),
                nu_z = np.array([[0.5]]),
                Sigma_z = np.array([[ 1.5]]),
                Gamma_z = np.array([[-2.0]]),
                Delta_z = np.array([[1.0]]),
                q = 0,
            )

class test_operations(unittest.TestCase):
    def setUp(self) -> None:
        # closed skew normal obsjects
        self.csn_1n1q_1 = ClosedSkewNormal(
            mu_z = np.array([[ 0.0]]),
            nu_z = np.array([[ 5.0]]),
            Sigma_z = np.array([[ 8.0]]),
            Gamma_z = np.array([[ 5.0]]),
            Delta_z = np.array([[ 3.0]])
        )

        self.csn_1n1q_2 = ClosedSkewNormal(
            mu_z = np.array([[-1.8]]),
            nu_z = np.array([[0.5]]),
            Sigma_z = np.array([[ 1.5]]),
            Gamma_z = np.array([[-2.0]]),
            Delta_z = np.array([[1.0]])
        )

        self.csn_1n2q_1 = ClosedSkewNormal(
            mu_z = np.array([[-1.8]]),
            nu_z = np.array([[0.5, -3.]]),
            Sigma_z = np.array([[ 1.5]]),
            Gamma_z = np.array([[-2.0], [4.0]]),
            Delta_z = np.array([[1.2, .4], [.4, 1.2]])
        )

        self.csn_2n1q_1 = ClosedSkewNormal(
            mu_z = np.array([[0.0], [1.0]]),
            Sigma_z = np.array([[ 1.0, .1], [.1, .8]]),
            nu_z = np.array([[ 0.0]]),
            Gamma_z = np.array([[ -2.0, -.5]]),
            Delta_z = np.array([[ 3.0]])
        )

        self.csn_2n1q_2 = ClosedSkewNormal(
            mu_z = np.array([[-5.0], [3.0]]),
            Sigma_z = np.array([[ 2.0, -.1], [-.1, 1.8]]),
            nu_z = np.array([[ 1.2]]),
            Gamma_z = np.array([[ -2.0, .8]]),
            Delta_z = np.array([[ .5]])
        )

    #########  EQUAL  #########

    def test_equal_to_iself(self):

        # True
        msg = "CSN object not equal to itself."
        self.assertTrue(self.csn_1n1q_1 == self.csn_1n1q_1, msg)
        self.assertTrue(self.csn_1n1q_2 == self.csn_1n1q_2, msg)
        self.assertTrue(self.csn_1n2q_1 == self.csn_1n2q_1, msg)
        self.assertTrue(self.csn_2n1q_1 == self.csn_2n1q_1, msg)
        self.assertTrue(self.csn_2n1q_2 == self.csn_2n1q_2, msg)

        # False
        msg = "Different CSN objects are equal."
        self.assertFalse(self.csn_1n1q_1 == self.csn_1n1q_2, msg)

    #########  SUM  #########

    def test_sumCSN_1(self):
        csn1, csn2 = self.csn_1n1q_1, self.csn_1n1q_2
        csn_result = csn1 + csn2

        mu_ref = np.array([-1.8, -5., -0.5])
        Sigma_ref =  np.array(
            [
                [  9.5,  40. ,  -3. ],
                [ 40. , 203. ,   0. ],
                [ -3. ,   0. ,   7. ]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    def test_sumCSN_2(self):
        csn1, csn2 = self.csn_1n1q_2, self.csn_1n2q_1
        csn_result = csn1 + csn2

        mu_ref = np.array([-3.6, -0.5, -0.5, 3. ])
        Sigma_ref = np.array(
            [
                [  3. ,  -3. ,  -3. ,   6. ],
                [ -3. ,   7. ,   0. ,   0. ],
                [ -3. ,   0. ,   7.2, -11.6],
                [  6. ,   0. , -11.6,  25.2]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    def test_sumCSN_3(self):
        csn1, csn2 = self.csn_2n1q_1, self.csn_2n1q_2
        csn_result = csn1 + csn2

        mu_ref = np.array([-5.,4.,-0.,-1.2])
        Sigma_ref = np.array(
            [
                [ 3.        ,  0.        , -2.05      ,  0.        ],
                [ 0.        ,  2.6       ,  0.        ,  1.64      ],
                [-2.05      ,  0.        ,  7.26153846, -2.40953846],
                [ 0.        ,  1.64      , -2.40953846,  4.4232    ]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    #########  SUBTRACTION  #########

    def test_subtractionCSN_1(self):
        csn1, csn2 = self.csn_1n1q_1, self.csn_1n1q_2
        csn_result = csn1 - csn2

        mu_ref = np.array([ 1.8, -5., -0.5])
        Sigma_ref = np.array(
            [
                [  9.5,  40. ,   3. ],
                [ 40. , 203. ,   0. ],
                [  3. ,   0. ,   7. ]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    def test_subtractionCSN_2(self):
        csn1, csn2 = self.csn_1n1q_2, self.csn_1n2q_1
        csn_result = csn1 - csn2

        mu_ref = np.array([ 0.,-0.5,-0.5,3. ])
        Sigma_ref = np.array(
            [
                [  3. ,  -3. ,   3. ,  -6. ],
                [ -3. ,   7. ,   0. ,   0. ],
                [  3. ,   0. ,   7.2, -11.6],
                [ -6. ,   0. , -11.6,  25.2]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    def test_subtractionCSN_3(self):
        csn1, csn2 = self.csn_2n1q_1, self.csn_2n1q_2
        csn_result = csn1 - csn2

        mu_ref = np.array([ 5., -2., -0., -1.2])
        Sigma_ref = np.array(
            [
                [ 3.        ,  0.        , -2.05      ,  0.        ],
                [ 0.        ,  2.6       ,  0.        , -1.64      ],
                [-2.05      ,  0.        ,  7.26153846,  2.40953846],
                [ 0.        , -1.64      ,  2.40953846,  4.4232    ]
            ]
        )

        # assert dimensions
        self.assertEqual(csn_result.q, csn1.q + csn2.q, "q_result != q_1 + q_2")
        self.assertEqual(csn_result.n, csn1.n, "n_result != n_1")
        self.assertEqual(csn_result.n, csn2.n, "n_result != n_2")

        # assert mu and Sigma
        # print(csn_result.get_mvn_parameters())
        np.testing.assert_almost_equal(mu_ref, csn_result.mu.flatten(), err_msg="Parameter mu is wrong.")
        np.testing.assert_almost_equal(Sigma_ref, csn_result.Sigma, err_msg="Parameter Sigma is wrong.")

    # TODO: #########  MULTIPLICATION  #########

    # def test_multiplication_CSN_with_constant_1(self):
    #     csn, cte = self.csn_1n1q_1, np.pi
    #     csn_times_pi = csn*cte
    #     csn_plus_csn = csn + csn + csn

    #     print(csn_times_pi)
    #     print(csn_plus_csn)

class test_CSN_1D_sampling(unittest.TestCase):
    def setUp(self):
        self.csn_1 = ClosedSkewNormal(
            mu_z = np.array([[ 0.0]]),
            nu_z = np.array([[ 5.0]]),
            Sigma_z = np.array([[ 8.0]]),
            Gamma_z = np.array([[ 5.0]]),
            Delta_z = np.array([[ 3.0]])
        )
        self.csn_2 = ClosedSkewNormal(
            mu_z = np.array([[-1.8]]),
            nu_z = np.array([[0.5]]),
            Sigma_z = np.array([[ 1.5]]),
            Gamma_z = np.array([[-2.0]]),
            Delta_z = np.array([[1.0]])
        )
        self.csn_3 = ClosedSkewNormal(
            mu_z = np.array([[ 0.0]]),
            nu_z = np.array([[ 0.0]]),
            Sigma_z = np.array([[ 1.0]]),
            Gamma_z = np.array([[ -2.0]]),
            Delta_z = np.array([[ 3.0]])
        )

    def test_csn1_1D_rvs(self):
        ref_samples = [1.29217214, 1.57001725, 0.76932106, 1.89847131, 1.72734761]
        np.random.seed(15000422)
        with np.errstate(invalid='ignore'): # sometimes, g(x)=0
            samples = self.csn_1.rvs(5)
        
        np.testing.assert_array_almost_equal(ref_samples, samples, err_msg="Wrong sampling from CSN.")

    def test_csn2_1D_rvs(self):
        ref_samples = [-2.55132361, -1.81362587, -4.27915516, -1.62783108, -3.08272455]
        np.random.seed(18220907)
        with np.errstate(invalid='ignore'): # sometimes, g(x)=0
            samples = self.csn_2.rvs(5)

        np.testing.assert_array_almost_equal(ref_samples, samples, err_msg="Wrong sampling from CSN.")

    def test_csn3_1D_rvs(self):
        ref_samples = [-1.22783799, -0.19801904, -2.21566253, 0.17347386, -1.34880619]
        np.random.seed(18891115)
        with np.errstate(invalid='ignore'): # sometimes, g(x)=0
            samples = self.csn_3.rvs(5)

        np.testing.assert_array_almost_equal(ref_samples, samples, err_msg="Wrong sampling from CSN.")

class test_examples(unittest.TestCase):
    def test_Rezaie_parameters_effect(self):
        # Figure 2: effect of CSN distribution parameters (Rezaie & Eidsvik, 2016) 
        pdfs_ref_arr = np.array(
            [
                [6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03, 2.67660452e-04, 2.97343903e-06], #  0 : (a) lower
                [1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03], #  1 : (a) middle
                [1.09153942e-56, 4.75494004e-37, 1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01], #  2 : (a) upper
                [7.70166857e-15, 2.32187994e-06, 5.21279744e-02, 5.29382100e-01, 3.28561193e-01, 7.33119325e-02, 6.01780988e-03], #  3 : (b) lower
                [1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03], #  4 : (b) middle
                [3.21302455e-30, 1.27445464e-16, 2.63186121e-07, 3.44381503e-02, 7.72471816e-01, 2.04858118e-01, 1.68163109e-02], #  5 : (b) upper
                [1.57159380e-23, 2.03897924e-11, 5.60352794e-04, 5.64189584e-01, 4.14547145e-01, 2.06669853e-02, 1.39253052e-04], #  6 : (c) lower
                [1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03], #  7 : (c) middle
                [6.71116728e-21, 2.04769964e-10, 5.93133437e-04, 2.82094792e-01, 4.38798156e-01, 2.07553749e-01, 5.94651446e-02], #  8 : (c) upper
                [8.86369682e-03, 1.07981933e-01, 4.83288177e-01, 3.98942280e-01, 6.53271609e-04, 1.06533641e-10, 1.00034655e-21], #  9 : (d) lower
                [1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03], # 10 : (d) middle
                [4.43184841e-03, 5.39909665e-02, 2.41970725e-01, 3.98942280e-01, 2.41970725e-01, 5.39909665e-02, 4.43184841e-03], # 11 : (d) upper
                [1.83346976e-39, 1.16187138e-18, 5.34525356e-06, 3.98942280e-01, 4.83936104e-01, 1.07981933e-01, 8.86369682e-03], # 12 : (e) lower
                [1.00034655e-21, 1.06533641e-10, 6.53271609e-04, 3.98942280e-01, 4.83288177e-01, 1.07981933e-01, 8.86369682e-03], # 13 : (e) middle
                [8.71372503e-13, 1.19268728e-06, 8.20156226e-03, 3.98942280e-01, 4.75739887e-01, 1.07980740e-01, 8.86369682e-03], # 14 : (e) upper
                [1.19489084e-04, 1.86912375e-03, 1.77337524e-02, 1.02050699e-01, 3.56097191e-01, 5.41525918e-01, 1.01283330e-02] # 15 : (f) all
            ]
        )
        pdfs_arr = get_all_example_pdf(x=np.arange(-3,4))

        for pdf_ref, pdf, plot_id in zip(pdfs_ref_arr, pdfs_arr, np.arange(16)):
            np.testing.assert_array_almost_equal(pdf_ref, pdf, err_msg=f"Difference at axis id {plot_id}")

if __name__ == "__main__":
    unittest.main()