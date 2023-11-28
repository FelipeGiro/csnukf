import numpy as np
from csnukf import ClosedSkewNormal
import matplotlib.pyplot as plt

def get_csn_parameters():

    parameters_list_raw = [
        # ax, ls, label, mu_z, nu_z, Sigma_z, Gamma_z, Delta_z
        (-2.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 1.0),
        (2.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, -2.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, 2.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 0.5, 3.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 2.0, 3.0, 1.0),
        (0.0, 0.0, 1.0, -3.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 0.5),
        (0.0, 0.0, 1.0, 3.0, 1.0),
        (0.0, 0.0, 1.0, 3.0, 2.0),
        (3.0, 4.0, 2.0, -5.0, 3.0)
    ]

    # convert to dict
    parameters_list = list()
    for mu_x, nu_x, Sigma_x, Gamma_x, Delta_x in parameters_list_raw:
        parameters_list.append(
            {
                "mu_z" : mu_x, 
                "nu_z" : nu_x, 
                "Sigma_z" : Sigma_x, 
                "Gamma_z" : Gamma_x, 
                "Delta_z" : Delta_x
            }
        )

    return parameters_list

def get_all_example_pdf(x):
    csn_param_list = get_csn_parameters()

    pdf_list = list()
    for params_dict in csn_param_list:
        csn = ClosedSkewNormal(**params_dict)
        pdf_list.append(csn.pdf_z(x))
    
    return np.vstack(pdf_list)

def plot_effect_of_csn_parameters():
    csn_param_list = get_csn_parameters()
    ax_i_list = np.append(np.repeat([0,1,2,3,4], 3), [5])
    ls_list = np.tile(['-.', '-', '--'], 6)
    label_list = [
        '$\u03BC_x=-2$', '$\u03BC_x= 0$', '$\u03BC_x= 2$',
        '$\u03BD_x=-2$', '$\u03BD_x= 0$', '$\u03BD_x= 2$',
        '$\u03A3_x= 0.5$', '$\u03A3_x= 1$', '$\u03A3_x= 2$',
        '$\u0393_x=-3$', '$\u0393_x= 3$', '$\u0393_x= 0$',
        '$\u0394_x= 0.5$', '$\u0394_x= 1$', '$\u0394_x= 2$',
        '$\u03BC_x= 3$, $\u03A3_x= 2$, $\u0393_x=-5$, $\u03BD_x= 4$, $\u0394_x= 3$'
    ]

    x = np.linspace(-4, 4, num=401)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,7))
    axes = axes.flatten()
    # for ax_i, ls, label, mu_x, nu_x, Sigma_x, Gamma_x, Delta_x in parameters_list:
    for params_dict, ax_i, ls, label in zip(csn_param_list, ax_i_list, ls_list, label_list):
        csn = ClosedSkewNormal(**params_dict)
        csn_dist = csn.pdf_z(x)
        axes[ax_i].plot(x, csn_dist, color='k', ls=ls, label=label)
        axes[ax_i].set_xlim((-4, 4))
        axes[ax_i].set_ylim((0, 1))

    for ax, title in zip(axes, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        ax.legend(loc='upper right')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("$\pi(X)$")
        
    fig.tight_layout()
    
    return fig, axes

if __name__ == "__main__":

    fig, axes = plot_effect_of_csn_parameters()

    print(get_all_example_pdf(x=np.arange(-3,4)))