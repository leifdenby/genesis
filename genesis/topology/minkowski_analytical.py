
# coding: utf-8

# In[287]:

import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plot
import seaborn as sns
from matplotlib.patches import Ellipse, Arc

sns.set(style='ticks')

@np.vectorize
def f(x):
    # http://www.suitcaseofdreams.net/inverse_functions.htm#P1
    d = np.sqrt(1 + 0J - x**2.)

    if d.imag > 0.0:
        assert np.all(d.real == 0.0)
        return np.log(x + np.sqrt(x**2. - 1.))/d.imag
    else:
        return np.arccos(x)/d.real

def f2(x):
    return np.arccos(x)/np.sqrt(1 - x**2.)

def spheroid_minkowski(r, lm):
    V0 = 4.*pi/3.*r**3.*lm
    V1 = pi/3.*r**2.*(1+f(1./lm))
    V2 = 2./3.*r*(lm+f(lm))
    V3 = 1.
    
    return V0, V1, V2, V3

@np.vectorize
def spheroid_minkowski2(r, lm):
    V0 = 4.*pi/3.*r**3.*lm
    
    if lm > 1.0:
        e = np.sqrt(1 - (1./lm)**2.)
        V1 = 2./6.*pi*r**2.*(1 + lm/e*np.arcsin(e))
    else:
        e = np.sqrt(1 - lm**2.)
        V1 = 2./6.*pi*r**2.*(1 + (1 - e**2.)/e*np.arctanh(e))
    
    V2 = 2./3.*r*(lm+f(lm))
    V3 = 1.
    
    return V0, V1, V2, V3

def cylinder_minkowski(r, lm):
    V0 = pi*r**3.*lm
    V1 = pi/3.*r**2.*(1+lm)
    V2 = 1./3.*r*(pi+lm)
    V3 = 1.
    
    return V0, V1, V2, V3

def length_scales(fn_mink, r, lm):
    V0, V1, V2, V3 = fn_mink(r=r, lm=lm)
    
    T = V0/(2.*V1)
    W = 2*V1/(pi*V2)
    L = 3*V2/(4*V3)
    
    return L, W, T
    
def filamentarity_planarity(fn_mink, r, lm):
    L, W, T = length_scales(fn_mink=fn_mink, r=r, lm=lm)
    
    P = (W-T)/(W+T)
    F = (L-W)/(L+W)
    
    return F, P


def plot_cylinder_diagram(ax, x_c, y_c, l, r, color, r_label="r", h_label="h"):
    e_l = 0.1*l
    # sides
    ax.add_line(plot.Line2D((x_c-r, x_c-r), (y_c-l/2., y_c+l/2.), color=color))
    ax.add_line(plot.Line2D((x_c+r, x_c+r), (y_c-l/2., y_c+l/2.), color=color))
    # centerline
    ax.add_line(plot.Line2D((x_c, x_c), (y_c-l/2., y_c+l/2.), color=color, ls='--'))
    # radius indicator
    ax.add_line(plot.Line2D((x_c-r, x_c), (y_c, y_c), color=color, ls='--'))
    
    # ends
    c = ax.get_lines()[-1].get_color()
    ax.add_patch(Ellipse((x_c, y_c-l/2.), r*2, e_l, facecolor="None", edgecolor=c, linewidth=2, 
                         linestyle=":", alpha=0.4))
    ax.add_patch(Arc((x_c, y_c-l/2.), r*2, e_l, facecolor="None", edgecolor=c, linewidth=2,
                        theta1=180, theta2=360))

    ax.add_patch(Ellipse((x_c, y_c+l/2.), r*2, e_l, facecolor="None", edgecolor=c, linewidth=2))

    
    # labels
    ax.annotate(r_label, (x_c - r/2., y_c), color=c,  xytext=(0, 6), textcoords='offset points')
    ax.annotate(h_label, (x_c, y_c + l/4), color=c,  xytext=(6, 0), textcoords='offset points')

def plot_spheroid_diagram(ax, x_c, y_c, l, r, color, r_label="r", h_label="h", render_back=True):
    w = 2*r
    w_yz = w/3.
    # yz-plane arc
    if render_back:
        ax.add_patch(Arc((x_c, y_c), w_yz, l*2, facecolor="None", edgecolor=color, 
                         linewidth=2, linestyle=':', alpha=0.4))
    ax.add_patch(Arc((x_c, y_c), w_yz, l*2, facecolor="None", edgecolor=color, 
                     linewidth=2, linestyle='-', theta1=90., theta2=270.))
    
    # xy-plane arc
    if render_back:
        ax.add_patch(Arc((x_c, y_c), w, l, facecolor="None", edgecolor=color, 
                       linewidth=2, linestyle=':', theta1=0, theta2=180, alpha=0.4))
    a_xy = Arc((x_c, y_c), w, l, facecolor="None", edgecolor=color, 
               linewidth=2, linestyle='-', theta1=180, theta2=360)
    ax.add_patch(a_xy)

    # xz-plane edge
    ax.add_patch(Ellipse((x_c, y_c), w, l*2, facecolor="None", edgecolor=color, linewidth=2))

    ax.add_line(plot.Line2D((x_c, x_c+r), (y_c, y_c), ls='--', color=color))
    ax.add_line(plot.Line2D((x_c, x_c), (y_c, y_c+l), ls='--', color=color))
    ax.add_line(plot.Line2D((x_c, x_c - w_yz*0.5), (y_c, y_c-l*0.5), ls='--', color=color))
    
    # labels
    ax.annotate(r_label, (x_c + r/2., y_c), color=color,  xytext=(0, 6), textcoords='offset points')
    ax.annotate(r_label, (x_c - w_yz*0.25, y_c-0.25*l), color=color,  xytext=(0, 10), textcoords='offset points')
    ax.annotate(h_label, (x_c, y_c + l/2), color=color,  xytext=(4, 0), textcoords='offset points')


def plot_filamentarity_reference(ax, plot_spheroid=True, plot_cylinder=True):
    r_ = 100.
    m_ = 7
    m = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    lm = 2.**m
    r = np.array([r_,]*len(lm))

    if plot_spheroid:
        F, P = filamentarity_planarity(spheroid_minkowski2, r=r, lm=lm + 1.0e-10)
        l_sphere, = ax.plot(P, F, marker='o', linestyle='-', label='spheroid')

        for n, (x_, y_, m__) in enumerate(zip(P, F, m)):
            
            if m__ >= 0:
                s = "{:d}".format(2**m__)
            else:
                s = "1/{:d}".format(2**(-m__))
            if n == len(m)-1:
                s = r"$\lambda=$"+s
            ax.annotate(s, (x_, y_), color=l_sphere.get_color(), xytext=(3, 3), textcoords='offset points')

        plot_spheroid_diagram(ax, 0.5, 0.5, l=0.11, r=0.18, color=l_sphere.get_color(), h_label=r"$\lambda r$")

    if plot_cylinder:
        F, P = filamentarity_planarity(cylinder_minkowski, r=r, lm=lm)
        l_cyl, = ax.plot(P, F, marker='o', linestyle='--', label='cylinder')

        for n, (x_, y_, m__) in enumerate(zip(P, F, m)):
            if m__ >= 0:
                s = "{:d}".format(2**m__)
            else:
                s = "1/{:d}".format(2**(-m__))
            if n == len(m)-1:
                s = r"$\lambda=$"+s
            ax.annotate(s, (x_, y_), color=l_cyl.get_color(), xytext=(3, 3), textcoords='offset points')

        plot_cylinder_diagram(ax, 0.4, 0.7, l=0.3, r=0.08, color=l_cyl.get_color(), h_label=r"$\lambda r$")

if __name__ == "__main__":
    plot.figure(figsize=(6,6))

    plot_filamentarity_reference(plot.gca())

    plot.xlabel("Planarity")
    plot.ylabel("Filamentarity")
    plot.legend()
    sns.despine()

    plot.savefig("planarity-filamentarity-spheroid-and-cylinder.pdf")
