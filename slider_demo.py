"""
======
Slider
======

In this example, sliders are used to control the frequency and amplitude of
a sine wave.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.

See :doc:`/gallery/widgets/range_slider` for an example of using
a ``RangeSlider`` to define a range of values.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def f(X, Y, fx, fy):
    return np.cos(2 * np.pi * fx * X)**2 + np.sin(2 * np.pi * fy * Y)**2

def g(X,Y,fx,fy):
    return X**fx + Y**fy
n = 51
X, Y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n), indexing = 'ij')
# Define initial parameters

fx0 = 1.0
fy0 = 1.0
#global cf
#global cg
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(1, 2, sharey = True)
cf = ax[0].contourf(X, Y, f(X, Y, fx0,fy0))
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title(r'$f(x,y) = \cos(2\pi f_x x)^2 + \sin(2\pi f_y y)^2$')

cg = ax[1].contourf(X, Y, g(X, Y, fx0,fy0))
ax[1].set_xlabel('x')
ax[1].set_title(r'$g(x,y) = x^{f_x} + y^{f_y}$')



axcolor = 'lightgoldenrodyellow'
ax[0].margins(x=0)
ax[1].margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
fx_slider = Slider(
    ax = axfx,
    label = 'f_x',
    valmin = 0,
    valmax = 5,
    valinit = fx0,
    valstep = .5
)

# Make a vertically oriented slider to control the amplitude
axfy = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
fy_slider = Slider(
    ax = axfy,
    label = 'f_y',
    valmin = 0,
    valmax = 5,
    valinit = fy0,
    orientation = "vertical",
    valstep = .5
)



# The function to be called anytime a slider's value changes
def update(val):
    #for c in cf.collections:
    #    c.remove()
    #for c in cg.collections:
    #    c.remove() 
    cf = ax[0].contourf(X, Y, f(X, Y, fx_slider.val, fy_slider.val))
    cg = ax[1].contourf(X, Y, g(X, Y, fx_slider.val, fy_slider.val))
    #line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
fx_slider.on_changed(update)
fy_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    fx_slider.reset()
    fy_slider.reset()
button.on_clicked(reset)

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

# import matplotlib
# matplotlib.widgets.Button
# matplotlib.widgets.Slider
