import numpy as np
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.viz import actor, window
import vtk

n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)

interactive = True

ren = window.Renderer()
ren.SetBackground(1, 1, 1)

ren.add(actor.point(hsph_initial.vertices, window.colors.red, point_radius=.05))
ren.add(actor.point(hsph_updated.vertices, window.colors.green, point_radius=.05))

if interactive:
    window.show(ren)
