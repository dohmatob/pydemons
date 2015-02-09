"""
How about log-doman diffeomorphic demons in pure python ;)
"""
# Author: DOHMATOB Elvis Dopgima

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def imagepad(im, scale=2.):
    """Pad input image to the left and right with zeros."""
    shape = np.array(im.shape)
    if scale <= 1.:
        return im, ((0, shape[0]), (0, shape[1]))
    new_shape = np.ceil(np.array(shape) * scale)
    before = np.floor((new_shape - shape) // 2)
    after = new_shape - shape - before
    pad_width = zip(before, after)
    lim = zip(before, before + shape)
    return np.pad(im, pad_width, mode="constant"), lim


def _standard_grid(shape):
    """Makes an (m)grid of regularly spaced points."""
    return np.mgrid[0.:shape[0], 0.:shape[1]]


def _deformed_grid(grid, sx, sy):
    """Makes a deformed (m)grid."""
    xx, yy = grid
    return xx + sx, yy + sy


def iminterpolate(im, grid=None, sx=None, sy=None, new_grid=None):
    """Interpolate image (2D)."""
    # make grids for interpolation
    im = np.array(im, dtype=np.float)
    shape = im.shape
    if grid is None:
        grid = _standard_grid(shape)
    if new_grid is None:
        new_grid = _deformed_grid(grid, sx, sy)

    # interpolate
    return ndimage.map_coordinates(im, new_grid, order=1)


def compose(ax, ay, bx, by):
    """Compose two vector fields u = [ax, ay] and v = [bx, by].

    Returns
    -------
    vx, vy: ax composed with bx, and ay composed with by

    Notes
    -----
    The first field [ax, ay] gets modified in-place.
    """
    if not (ax.shape == ay.shape == bx.shape == by.shape):
        raise RuntimeError("Fields are shape-inconsistent!")

    # make grids for interpolation
    shape = ax.shape
    grid = _standard_grid(shape)
    grid_prime = _deformed_grid(grid, ax, ay)

    # interpolate bx and by at points x + ax and y + by resp., then
    # at the result to ax and ay resp.
    ax += iminterpolate(bx, grid=grid, new_grid=grid_prime)
    ay += iminterpolate(by, grid=grid, new_grid=grid_prime)
    return ax, ay


def expfield(vx, vy):
    """Exponentiate a vector field.

    Note that the result is a deformation phi with is the solution of the
    following flow equation:

    d phi(t, z) = v(phi(t, z), phi(0, t) = z
    --
    dt
    """
    vx = vx.copy()
    vy = vy.copy()
    normv2 = vx ** 2 + vy ** 2
    m = np.sqrt(np.max(normv2))
    n = int(max(1 + np.ceil(np.log2(m)), 0.))
    p = 2. ** n
    vx /= p
    vy /= p
    for _ in xrange(n):
        vx, vy = compose(vx, vy, vx, vy)
    return vx, vy


def jacobian(sx, sy):
    """Computes jacobian of given deformation phi = [sx, sy]."""
    gx_x, gx_y = np.gradient(sx)
    gy_x, gy_y = np.gradient(sy)
    gx_x += 1.
    gy_y += 1.
    return gx_x * gy_y - gy_x * gx_y


def energy(fixed, moving, sx, sy, sigma_i, sigma_x):
    """Compute energy of current configuration."""
    # intensity difference
    moving_prime = iminterpolate(moving, sx=sx, sy=sy)
    diff2 = (fixed - moving_prime) ** 2
    area = np.prod(moving.shape) * 1.

    # deformation gradient
    jac = jacobian(sx, sy)

    # two energy components
    e_sim = diff2.sum() / area
    e_reg = (jac ** 2).sum() / area

    # total energy
    return e_sim + ((1. * sigma_i / sigma_x) ** 2) * e_reg


def imgaussian(im, sigma):
    """Apply Gaussian filter to input image (im), in-place."""
    ndimage.gaussian_filter(im, sigma, mode="constant", output=im)
    return im


def findupdate(fixed, moving, vx, vy, sigma_i, sigma_x):
    """Find demons force update field."""
    # get deformation
    [sx, sy] = expfield(vx, vy)

    # interpolate updated moving image
    moving_prime = iminterpolate(moving, sx=sx, sy=sy)

    # image intensity difference
    diff = fixed - moving_prime

    # moving image gradient
    gx, gy = np.gradient(moving_prime)
    normg2 = gx ** 2 + gy ** 2  # squared norm of moving image gradient

    # update is (idiff / (||J|| ** 2 + (idiff / sigma_x) ** 3) J, where
    # idiff := sigma_i * fixed(x) - moving(x + s) and J = grad(moving(x + s))
    scale = diff / (normg2 + ((diff * sigma_i) / sigma_x) ** 2)
    scale[normg2 == 0.] = 0.
    scale[diff == 0.] = 0.
    gx *= scale
    gy *= scale

    # zero-out nonoverlapping zones
    anti_fixed = (fixed == 0.)
    anti_moving_prime = (moving_prime == 0.)
    gx[anti_fixed] = 0.
    gx[anti_moving_prime] = 0.
    gy[anti_fixed] = 0.
    gy[anti_moving_prime] = 0.

    return gx, gy


def register(fixed, moving, sigma_fluid=1., sigma_diffusion=1., sigma_i=1.,
             sigma_x=1., niter=250, vx=None, vy=None, stop_criterion=.01,
             imagepad_scale=1.2):
    """Register moving image to moving image via log-domains diffeo demons."""
    shape = fixed.shape
    # init velocity
    if vx is None:
        vx = np.zeros(shape, dtype=np.float)
    if vy is None:
        vy = np.zeros(shape, dtype=np.float)

    # pad images and velocity
    if not (fixed.shape == moving.shape == vx.shape == vy.shape):
        raise RuntimeError(
            "Images and velocity are shape-inconsistent!")
    fixed, (lim_x, lim_y) = imagepad(fixed, imagepad_scale)
    moving, _ = imagepad(moving, imagepad_scale)
    vx, _ = imagepad(vx, imagepad_scale)
    vy, _ = imagepad(vy, imagepad_scale)

    # main loop
    step = sigma_x
    e = np.inf
    e_min = e
    energies = []
    for k in xrange(niter):
        print "Iter %03i/%03i: energy=%g" % (k + 1, niter, e)

        # find and smooth demons force field update
        ux, uy = findupdate(fixed, moving, vx, vy, sigma_i, sigma_x)
        ux = imgaussian(ux, sigma_fluid)
        uy = imgaussian(uy, sigma_fluid)

        # update velocity (additive demons) and then smooth
        vx += step * ux
        vy += step * uy
        vx = imgaussian(vx, sigma_diffusion)
        vy = imgaussian(vy, sigma_diffusion)

        # get deformation
        sx, sy = expfield(vx, vy)

        # compute energy
        e = energy(fixed, moving, sx, sy, sigma_i, sigma_x)
        energies.append(e)
        if e < e_min:
            e_min = e
            vx_min = vx
            vy_min = vy
            sx_min = sx
            sy_min = sy

        # check convergence
        if k > 0 and abs(e - energies[max(0, k - 5)]) < \
           energies[0] * stop_criterion:
            print "Converged!"
            break

    # apply optimal deformation to moving image
    moving_prime = iminterpolate(moving, sx=sx, sy=sy)

    # unpad velocity, deformation, and deformed moving image
    vx_min = vx_min[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    vy_min = vy_min[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    sx_min = sx_min[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    sy_min = sy_min[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    moving_prime = moving_prime[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    if not (fixed.shape == moving.shape == vx.shape == vy.shape):
        raise RuntimeError(
            "Images and velocity are shape-inconsistent!")

    return moving_prime, sx_min, sy_min, vx_min, vy_min, energies


def imresize(im, scale):
    """Resize image by given scale."""
    return ndimage.zoom(im, scale, order=1, prefilter=True)


def demons(fixed, moving, nlevel=3, sigma_fluid=1., sigma_diffusion=1.,
           sigma_i=1., sigma_x=1., niter=250, vx=None, vy=None,
           stop_criterion=.01, imagepad_scale=1.2):
    """Multi-resolution demons algorithm."""
    # init velocity
    if vx is None:
        vx = np.zeros(fixed.shape, dtype=np.float)
    if vy is None:
        vy = np.zeros(fixed.shape, dtype=np.float)

    # multi-resolution loop
    shape = np.array(fixed.shape, dtype=np.float)
    scale = 1. / 2 ** (nlevel - 1)
    for _ in xrange(nlevel):
        # downsample
        fixed_ = imresize(fixed, scale)
        moving_ = imresize(moving, scale)
        vx_ = imresize(vx * scale, scale)
        vy_ = imresize(vy * scale, scale)

        # register
        _, _, _, vx_, vy_, _ = register(
            fixed_, moving_, sigma_fluid=sigma_fluid,
            sigma_diffusion=sigma_diffusion, sigma_i=sigma_i, sigma_x=sigma_x,
            niter=niter, vx=vx_, vy=vy_, stop_criterion=stop_criterion,
            imagepad_scale=imagepad_scale)

        # upsample
        vx = imresize(vx_ / scale, shape / np.array(vx_.shape))
        vy = imresize(vy_ / scale, shape / np.array(vy_.shape))
        scale *= 2.

    sx, sy = expfield(vx, vy)
    return sx, sy, vx, vy


def test_iminterpolate():
    im = np.array([[1., 4., 5.], [0., 2., 1.], [-1., 0., 10.]])
    sx = np.array([[.5, .5, -2.], [0., 0., 1.], [2., 0., .5]])
    sy = np.array([[0., 0., -.5], [0., .5, 0.], [0., 0., .1]])
    np.testing.assert_array_equal(iminterpolate(im, sx=sx, sy=sy),
                                  [[.5, 3., 0.], [0., 1.5, 10.],
                                   [0., 0., 0.]])


def test_compose():
    ax = np.array([[.5, .5, -2.], [0., 0., 1.], [2., 0., .5]])
    ay = np.array([[0., 0., -.5], [0., .5, 0.], [0., 0., .1]])
    bx = .5 * ax
    by = .5 * ay
    vx, vy = compose(ax, ay, bx, by)
    np.testing.assert_array_equal(vx, ax)
    np.testing.assert_array_equal(vy, ay)
    np.testing.assert_array_equal(vx, [[.625, .625, -2.], [0., .25, 1.25],
                                       [2., 0., .5]])
    np.testing.assert_array_equal(vy, [[0., .125, -.5], [0., .625, .05],
                                       [0., 0., .1]])


def test_expfield():
    vx = np.array([[1., 2.], [3., 4.]])
    vy = np.array([[4., 3.], [2., 1.]])
    ex, ey = expfield(vx, vy)
    np.testing.assert_array_almost_equal(ex, [[.4824, .125], [.1875, .25]],
                                         decimal=4)
    np.testing.assert_array_almost_equal(ey, [[1.0254, .1875], [.125, .0625]],
                                         decimal=4)


def test_jacobian():
    vx = np.array([[1., 2.], [3., 4.]])
    vy = np.array([[4., 3.], [2., 1.]])
    np.testing.assert_array_equal(jacobian(vx, vy), 2)
    np.testing.assert_array_equal(jacobian(vy, vy), -2.)
    np.testing.assert_array_equal(jacobian(vx, vx), 4.)
    np.testing.assert_array_equal(jacobian(vx, vy * 0.), 3.)
    np.testing.assert_array_equal(jacobian(vx * 0., vy), 0.)
    np.testing.assert_array_equal(jacobian(vy, vy * 0.), -1.)


def test_imagepad():
    # test that scale <= 1 means no scaling
    im = np.random.randn(3, 4)
    for scale in [0., .1, .5, .6, 1.]:
        np.testing.assert_array_equal(im, imagepad(im, scale=scale)[0])

    # test scale=2 is default
    im = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(imagepad(im, 2.)[0], imagepad(im)[0])
    np.testing.assert_array_equal(imagepad(im, 2.)[1], imagepad(im)[1])

    # misc tests
    np.testing.assert_array_equal(imagepad(im, 1.2)[0], [[1, 2, 0], [3, 4, 0],
                                                         [0, 0, 0]])
    np.testing.assert_array_equal(imagepad(im, 2.)[0],
                                  [[0, 0, 0, 0], [0, 1, 2, 0],
                                   [0, 3, 4, 0], [0, 0, 0, 0]])


def test_energy():
    im = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_almost_equal(energy(im, im, im, im, 1., 1.), 25.5)
    np.testing.assert_almost_equal(energy(im, im, im, im, 2., 3.), 16.6111111)


def test_imgaussian():
    im = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_almost_equal(imgaussian(im, 1.), [[.876, .977],
                                                              [1.077, 1.178]],
                                         decimal=3)


def test_findupdate():
    fixed = np.array([[1., 2.], [3., 4.]])
    moving = fixed - 1.
    vx = np.eye(2)
    vy = -.5 * vx
    ux, uy = findupdate(fixed, moving, vx, vy, 1, 1)
    np.testing.assert_array_almost_equal(ux, [[0., -.333333], [.222222, 0.]])
    np.testing.assert_array_almost_equal(uy, [[0., .333333], [-.222222, 0.]])


if __name__ == "__main__":
    # load data
    from PIL import Image
    fixed = Image.open(
        "/home/elvis/CODE/FORKED/demons/demons/demons2d/data/lenag2.png")
    fixed = np.array(fixed, dtype=np.float)
    moving = Image.open(
        "/home/elvis/CODE/FORKED/demons/demons/demons2d/data/lenag1.png")
    moving = np.array(moving, dtype=np.float)

    # plot input images
    plt.figure()
    plt.gray()
    ax = plt.subplot(241)
    ax.set_title("fixed")
    plt.axis("off")
    ax.imshow(fixed)
    ax = plt.subplot(242)
    ax.set_title("moving")
    plt.axis("off")
    ax.imshow(moving)

    # run demons (log-domains diffeo)
    sx, sy, vx, vy = demons(fixed, moving, )
    warped = iminterpolate(moving, sx=sx, sy=sy)

    # plot warped image and difference to fixed image
    diff = warped - fixed
    ax = plt.subplot(243)
    ax.set_title("warped")
    ax.axis("off")
    warped_thumb = ax.imshow(warped)
    ax = plt.subplot(244)
    ax.set_title("diff")
    ax.axis("off")
    ax.imshow(diff)

    plt.show()
