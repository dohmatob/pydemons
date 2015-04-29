"""
How about log-domain diffeomorphic demons in Python ? ;)
"""
# Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

import numpy as np
from scipy import ndimage

# 'texture' of floats in machine-precision
EPS = np.finfo(float).eps


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
    """Interpolate image (2D).

    Interpolation is by far the most costly operation in the whole
    algorithm.
    """
    # make grids for interpolation
    im = np.array(im, dtype=np.float)
    shape = im.shape
    if grid is None:
        grid = _standard_grid(shape)
    if new_grid is None:
        new_grid = _deformed_grid(grid, sx, sy)

    # interpolate
    return ndimage.map_coordinates(im, new_grid, order=1, mode="constant")


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


def jacobian(sx, sy, add_identity=True):
    """Computes jacobian of given deformation phi = [sx, sy]."""
    gx_x, gx_y = np.gradient(sx)
    gy_x, gy_y = np.gradient(sy)
    if add_identity:
        gx_x += 1.
        gy_y += 1.
    return gx_x * gy_y - gy_x * gx_y


def energy(fixed, moving, sx, sy, sigma_i, sigma_x):
    """Compute energy of current configuration."""
    # intensity difference
    warped = iminterpolate(moving, sx=sx, sy=sy)
    diff2 = (fixed - warped) ** 2
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
    warped = iminterpolate(moving, sx=sx, sy=sy)

    # image intensity difference
    diff = fixed - warped

    # moving image gradient
    gx, gy = np.gradient(warped)
    normg2 = gx ** 2 + gy ** 2  # squared norm of moving image gradient

    # update is (idiff / (||J|| ** 2 + (idiff / sigma_x) ** 3) J, where
    # idiff := sigma_i * fixed(x) - moving(x + s) and J = grad(moving(x + s))
    scale = diff / (EPS + normg2 + ((diff * sigma_i) / sigma_x) ** 2)
    scale[normg2 == 0.] = 0.
    scale[diff == 0.] = 0.
    gx *= scale
    gy *= scale

    # zero-out non-overlapping zones
    anti_fixed = (fixed == 0.)
    anti_warped = (warped == 0.)
    gx[anti_fixed] = 0.
    gx[anti_warped] = 0.
    gy[anti_fixed] = 0.
    gy[anti_warped] = 0.

    return gx, gy


def imagecrop(im, lim_x, lim_y):
    """Crop image to fit given bounding box."""
    return im[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]


def bch(vx, vy, ux, uy, bch_order=0):
    """Backer-Campbell-Hausdorf approximation of log(exp(v) o exp(u)).

    Note that
    log(exp(v) o exp(u)) = v + u + 1/2 [v, u] + 1/12 [v, [v, u]] + H.O.T

    Paremeters
    ----------
    bch_order: int, which is 0 or 1 (optional, default 0)
        The order of the BCH approximation. bch_order=0 corresponds to a
        commutativity assumption (for speed!).
    """
    if not bch_order in [0, 1]:
        raise ValueError(
            "BCH: `bch_order` must be 0 or 1. Got %i." % bch_order)
    if bch_order == 0:
        # commutativity assumption
        vx += ux
        vy += uy
    else:
        # expand upto linear order. Not that [v, u] = jac(v)u - jac(u)v
        jacu = jacobian(ux, uy, add_identity=False)
        jacv = jacobian(vx, vy, add_identity=False)
        vx += ux + .5 * (jacv * ux - jacu * vx)
        vy += uy + .5 * (jacv * uy - jacu * vy)
    return vx, vy


def register(fixed, moving, symmetric=True, sigma_fluid=1., sigma_diffusion=1.,
             sigma_i=1., sigma_x=1., niter=250, vx=None, vy=None, bch_order=0,
             stop_criterion=.01, imagepad_scale=1.2, callback=None):
    """Register moving image to moving image via log-domains diffeo demons.

    Parameters
    ----------
    symmetric: bool (optional, default True)
        If True, then symmetrized energy will be used.
    """
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
    orig_fixed = fixed
    fixed, (lim_x, lim_y) = imagepad(fixed, imagepad_scale)
    moving, _ = imagepad(moving, imagepad_scale)
    vx, _ = imagepad(vx, imagepad_scale)
    vy, _ = imagepad(vy, imagepad_scale)

    # main loop
    e = np.inf
    e_min = e
    energies = []
    for k in xrange(niter):
        # find demons force field update and then smooth
        ux_forw, uy_forw = findupdate(fixed, moving, vx, vy, sigma_i, sigma_x)
        if not symmetric:
            ux, uy = ux_forw, uy_forw
        else:
            # symmetric regime: compute backward force field, then average
            ux_back, uy_back = findupdate(moving, fixed, -vx, -vy, sigma_i,
                                          sigma_x)
            ux, uy = .5 * (ux_forw - ux_back), .5 * (uy_forw - uy_back)
        ux = imgaussian(ux, sigma_fluid)
        uy = imgaussian(uy, sigma_fluid)

        # update velocity (= log(exp(v) o exp(u))) and then smooth
        vx, vy = bch(vx, vy, ux, uy, bch_order=bch_order)
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

        # invoke callback
        print "Iter %03i/%03i: energy=%g" % (k + 1, niter, e)
        if callback and k % 10 == 0:
            variables = locals()
            variables["fixed"] = orig_fixed
            callback(variables)

        # check convergence
        if k > 0 and abs(e - energies[max(0, k - 5)]) < \
           energies[0] * stop_criterion:
            print "Converged!"
            break

    # apply optimal deformation to moving image
    warped = iminterpolate(moving, sx=sx, sy=sy)

    # unpad velocity, deformation, and deformed moving image
    vx_min = imagecrop(vx_min, lim_x, lim_y)
    vy_min = imagecrop(vy_min, lim_x, lim_y)
    sx_min = imagecrop(sx_min, lim_x, lim_y)
    sy_min = imagecrop(sy_min, lim_x, lim_y)
    warped = warped[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
    if not (fixed.shape == moving.shape == vx.shape == vy.shape):
        raise RuntimeError(
            "Images and velocity are shape-inconsistent!")

    return warped, sx_min, sy_min, vx_min, vy_min, energies


def imresize(im, scale):
    """Resize image by given scale."""
    return ndimage.zoom(im, scale, order=1, prefilter=True)


def demons(fixed, moving, nlevel=3, symmetric=True, sigma_fluid=1.,
           sigma_diffusion=1., sigma_i=1., sigma_x=1., niter=250, vx=None,
           vy=None, bch_order=0, stop_criterion=.01, imagepad_scale=1.2,
           callback=None):
    """Multi-resolution log-domain diffeomorphic demons algorithm.

    Parameters
    ----------
    symmetric: bool (optional, default True)
        If True, then symmetrized energy will be used.
    """
    # init velocity
    if vx is None:
        vx = np.zeros(fixed.shape, dtype=np.float)
    if vy is None:
        vy = np.zeros(fixed.shape, dtype=np.float)

    # multi-resolution loop in decreasing powers of 2
    shape = np.array(fixed.shape, dtype=np.float)
    scale = 1. / 2 ** (nlevel - 1)
    for _ in xrange(nlevel):
        # downsample images and velocities
        fixed_ = imresize(fixed, scale)
        moving_ = imresize(moving, scale)
        vx_ = imresize(vx * scale, scale)
        vy_ = imresize(vy * scale, scale)

        # register
        _, _, _, vx_, vy_, _ = register(
            fixed_, moving_, symmetric=symmetric, sigma_fluid=sigma_fluid,
            sigma_diffusion=sigma_diffusion, sigma_i=sigma_i, sigma_x=sigma_x,
            niter=niter, vx=vx_, vy=vy_, bch_order=bch_order,
            stop_criterion=stop_criterion, imagepad_scale=imagepad_scale,
            callback=callback)

        # upsample
        vx = imresize(vx_ / scale, shape / np.array(vx_.shape))
        vy = imresize(vy_ / scale, shape / np.array(vy_.shape))
        scale *= 2.

    sx, sy = expfield(vx, vy)
    return sx, sy, vx, vy
