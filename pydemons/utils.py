from nose.tools import assert_equal
import numpy as np
from scipy import interpolate


def imagepad(im, scale=2.):
    if scale <= 1.:
        return im
    shape = np.array(im.shape)
    new_shape = np.ceil(np.array(shape) * scale)
    before = np.floor((new_shape - shape) // 2)
    after = new_shape - shape - before
    return np.pad(im, zip(before, after), mode="constant")


def _standard_grid(shape):
    xx, yy = np.mgrid[0.:shape[0], 0.:shape[1]]
    return np.array(zip(xx.ravel(), yy.ravel()))


def _transformed_grid(grid, sx, sy):
    new_grid = grid.copy()
    new_grid[:, 0] += sx
    new_grid[:, 1] += sy
    return new_grid


def iminterpolate(im, grid=None, sx=None, sy=None, new_grid=None,
                  fill_value=0., **kwargs):
    """Interpolate image (2D)."""
    # make grids for interpolation
    im = np.array(im, dtype=np.float)
    shape = im.shape
    if grid is None:
        grid = _standard_grid(shape)
    im = im.ravel()
    if new_grid is None:
        sx, sy = sx.ravel(), sy.ravel()
        new_grid = _transformed_grid(grid, sx, sy)
    else:
        new_grid = np.array(new_grid, dtype=np.float)

    # interpolate
    return interpolate.griddata(grid, im, new_grid, fill_value=fill_value,
                                **kwargs).reshape(shape)


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
    grid_prime = _transformed_grid(grid, ax.ravel(), ay.ravel())

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
    movingp = iminterpolate(moving, sx=sx, sy=sy)
    diff2 = (fixed - movingp) ** 2
    area = np.prod(moving.shape) * 1.

    # deformation gradient
    jac = jacobian(sx, sy)

    # two energy components
    e_sim = diff2.sum() / area
    e_reg = (jac ** 2).sum() / area

    # total energy
    return e_sim + ((1. * sigma_i / sigma_x) ** 2) * e_reg


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
    np.testing.assert_array_almost_equal(ex, [[.5037, .125], [.1875, .25]],
                                         decimal=4)
    np.testing.assert_array_almost_equal(ey, [[1.07, .1875], [.125, .0625]],
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
        np.testing.assert_array_equal(im, imagepad(im, scale=scale))

    # test scale=2 is default
    im = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(imagepad(im, 2.), imagepad(im))

    # misc tests
    np.testing.assert_array_equal(imagepad(im, 1.2), [[1, 2, 0], [3, 4, 0],
                                                      [0, 0, 0]])
    np.testing.assert_array_equal(imagepad(im, 2.),
                                  [[0, 0, 0, 0], [0, 1, 2, 0],
                                   [0, 3, 4, 0], [0, 0, 0, 0]])


def test_energy():
    im = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_almost_equal(energy(im, im, im, im, 1., 1.), 25.5)
    np.testing.assert_almost_equal(energy(im, im, im, im, 2., 3.), 16.6111111)
