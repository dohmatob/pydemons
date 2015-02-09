import numpy as np
from ..demons import (iminterpolate, compose, expfield, jacobian,
                      imagepad, energy, imgaussian, findupdate)


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
