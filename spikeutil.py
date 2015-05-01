#!/usr/bin/env python3

"""Functions to generate spikes. These are mostly utility functions to
convert rates to spikes. For instance, the function rate_to_poission
will generate Poissoin spikes based on a matrix of rates and several
other input parameters."""

from numpy import exp, power, random, floor, ceil, zeros, linspace


def gaussian(x, mu, sig):
    """Gaussian function without normalizing term. This yields a maximum
    value at 1, although the integral of the function is obviously not 1
    anymore."""
    return exp(-.5 * power((x - mu) / sig, 2.0))


def rate_to_poisson(R, dt=0.001, spikes_per_s=65, alpha=1.0, beta=0.0):
    """Convert the instantaneous firing rate given in matrix R into
    poisson spikes. The default values will generate poisson spikes with
    a frequency of 65 Hz for a time integration window of 0.001 s. Note
    that the range of values in R needs to be [0.0, 1.0], where 1.0
    indicates maximal firing rate."""
    # alpha = 1.0
    # you can add a baseline activity to the neural response by increasing beta.
    # beta = 0.0
    #beta  = 0.001
    spikes = (beta * random.random(R.shape) + alpha * spikes_per_s * R * dt) > random.random(R.shape)
    return spikes


def random_spike_train(tmax, dt=0.001, freq=25):
    """Generate a random spike train with some spike probability. tmax and dt in
    seconds, frequency in Hertz"""

    p_per_s = (1 / freq)
    n = ceil(tmax / dt)
    # convert to integer here to get rid of floating point issues
    spiketimes = floor(random.rand(n * (freq * dt)) * n).astype(int)
    return sorted(spiketimes)


def gen_homogeneous_response(v, vrange=[-1.0, 1.0], N=100):
    """generate a population response for neurons with a gaussian
    tuning curve that is determined by the number of neurons in the
    network.  The tuning curve of each neuron is the same. In addition,
    the network is assumed to be circular, thus mapping values which are
    outside the range to the opposing side. This is helpful, i.e. if you
    want to simulate head direction cells."""

    zro = (vrange[1] - vrange[0]) / 2
    vrange[0] -= zro
    vrange[1] -= zro
    rnge = abs(vrange[0]) + abs(vrange[1])

    v = 2 * v / rnge
    lspace = linspace(-1.0, 1.0, N) - v
    while (lspace[lspace < -1.0].size): lspace[lspace < -1.0] += 2.0
    while (lspace[lspace >  1.0].size): lspace[lspace >  1.0] -= 2.0

    # tuning curve specification
    #response = gaussian(lspace, 0, N/2500)
    #response = gaussian(lspace, 0, 1/(0.05*N))
    response = gaussian(lspace, 0, 1/(0.1*N))
    #response = gaussian(lspace, 0, 1/N**2)
    #response = gaussian(lspace, 0, 1/log(N))
    #response = gaussian(lspace, 0, 1/(log(N)**2))

    return response
