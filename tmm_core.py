# -*- coding: utf-8 -*-
"""
For information see the docstring of each function, and also see
manual.pdf (should be included with the distribution, otherwise get it
at https://github.com/sbyrnes321/tmm/blob/master/manual.pdf ). Physics background,
conventions, and derivations are at https://arxiv.org/abs/1603.02720

The most two important functions are:

coh_tmm(...) -- the transfer-matrix-method calculation in the coherent
case (i.e. thin films)

inc_tmm(...) -- the transfer-matrix-method calculation in the incoherent
case (i.e. films tens or hundreds of wavelengths thick, or whose
thickness is not very uniform.)

These functions are all imported into the main package (tmm) namespace,
so you can call them with tmm.coh_tmm(...) etc.
"""

from __future__ import division, print_function, absolute_import

from numpy import cos, inf, zeros, array, exp, conj, nan, isnan, pi, sin, seterr
from numpy.lib.scimath import arcsin

import numpy as np

import sys
EPSILON = sys.float_info.epsilon # typical floating-point calculation error

def make_2x2_array(a, b, c, d, dtype=float):

    my_array = np.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

def is_forward_angle(n, theta):

    assert n.real * n.imag >= 0, ("n: " + str(n) + "   angle: " + str(theta))
    ncostheta = n * cos(theta)
    if abs(ncostheta.imag) > 100 * EPSILON:
        answer = (ncostheta.imag > 0)
    else:
        answer = (ncostheta.real > 0)
    answer = bool(answer)
    error_string = ("n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ncostheta.real > -100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ncostheta.real < 100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real < 100 * EPSILON, error_string
    return answer

def snell(n_1, n_2, th_1):
    th_2_guess = arcsin(n_1*np.sin(th_1) / n_2)
    if is_forward_angle(n_2, th_2_guess):
        return th_2_guess
    else:
        return pi - th_2_guess

def list_snell(n_list, th_0):
    angles = arcsin(n_list[0]*np.sin(th_0) / n_list)
    if not is_forward_angle(n_list[0], angles[0]):
        angles[0] = pi - angles[0]
    if not is_forward_angle(n_list[-1], angles[-1]):
        angles[-1] = pi - angles[-1]
    return angles


def interface_r(polarization, n_i, n_f, th_i, th_f):
    if polarization == 's':
        return ((n_i * cos(th_i) - n_f * cos(th_f)) /
                (n_i * cos(th_i) + n_f * cos(th_f)))
    elif polarization == 'p':
        return ((n_f * cos(th_i) - n_i * cos(th_f)) /
                (n_f * cos(th_i) + n_i * cos(th_f)))
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_t(polarization, n_i, n_f, th_i, th_f):
    if polarization == 's':
        return 2 * n_i * cos(th_i) / (n_i * cos(th_i) + n_f * cos(th_f))
    elif polarization == 'p':
        return 2 * n_i * cos(th_i) / (n_f * cos(th_i) + n_i * cos(th_f))
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def R_from_r(r):
    return abs(r)**2

def T_from_t(pol, t, n_i, n_f, th_i, th_f):
    if pol == 's':
        return abs(t**2) * (((n_f*cos(th_f)).real) / (n_i*cos(th_i)).real)
    elif pol == 'p':
        return abs(t**2) * (((n_f*conj(cos(th_f))).real) /
                                (n_i*conj(cos(th_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def power_entering_from_r(pol, r, n_i, th_i):
    if pol == 's':
        return ((n_i*cos(th_i)*(1+conj(r))*(1-r)).real
                     / (n_i*cos(th_i)).real)
    elif pol == 'p':
        return ((n_i*conj(cos(th_i))*(1+r)*(1-conj(r))).real
                      / (n_i*conj(cos(th_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_R(polarization, n_i, n_f, th_i, th_f):
    r = interface_r(polarization, n_i, n_f, th_i, th_f)
    return R_from_r(r)

def interface_T(polarization, n_i, n_f, th_i, th_f):
    t = interface_t(polarization, n_i, n_f, th_i, th_f)
    return T_from_t(polarization, t, n_i, n_f, th_i, th_f)

def coh_tmm(pol, n_list, d_list, th_0, lam_vac):

    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
          or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('calculation at a time ')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n d")
    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'
    assert abs((n_list[0]*np.sin(th_0)).imag) < 100*EPSILON, 'Error in n0 or th0!'
    assert is_forward_angle(n_list[0], th_0), 'Error in n0 or th0!'
    num_layers = n_list.size

    th_list = list_snell(n_list, th_0)

    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac


    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    for i in range(1, num_layers-1):
        if delta[i].imag > 35:
            delta[i] = delta[i].real + 35j
            if 'opacity_warning' not in globals():
                global opacity_warning
                opacity_warning = True
                print("Warning")

    t_list = zeros((num_layers, num_layers), dtype=complex)
    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers-1):
        t_list[i,i+1] = interface_t(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        r_list[i,i+1] = interface_r(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])

    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers-1):
        M_list[i] = (1/t_list[i,i+1]) * np.dot(
            make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers-1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Mtilde)


    r = Mtilde[1,0]/Mtilde[0,0]
    t = 1/Mtilde[0,0]


    vw_list = zeros((num_layers, 2), dtype=complex)
    vw = array([[t],[0]])
    vw_list[-1,:] = np.transpose(vw)
    for i in range(num_layers-2, 0, -1):
        vw = np.dot(M_list[i], vw)
        vw_list[i,:] = np.transpose(vw)


    R = R_from_r(r)
    T = T_from_t(pol, t, n_list[0], n_list[-1], th_0, th_list[-1])
    power_entering = power_entering_from_r(pol, r, n_list[0], th_0)

    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'vw_list': vw_list, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th_0': th_0,
            'lam_vac':lam_vac}


def coh_tmm_dispersion(pol, dispersion_list, d_list, th_0, lam_vac):
    n_list = np.array([dispersion(lam_vac) for dispersion in dispersion_list])
    d_list = np.array(d_list, dtype=float)

    if ((hasattr(lam_vac, 'size') and lam_vac.size > 1)
          or (hasattr(th_0, 'size') and th_0.size > 1)):
        raise ValueError('calculation at a time ')
    if (n_list.ndim != 1) or (d_list.ndim != 1) or (n_list.size != d_list.size):
        raise ValueError("Problem with n d")
    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'
    assert abs((n_list[0]*np.sin(th_0)).imag) < 100*EPSILON, 'Error in n0 or th0!'
    assert is_forward_angle(n_list[0], th_0), 'Error in n0 or th0!'
    num_layers = n_list.size

    th_list = list_snell(n_list, th_0)

    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac

    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)

    for i in range(1, num_layers-1):
        if delta[i].imag > 35:
            delta[i] = delta[i].real + 35j
            if 'opacity_warning' not in globals():
                global opacity_warning
                opacity_warning = True
                print("Warning")

    t_list = zeros((num_layers, num_layers), dtype=complex)
    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers-1):
        t_list[i,i+1] = interface_t(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        r_list[i,i+1] = interface_r(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])

    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers-1):
        M_list[i] = (1/t_list[i,i+1]) * np.dot(
            make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers-1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Mtilde)

    r = Mtilde[1,0]/Mtilde[0,0]
    t = 1/Mtilde[0,0]

    vw_list = zeros((num_layers, 2), dtype=complex)
    vw = array([[t],[0]])
    vw_list[-1,:] = np.transpose(vw)
    for i in range(num_layers-2, 0, -1):
        vw = np.dot(M_list[i], vw)
        vw_list[i,:] = np.transpose(vw)

    R = R_from_r(r)
    T = T_from_t(pol, t, n_list[0], n_list[-1], th_0, th_list[-1])
    power_entering = power_entering_from_r(pol, r, n_list[0], th_0)

    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'vw_list': vw_list, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th_0': th_0,
            'lam_vac': lam_vac}


def sellmeier_eq(wavelength, B1, B2, B3, C1, C2, C3):
    wavelength_um = wavelength * 1e6  # Convert wavelength to micrometers
    n_squared = 1 + (B1 * wavelength_um**2) / (wavelength_um**2 - C1) + \
                   (B2 * wavelength_um**2) / (wavelength_um**2 - C2) + \
                   (B3 * wavelength_um**2) / (wavelength_um**2 - C3)
    return np.sqrt(n_squared)

def n_sio2(wavelength):
    return sellmeier_eq(wavelength, 0.6961663, 0.4079426, 0.8974794, 0.0684043, 0.1162414, 9.896161)

def n_tio2(wavelength):
    # Example Sellmeier coefficients for TiO2 (these values may vary)
    return sellmeier_eq(wavelength, 5.913, 0.2441, 0, 0, 0.0803, 0)








def coh_tmm_reverse(pol, n_list, d_list, th_0, lam_vac):

    th_f = snell(n_list[0], n_list[-1], th_0)
    return coh_tmm(pol, n_list[::-1], d_list[::-1], th_f, lam_vac)

def ellips(n_list, d_list, th_0, lam_vac):
    s_data = coh_tmm('s', n_list, d_list, th_0, lam_vac)
    p_data = coh_tmm('p', n_list, d_list, th_0, lam_vac)
    rs = s_data['r']
    rp = p_data['r']
    return {'psi': np.arctan(abs(rp/rs)), 'Delta': np.angle(-rp/rs)}

def unpolarized_RT(n_list, d_list, th_0, lam_vac):

    s_data = coh_tmm('s', n_list, d_list, th_0, lam_vac)
    p_data = coh_tmm('p', n_list, d_list, th_0, lam_vac)
    R = (s_data['R'] + p_data['R']) / 2.
    T = (s_data['T'] + p_data['T']) / 2.
    return {'R': R, 'T': T}

def position_resolved(layer, distance, coh_tmm_data):
    if layer > 0:
        v,w = coh_tmm_data['vw_list'][layer]
    else:
        v = 1
        w = coh_tmm_data['r']
    kz = coh_tmm_data['kz_list'][layer]
    th = coh_tmm_data['th_list'][layer]
    n = coh_tmm_data['n_list'][layer]
    n_0 = coh_tmm_data['n_list'][0]
    th_0 = coh_tmm_data['th_0']
    pol = coh_tmm_data['pol']

    assert ((layer >= 1 and 0 <= distance <= coh_tmm_data['d_list'][layer])
                or (layer == 0 and distance <= 0))

    Ef = v * exp(1j * kz * distance)
    Eb = w * exp(-1j * kz * distance)

    if pol == 's':
        poyn = ((n*cos(th)*conj(Ef+Eb)*(Ef-Eb)).real) / (n_0*cos(th_0)).real
    elif pol == 'p':
        poyn = (((n*conj(cos(th))*(Ef+Eb)*conj(Ef-Eb)).real)
                    / (n_0*conj(cos(th_0))).real)

    if pol == 's':
        absor = (n*cos(th)*kz*abs(Ef+Eb)**2).imag / (n_0*cos(th_0)).real
    elif pol == 'p':
        absor = (n*conj(cos(th))*
                 (kz*abs(Ef-Eb)**2-conj(kz)*abs(Ef+Eb)**2)
                ).imag / (n_0*conj(cos(th_0))).real

    if pol == 's':
        Ex = 0
        Ey = Ef + Eb
        Ez = 0
    elif pol == 'p':
        Ex = (Ef - Eb) * cos(th)
        Ey = 0
        Ez = (-Ef - Eb) * sin(th)

    return {'poyn': poyn, 'absor': absor, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez}

def find_in_structure(d_list, distance):
    if sum(d_list) == inf:
        raise ValueError('This function expects finite arguments')
    if distance < 0:
        return [-1, distance]
    layer = 0
    while (layer < len(d_list)) and (distance >= d_list[layer]):
        distance -= d_list[layer]
        layer += 1
    return [layer, distance]

def find_in_structure_with_inf(d_list, distance):
    if distance < 0:
        return [0, distance]
    [layer, distance_in_layer] = find_in_structure(d_list[1:-1], distance)
    return [layer+1, distance_in_layer]

def layer_starts(d_list):
    final_answer = zeros(len(d_list))
    final_answer[0] = -inf
    final_answer[1] = 0
    for i in range(2, len(d_list)):
        final_answer[i] = final_answer[i-1] + d_list[i-1]
    return final_answer

class absorp_analytic_fn:
    def fill_in(self, coh_tmm_data, layer):

        pol = coh_tmm_data['pol']
        v = coh_tmm_data['vw_list'][layer][0]
        w = coh_tmm_data['vw_list'][layer][1]
        kz = coh_tmm_data['kz_list'][layer]
        n = coh_tmm_data['n_list'][layer]
        n_0 = coh_tmm_data['n_list'][0]
        th_0 = coh_tmm_data['th_0']
        th = coh_tmm_data['th_list'][layer]
        self.d = coh_tmm_data['d_list'][layer]

        self.a1 = 2*kz.imag
        self.a3 = 2*kz.real

        if pol == 's':
            temp = (n*cos(th)*kz).imag / (n_0*cos(th_0)).real
            self.A1 = temp * abs(w)**2
            self.A2 = temp * abs(v)**2
            self.A3 = temp * v * conj(w)
        else: # pol=='p'
            temp = (2*(kz.imag)*(n*cos(conj(th))).real /
                    (n_0*conj(cos(th_0))).real)
            self.A1 = temp * abs(w)**2
            self.A2 = temp * abs(v)**2
            self.A3 = v * conj(w) * (-2*(kz.real)*(n*cos(conj(th))).imag /
                                     (n_0*conj(cos(th_0))).real)
        return self

    def copy(self):
        """
        Create copy of an absorp_analytic_fn object
        """
        a = absorp_analytic_fn()
        (a.A1, a.A2, a.A3, a.a1, a.a3, a.d) = (
           self.A1, self.A2, self.A3, self.a1, self.a3, self.d)
        return a

    def run(self, z):
        """
        Calculates absorption at a given depth z, where z=0 is the start of the
        layer.
        """
        return (self.A1*exp(self.a1 * z) + self.A2*exp(-self.a1 * z)
             + self.A3*exp(1j*self.a3*z) + conj(self.A3)*exp(-1j*self.a3*z))

    def flip(self):
        """
        Flip the function front-to-back, to describe a(d-z) instead of a(z),
        where d is layer thickness.
        """
        newA1 = self.A2*exp(-self.a1 * self.d)
        newA2 = self.A1*exp(self.a1 * self.d)
        self.A1, self.A2 = newA1, newA2
        self.A3 = conj(self.A3 * exp(1j * self.a3 * self.d))
        return self

    def scale(self, factor):
        """
        multiplies the absorption at each point by "factor".
        """
        self.A1 *= factor
        self.A2 *= factor
        self.A3 *= factor
        return self

    def add(self, b):
        """
        adds another compatible absorption analytical function
        """
        if (b.a1 != self.a1) or (b.a3 != self.a3):
            raise ValueError('Incompatible absorption analytical functions!')
        self.A1 += b.A1
        self.A2 += b.A2
        self.A3 += b.A3
        return self

def absorp_in_each_layer(coh_tmm_data):
    num_layers = len(coh_tmm_data['d_list'])
    power_entering_each_layer = zeros(num_layers)
    power_entering_each_layer[0] = 1
    power_entering_each_layer[1] = coh_tmm_data['power_entering']
    power_entering_each_layer[-1] = coh_tmm_data['T']
    for i in range(2, num_layers-1):
        power_entering_each_layer[i] = position_resolved(i, 0, coh_tmm_data)['poyn']
    final_answer = zeros(num_layers)
    final_answer[0:-1] = -np.diff(power_entering_each_layer)
    final_answer[-1] = power_entering_each_layer[-1]
    return final_answer

def inc_group_layers(n_list, d_list, c_list):

    if (n_list.ndim != 1) or (d_list.ndim != 1):
        raise ValueError("Problem with n_list or d_list!")
    if (d_list[0] != inf) or (d_list[-1] != inf):
        raise ValueError('d_list must start and end with inf!')
    if (c_list[0] != 'i') or (c_list[-1] != 'i'):
        raise ValueError('c_list should start and end with "i"')
    if not n_list.size == d_list.size == len(c_list):
        raise ValueError('List sizes do not match!')
    inc_index = 0
    stack_index = 0
    stack_d_list = []
    stack_n_list = []
    all_from_inc = []
    inc_from_all = []
    all_from_stack = []
    stack_from_all = []
    inc_from_stack = []
    stack_from_inc = []
    stack_in_progress = False
    for alllayer_index in range(n_list.size):
        if c_list[alllayer_index] == 'c': #coherent layer
            inc_from_all.append(nan)
            if not stack_in_progress: #this layer is starting new stack
                stack_in_progress = True
                ongoing_stack_d_list = [inf, d_list[alllayer_index]]
                ongoing_stack_n_list = [n_list[alllayer_index-1],
                                        n_list[alllayer_index]]
                stack_from_all.append([stack_index,1])
                all_from_stack.append([alllayer_index-1, alllayer_index])
                inc_from_stack.append(inc_index-1)
                within_stack_index = 1
            else: #another coherent layer in the same stack
                ongoing_stack_d_list.append(d_list[alllayer_index])
                ongoing_stack_n_list.append(n_list[alllayer_index])
                within_stack_index += 1
                stack_from_all.append([stack_index, within_stack_index])
                all_from_stack[-1].append(alllayer_index)
        elif c_list[alllayer_index] == 'i': #incoherent layer
            stack_from_all.append(nan)
            inc_from_all.append(inc_index)
            all_from_inc.append(alllayer_index)
            if not stack_in_progress: #previous layer was also incoherent
                stack_from_inc.append(nan)
            else: #previous layer was coherent
                stack_in_progress = False
                stack_from_inc.append(stack_index)
                ongoing_stack_d_list.append(inf)
                stack_d_list.append(ongoing_stack_d_list)
                ongoing_stack_n_list.append(n_list[alllayer_index])
                stack_n_list.append(ongoing_stack_n_list)
                all_from_stack[-1].append(alllayer_index)
                stack_index += 1
            inc_index += 1
        else:
            raise ValueError("Error: c_list entries must be 'i' or 'c'!")
    return {'stack_d_list':stack_d_list,
            'stack_n_list':stack_n_list,
            'all_from_inc':all_from_inc,
            'inc_from_all':inc_from_all,
            'all_from_stack':all_from_stack,
            'stack_from_all':stack_from_all,
            'inc_from_stack':inc_from_stack,
            'stack_from_inc':stack_from_inc,
            'num_stacks':len(all_from_stack),
            'num_inc_layers':len(all_from_inc),
            'num_layers':len(n_list)}

def inc_tmm(pol, n_list, d_list, c_list, th_0, lam_vac):
    # Convert lists to numpy arrays if they're not already.
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    # Input tests
    if (np.real_if_close(n_list[0]*np.sin(th_0))).imag != 0:
        raise ValueError('Error in n0 or th0!')

    group_layers_data = inc_group_layers(n_list, d_list, c_list)
    num_inc_layers = group_layers_data['num_inc_layers']
    num_stacks = group_layers_data['num_stacks']
    stack_n_list = group_layers_data['stack_n_list']
    stack_d_list = group_layers_data['stack_d_list']
    all_from_stack = group_layers_data['all_from_stack']
    all_from_inc = group_layers_data['all_from_inc']
    all_from_stack = group_layers_data['all_from_stack']
    stack_from_inc = group_layers_data['stack_from_inc']
    inc_from_stack = group_layers_data['inc_from_stack']

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # coh_tmm_data_list[i] is the output of coh_tmm for the i'th stack
    coh_tmm_data_list = []
    # coh_tmm_bdata_list[i] is the same stack as coh_tmm_data_list[i] but
    # with order of layers reversed
    coh_tmm_bdata_list = []
    for i in range(num_stacks):
        coh_tmm_data_list.append(coh_tmm(pol, stack_n_list[i],
                                         stack_d_list[i],
                                         th_list[all_from_stack[i][0]],
                                         lam_vac))
        coh_tmm_bdata_list.append(coh_tmm_reverse(pol, stack_n_list[i],
                                                  stack_d_list[i],
                                                  th_list[all_from_stack[i][0]],
                                                  lam_vac))

    # P_list[i] is fraction not absorbed in a single pass through i'th incoherent
    # layer.
    P_list = zeros(num_inc_layers)
    for inc_index in range(1,num_inc_layers-1): #skip 0'th and last (infinite)
        i = all_from_inc[inc_index]
        P_list[inc_index] = exp(-4 * np.pi * d_list[i]
                     * (n_list[i] * cos(th_list[i])).imag / lam_vac)
        # For a very opaque layer, reset P to avoid divide-by-0 and similar
        # errors.
        if P_list[inc_index] < 1e-30:
            P_list[inc_index] = 1e-30
    # T_list[i,j] and R_list[i,j] are transmission and reflection powers,
    # respectively, coming from the i'th incoherent layer, going to the j'th
    # incoherent layer. Only need to calculate this when j=i+1 or j=i-1.
    # (2D array is overkill but helps avoid confusion.)
    # initialize these arrays
    T_list = zeros((num_inc_layers, num_inc_layers))
    R_list = zeros((num_inc_layers, num_inc_layers))
    for inc_index in range(num_inc_layers-1): #looking at interface i -> i+1
        alllayer_index = all_from_inc[inc_index]
        nextstack_index = stack_from_inc[inc_index+1]
        if isnan(nextstack_index): #next layer is incoherent
            R_list[inc_index, inc_index+1] = (
                   interface_R(pol, n_list[alllayer_index],
                               n_list[alllayer_index+1],
                               th_list[alllayer_index],
                               th_list[alllayer_index+1]))
            T_list[inc_index, inc_index+1] = (
                   interface_T(pol, n_list[alllayer_index],
                               n_list[alllayer_index+1],
                               th_list[alllayer_index],
                               th_list[alllayer_index+1]))
            R_list[inc_index+1, inc_index] = (
                   interface_R(pol, n_list[alllayer_index+1],
                               n_list[alllayer_index],
                               th_list[alllayer_index+1],
                               th_list[alllayer_index]))
            T_list[inc_index+1, inc_index] = (
                   interface_T(pol, n_list[alllayer_index+1],
                               n_list[alllayer_index],
                               th_list[alllayer_index+1],
                               th_list[alllayer_index]))
        else: #next layer is coherent
            R_list[inc_index,inc_index+1] = (
                    coh_tmm_data_list[nextstack_index]['R'])
            T_list[inc_index,inc_index+1] = (
                    coh_tmm_data_list[nextstack_index]['T'])
            R_list[inc_index+1,inc_index] = (
                    coh_tmm_bdata_list[nextstack_index]['R'])
            T_list[inc_index+1,inc_index] = (
                    coh_tmm_bdata_list[nextstack_index]['T'])

    # L is the transfer matrix from the i'th to (i+1)st incoherent layer, see
    # manual
    L_list = [nan] # L_0 is not defined because 0'th layer has no beginning.
    Ltilde = (array([[1,-R_list[1,0]],
                     [R_list[0,1],
                      T_list[1,0]*T_list[0,1] - R_list[1,0]*R_list[0,1]]])
                / T_list[0,1])
    for i in range(1,num_inc_layers-1):
        L = np.dot(
           array([[1/P_list[i],0],[0,P_list[i]]]),
           array([[1,-R_list[i+1,i]],
                  [R_list[i,i+1],
                   T_list[i+1,i]*T_list[i,i+1] - R_list[i+1,i]*R_list[i,i+1]]])
           ) / T_list[i,i+1]
        L_list.append(L)
        Ltilde = np.dot(Ltilde,L)
    T = 1 / Ltilde[0,0]
    R = Ltilde[1,0] / Ltilde[0,0]

    # VW_list[n] = [V_n, W_n], the forward- and backward-moving intensities
    # at the beginning of the n'th incoherent layer. VW_list[0] is undefined
    # because 0'th layer has no beginning.
    VW_list=zeros((num_inc_layers, 2))
    VW_list[0,:] = [nan, nan]
    VW = array([[T],[0]])
    VW_list[-1,:] = np.transpose(VW)
    for i in range(num_inc_layers-2, 0, -1):
        VW = np.dot(L_list[i], VW)
        VW_list[i,:] = np.transpose(VW)

    # stackFB_list[n]=[F,B] means that F is light traveling forward towards n'th
    # stack and B is light traveling backwards towards n'th stack.cuo
    # Reminder: inc_from_stack[i] = j means that the i'th stack comes after the
    # layer with incoherent index j.
    stackFB_list = []
    for stack_index, prev_inc_index in enumerate(inc_from_stack):
        if prev_inc_index == 0: #stack starts right after semi-infinite layer.
            F = 1
        else:
            F = VW_list[prev_inc_index][0] * P_list[prev_inc_index]
        B = VW_list[prev_inc_index+1][1]
        stackFB_list.append([F,B])

    # power_entering_list[i] is the normalized Poynting vector crossing the
    # interface into the i'th incoherent layer from the previous (coherent or
    # incoherent) layer. See manual.
    power_entering_list = [1] #"1" by convention for infinite 0th layer.
    for i in range(1,num_inc_layers):
        prev_stack_index = stack_from_inc[i]
        if isnan(prev_stack_index):
            #case where this layer directly follows another incoherent layer
            if i == 1: #special case because VW_list[0] & A_list[0] are undefined
                power_entering_list.append(T_list[0,1]
                                            - VW_list[1][1]*T_list[1,0])
            else:
                power_entering_list.append(
                    VW_list[i-1][0]*P_list[i-1]*T_list[i-1,i]
                    - VW_list[i][1]*T_list[i,i-1])
        else: #case where this layer follows a coherent stack
            power_entering_list.append(
                stackFB_list[prev_stack_index][0] *
                 coh_tmm_data_list[prev_stack_index]['T']
                - stackFB_list[prev_stack_index][1] *
                 coh_tmm_bdata_list[prev_stack_index]['power_entering'])
    ans = {'T':T, 'R':R, 'VW_list':VW_list,
            'coh_tmm_data_list':coh_tmm_data_list,
            'coh_tmm_bdata_list':coh_tmm_bdata_list,
            'stackFB_list':stackFB_list,
            'power_entering_list':power_entering_list}
    ans.update(group_layers_data)
    return ans

def inc_absorp_in_each_layer(inc_data):
    """
    A list saying what proportion of light is absorbed in each layer.

    Assumes all reflected light is eventually absorbed in the 0'th medium, and
    all transmitted light is eventually absorbed in the final medium.

    Returns a list [layer0absorp, layer1absorp, ...]. Entries should sum to 1.

    inc_data is output of incoherent_main()
    """
    # Reminder: inc_from_stack[i] = j means that the i'th stack comes after the
    # layer with incoherent index j.
    # Reminder: stack_from_inc[i] = j means that the layer
    # with incoherent index i comes immediately after the j'th stack (or j=nan
    # if it's not immediately following a stack).

    stack_from_inc = inc_data['stack_from_inc']
    power_entering_list = inc_data['power_entering_list']
    # stackFB_list[n]=[F,B] means that F is light traveling forward towards n'th
    # stack and B is light traveling backwards towards n'th stack.
    stackFB_list = inc_data['stackFB_list']
    absorp_list = []

    # loop through incoherent layers, excluding the final layer
    for i, power_entering in enumerate(power_entering_list[:-1]):
        if isnan(stack_from_inc[i+1]):
            # case that incoherent layer i is right before another incoherent layer
            absorp_list.append(power_entering_list[i]-power_entering_list[i+1])
        else: #incoherent layer i is immediately before a coherent stack
            j = stack_from_inc[i+1]
            coh_tmm_data = inc_data['coh_tmm_data_list'][j]
            coh_tmm_bdata = inc_data['coh_tmm_bdata_list'][j]
            # First, power in the incoherent layer...
            power_exiting = (
               stackFB_list[j][0] * coh_tmm_data['power_entering']
                  - stackFB_list[j][1] * coh_tmm_bdata['T'])
            absorp_list.append(power_entering_list[i]-power_exiting)
            # Next, power in the coherent stack...
            stack_absorp = ((stackFB_list[j][0] *
                        absorp_in_each_layer(coh_tmm_data))[1:-1]
                       + (stackFB_list[j][1] *
                        absorp_in_each_layer(coh_tmm_bdata))[-2:0:-1])
            absorp_list.extend(stack_absorp)
    # final semi-infinite layer
    absorp_list.append(inc_data['T'])
    return absorp_list

def inc_find_absorp_analytic_fn(layer, inc_data):
    """
    Outputs an absorp_analytic_fn object for a coherent layer within a
    partly-incoherent stack.

    inc_data is output of incoherent_main()
    """
    j = inc_data['stack_from_all'][layer]
    if isnan(j):
        raise ValueError('layer must be coherent for this function!')
    [stackindex, withinstackindex] = j
    forwardfunc = absorp_analytic_fn()
    forwardfunc.fill_in(inc_data['coh_tmm_data_list'][stackindex],
                        withinstackindex)
    forwardfunc.scale(inc_data['stackFB_list'][stackindex][0])
    backfunc = absorp_analytic_fn()
    backfunc.fill_in(inc_data['coh_tmm_bdata_list'][stackindex],
               -1-withinstackindex)
    backfunc.scale(inc_data['stackFB_list'][stackindex][1])
    backfunc.flip()
    return forwardfunc.add(backfunc)




if __name__ =='__main__':
    wavelength = 500e-9  # Wavelength in meters (e.g., 532 nm)
    n_tio2_500nm = n_tio2(wavelength)
    print("Refractive index of TiO2 at 500 nm:", n_tio2_500nm)
#     dispersion_list = [n_sio2, n_tio2, n_sio2]  # Example materials
#     d_list = [inf, 0.5e-6, 0.5e-6, inf]  # Example thicknesses in meters
#     th_0 = 0  # Normal incidence
#     lam_vac = 550e-9  # Wavelength in meters (550 nm)
#
#
#     result = coh_tmm_dispersion('s', dispersion_list, d_list, th_0, lam_vac)
#     print(result)
