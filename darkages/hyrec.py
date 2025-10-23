import numpy as np
import os

def set_up_hyrec(H0, omb, omc, omk, yhe,
                 base_dir='./'):
    """
    Code to set up hyrec. Code builds the input.dat file from a template
    given the cosmological parameters input to this class.
    """
    
    labels = ['h', 'T0CMB', 'Omega_b', 'Omega_m',
                'Omega_k', 'w0, wa', 'Nmnu', 'mnu1', 'mnu2', 'mnu3',
                'Y_He', 'Neff', ' ', 'alpha(rec)/alpha(today)', 
                'me(rec)/me(today)',
                'pann', 'pann_halo', 'ann_z', 'ann_zmax',
                'ann_zmin', 'ann_var', 'ann_z_halo', 'on_the_spot',
                'decay', ' ', 'Mpbh', 'fpbh', ' ', ' ']

    with open('HYREC-2/input.dat') as file:
        data = {}
        for i, line in enumerate(file):
            if i <= 28:
                if labels[i] == ' ':
                    data[i] = line.rstrip()
                else:
                    data[labels[i]] = line.rstrip()

    data['h'] = str(H0/100)
    data['Omega_b'] = str(omb*(H0/100)**2)
    data['Omega_m'] = str((omc+omb)*(H0/100)**2)
    data['Omega_k'] = str(omk)
    data['Y_He'] = str(yhe)

    with open(base_dir + 'hyrec_input.dat', 'w') as file:
        for key, value in data.items():
            file.write(value + '\n')

def call_hyrec(base_dir='./', redshift=1100):
    """
    Code to call hyrec. Code runs the hyrec executable and returns the
    output as 3D grids with the same shape as the initial conditions.
    """

    # need to be in the hyrec dirtectory to run...
    os.chdir('HYREC-2')
    os.system('./hyrec < ../' + base_dir + '/hyrec_input.dat' + '> /dev/null 2>&1')
    os.chdir('..')
    
    data = np.loadtxt('HYREC-2/output_xe.dat')
    hyrec_z = data[:, 0][::-1]
    xe = data[:, 1][::-1]
    Tk = data[:, 2][::-1]

    init_xe = np.interp(redshift, hyrec_z, xe)
    init_Tk = np.interp(redshift, hyrec_z, Tk) # in Kelvin

    return init_xe, init_Tk