from sympl import initialize_numpy_arrays_with_properties, get_constant
from sympl import Stepper
import numpy as np
from numba import jit
from ..._core import bolton_q_sat


# @jit(nopython=True)
def calculate_fields_flux(air_temp, air_press, air_press_int, surf_temp, surf_press, spec_hum, north_wind, east_wind, Rd, Rh2o, Cp_dry, g,
                         P0, k, z0, Ri_c, surf_hum, scaling):

    z_a =  (Rd*air_temp*(1+0.608*spec_hum)/g) * np.log(surf_press/air_press)

    surf_hum[:] = bolton_q_sat(surf_temp, surf_press, Rd, Rh2o) * scaling
    surf_hum[surf_hum>1] = 1
    pot_temp_a = air_temp * np.power((P0/air_press), Rd/Cp_dry)
    pot_temp_surf = surf_temp * np.power((P0/surf_press), Rd/Cp_dry)
    pot_virt_temp_a = pot_temp_a*(1+0.608*spec_hum)
    pot_virt_temp_surf = pot_temp_surf*(1+0.608*surf_hum)
    
    wind_a = np.sqrt(np.power(north_wind, 2)+np.power(east_wind, 2))
    wind_a[np.where(wind_a<1)]=1

    rho_a = air_press/(Rd * (1+0.608*spec_hum) * air_temp)
    layer_thickness = (air_press_int[0]-air_press_int[1])/g

    Ri_a = g*z_a*(pot_virt_temp_a-pot_virt_temp_surf)/(pot_virt_temp_surf*wind_a*wind_a)

    # calculate drag coefficients
    C=np.zeros(Ri_a.shape)
    mask=Ri_a <= 0
    C[mask]=(k*k*np.power(np.log(z_a[mask]/z0), -2))
    mask=np.logical_and( Ri_a>0, Ri_a < Ri_c)
    C[mask] = (k*k*np.power(np.log(z_a[mask]/z0), -2)*np.power((1-Ri_a[mask]/Ri_c), 2))

    return pot_temp_a, pot_temp_surf, wind_a, rho_a, layer_thickness, Ri_a, C


# @jit(nopython=True)
def flux(air_temp,spec_hum, north_wind, east_wind, sat_spec_hum, rho, pot_temp, pot_temp_surf,
        wind, layer_thickness, Cp_dry, L, timestep, north_stress, east_stress, sens_flux, lat_flux, C):
    
    temp = rho*C*wind
    north_stress[:] = temp*north_wind[0]
    east_stress[:] = temp*east_wind[0]
    sens_flux[:] = -temp*Cp_dry*(pot_temp-pot_temp_surf)
    evap = temp*(spec_hum[0]-sat_spec_hum)
    lat_flux[:] = -L*evap

    air_temp[0] = air_temp[0]+sens_flux\
            / (Cp_dry*layer_thickness) * timestep
    spec_hum[0] = spec_hum[0]-evap\
            / (layer_thickness) * timestep
    north_wind[0] = north_wind[0]-north_stress\
            / (layer_thickness) * timestep
    east_wind[0] = east_wind[0]-east_stress\
            / (layer_thickness) * timestep


# @jit
def calculate_fields_boundary(air_temp,spec_hum,north_wind, east_wind, air_press_int, surf_temp, surf_press,
                            sat_spec_hum, Rd, P0, Cp_dry, g, fb, Ri_a, C, k, Ri_c, z0, h):

   
    air_temp_int = 0.5*(air_temp[1:] +
                                       air_temp[:-1])
    spec_hum_int = 0.5*(spec_hum[1:] +
                                         spec_hum[:-1])
    north_wind_int = 0.5*(north_wind[1:]+north_wind[:-1])
    east_wind_int = 0.5*(east_wind[1:]+east_wind[:-1])
    rho = air_press_int[1:-1]/(Rd * (1+0.608 *
                                              spec_hum_int) *
                                              air_temp_int)

    n, col = np.shape(air_temp_int)[0], np.shape(air_temp_int)[1]
    
    wind_int = np.sqrt(np.power(north_wind_int, 2) +
                           np.power(east_wind_int, 2))
    
    wind_int[np.where(wind_int<1)]=1

    pot_virt_temp = air_temp_int *(1+0.608*spec_hum_int)*\
        np.power((P0/air_press_int[1:-1]), Rd/Cp_dry) 
    pot_virt_temp_surf = surf_temp *(1+0.608*sat_spec_hum)*\
        np.power((P0/surf_press), Rd/Cp_dry) 

    z_int = np.cumsum(Rd*(1+0.608*spec_hum_int) *air_temp_int/g *np.log(air_press_int[:-2]/air_press_int[1:-1]), axis=0)

    Rich = g*z_int*(pot_virt_temp-pot_virt_temp_surf)/(pot_virt_temp_surf*wind_int*wind_int)

    h[:] = z_int[(np.argmax(Rich > Ri_c, axis=0),range(col))]

    diff = np.zeros((n,col))

    h_cast = np.broadcast_to(h,(n,col))
    Ria_cast = np.broadcast_to(Ri_a,(n,col))
    C_cast = np.broadcast_to(C,(n,col))
    wind_a_cast = np.broadcast_to(wind_int[0],(n,col))

    mask = z_int<fb*h_cast
    mask_add = Ria_cast>0
    mask2 = np.logical_and(mask, mask_add) 
    diff[mask] =  k*wind_a_cast[mask]*np.sqrt(C_cast[mask])*z_int[mask]
    diff[mask2] = diff[mask2]/(1+Ria_cast[mask2]/Ri_c*np.log(z_int[mask2]/z0)/(1-Ria_cast[mask2]/Ri_c))

    mask=np.logical_and(z_int>=fb*h_cast,z_int<h_cast)
    diff[mask] = k*wind_a_cast[mask]*np.sqrt(C_cast[mask])*fb*h_cast[mask]*(z_int[mask]/(fb*h_cast[mask]) *\
        np.power((1-(z_int[mask]-fb*h_cast[mask])/((1-fb)*h_cast[mask])), 2))
    mask2 = np.logical_and(mask, mask_add) 
    diff[mask2] = diff[mask2]/(1+Ria_cast[mask2]/Ri_c*np.log(fb*h_cast[mask2]/z0)/(1-Ria_cast[mask2]/Ri_c))

    return rho, diff  


# @jit(nopython=True)
def TDMAsolver(a, b, c, d):
        
        n, m = np.shape(d)[0], np.shape(d)[1]
        w = np.zeros((n-1,m))
        g = np.zeros((n,m))
        p = np.zeros((n,m))

        w[0] = c[0]/b[0]
        g[0] = d[0]/b[0]

        for i in range(1, n-1):
            w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
        for i in range(1, n):
            g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
        p[n-1] = g[n-1]
        for i in range(n-1, 0, -1):
            p[i-1] = g[i-1] - w[i-1]*p[i]

        return p


# @jit(nopython=True)
def boundary(air_temp, spec_hum,north_wind, east_wind, air_press, air_press_int, rho, diff, g, P0, Rd, Cp, timestep):

    n, col = air_temp.shape[0], air_temp.shape[1]

    diag_m = np.zeros((n,col))
    diag_p = np.zeros((n,col))

    temp = g*g*rho*rho*diff*timestep/(air_press[:-1]-air_press[1:])

    diag_m[1:] = temp* 1/(air_press_int[1:-1]-air_press_int[2:])
    diag_p[:-1] = temp*1/(air_press_int[:-2]-air_press_int[1:-1])

    diag=1+diag_m+diag_p
            
    air_temp[:] = (TDMAsolver(-diag_m[1:],diag,-diag_p[:-1],air_temp*np.power((P0/air_press), Rd/Cp)))/np.power((P0/air_press), Rd/Cp)
    # air_temp[:] = (TDMAsolver(-diag_m[1:],diag,-diag_p[:-1],air_temp))
    spec_hum[:] = TDMAsolver(-diag_m[1:],diag,-diag_p[:-1],spec_hum)
    north_wind[:] = TDMAsolver(-diag_m[1:],diag,-diag_p[:-1],north_wind)
    east_wind[:] = TDMAsolver(-diag_m[1:],diag,-diag_p[:-1],east_wind)


class SimpleBoundaryLayer(Stepper):
    """
    This is a simple boundary layer component that diffuses heat, humidity and
    momemtum upwards from the lowest model level.
    This component assumes that a surface flux component has been already run,
    which has made the changes due to surface fluxes at the lowest model
    level. This component then diffuses heat, humidity and momentum using
    diffusion coefficients calculated using the simplified Monin-Obukhov
    theory.
    """

    input_properties = {
        'air_temperature': {
            'dims': ['mid_levels', '*'],
            'units': 'degK ',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'air_pressure': {
            'dims': ['mid_levels', '*'],
            'units': 'Pa',
        },
        'air_pressure_on_interface_levels': {
            'dims': ['interface_levels', '*'],
            'units': 'Pa',
        },
        'northward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'eastward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'surface_air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'area_type': {
            'dims': ['*'],
            'units': 'dimensionless',
        },
    }

    output_properties = {
        'air_temperature': {
            'dims': ['mid_levels', '*'],
            'units': 'degK ',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'northward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'eastward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
    }

    diagnostic_properties = {
        'surface_upward_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_upward_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'northward_wind_stress': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'eastward_wind_stress': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'boundary_layer_height': {
            'dims': ['*'],
            'units': 'm',
        },
        'surface_specific_humidity': {
            'dims': ['*'],
            'units': 'kg/kg ',
        },
    }

    def __init__(self, scaling_land=1, von_karman_constant=0.4, roughness_length=0.0000321,
                 specific_fraction=0.1, reference_pressure=100000,
                 critical_richardson_number=1, **kwargs):
        """
        Args:
        roughness_length:
            A measure of the surface roughness.
        specific_fraction:
            A parameter used in the calculation of diffusion coefficients.
        reference_pressure:
            The reference pressure used in the potential temperature
            calculations.
        critical_richardson_number:
            A set threshold value which determines the diffusion coefficients
            and the height of the boundary layer.
        """

        self._scaling_land = scaling_land
        self._k = von_karman_constant
        self._z0 = roughness_length
        self._fb = specific_fraction
        self._P0 = reference_pressure
        self._Ric = critical_richardson_number
        self._update_constants()

        super(SimpleBoundaryLayer, self).__init__(**kwargs)

    def _update_constants(self):

        self._Rd = get_constant('gas_constant_of_dry_air', 'J kg^-1 K^-1')
        self._Cp =\
            get_constant('heat_capacity_of_dry_air_at_constant_pressure',
                         'J kg^-1 K^-1')
        self._g = get_constant('gravitational_acceleration', 'm s^-2')
        self._Rh2o = get_constant('gas_constant_of_vapor_phase', 'J/kg/degK')
        self._L =\
            get_constant('latent_heat_of_vaporization_of_water', 'J kg^-1')

    def array_call(self, state, timestep):
        """
        Takes temperature, humidty and wind profiles for each column and
        returns diffused temperature, humidity and wind profiles.
        """

        new_state = initialize_numpy_arrays_with_properties(
            self.output_properties, state, self.input_properties
        )

        new_state['air_temperature'][:] = state["air_temperature"]
        new_state['specific_humidity'][:] = state['specific_humidity']
        new_state['northward_wind'][:] = state['northward_wind']
        new_state['eastward_wind'][:] = state['eastward_wind']

        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, state, self.input_properties
        )

        area_type=state['area_type'].astype(str)
        scaling=np.ones(area_type.shape)
        mask = area_type=='land'
        scaling[mask]=self._scaling_land

        pot_temp_a, pot_temp_surf, wind_a, rho_a, layer_thickness, Ri_a, C = \
        calculate_fields_flux(state["air_temperature"][0], state['air_pressure'][0], state['air_pressure_on_interface_levels'],
                            state['surface_temperature'], state['surface_air_pressure'], state['specific_humidity'][0],
                             state['northward_wind'][0], state['eastward_wind'][0], self._Rd, self._Rh2o, self._Cp, self._g,
                             self._P0, self._k, self._z0, self._Ric, diagnostics['surface_specific_humidity'], scaling)

        rho, diff = \
        calculate_fields_boundary(state["air_temperature"],state['specific_humidity'],state['northward_wind'], state['eastward_wind'],
                                state['air_pressure_on_interface_levels'],state['surface_temperature'], state['surface_air_pressure'],
                                diagnostics['surface_specific_humidity'], self._Rd, self._P0, self._Cp, self._g, self._fb, Ri_a, C,
                                self._k, self._Ric, self._z0, diagnostics['boundary_layer_height'])

        flux(new_state['air_temperature'],new_state['specific_humidity'],new_state['northward_wind'], new_state['eastward_wind'],
            diagnostics['surface_specific_humidity'], rho_a, pot_temp_a, pot_temp_surf, wind_a, layer_thickness, self._Cp, self._L,
            timestep.total_seconds(),diagnostics['northward_wind_stress'], diagnostics['eastward_wind_stress'],
            diagnostics['surface_upward_sensible_heat_flux'], diagnostics['surface_upward_latent_heat_flux'], C)

        boundary(new_state['air_temperature'],new_state['specific_humidity'],new_state['northward_wind'], new_state['eastward_wind'],
                state['air_pressure'], state['air_pressure_on_interface_levels'], rho, diff, self._g, self._P0, self._Rd, self._Cp,
                timestep.total_seconds())

        return diagnostics, new_state