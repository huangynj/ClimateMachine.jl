# Microphysics_0M

The `Microphysics_0M.jl` module defines a 0-moment bulk parameterization of
  precipitation removal.
It offers a very simple way of removing the excess water from the system without
  assuming anything about the size distributions of cloud
  or precipitation particles.

The `q_tot` (total water specific humidity) tendency
  is obtained by relaxation with a constant timescale
  to a state with precipitable water removed.
The threshold for when to remove `q_tot` is defined either by the
  condensate specific humidity or supersaturation.
The thresholds and the relaxation timescale are defined in
  CLIMAParameters.

!!! note

    The implementation assumes that
    `q_liq` (liquid water specific humidity) and
    `q_ice` (ice water specific humidity) are in the auxiliary variables.

!!! note

    To remove precipitation instantly the relaxation timescale should be
    equal to the timestep length.
    Not sure if we want to implement it this way though.

## Precipitation removal tendency

If based on maximum condensate specific humidity the tendency is defined as:
``` math
\begin{equation}
  \left. \frac{d \, q_{tot}}{dt} \right|_{precip} =-
    \frac{max(0, q_{liq} + q_{ice} - q_{c0})}{\tau_{precip}}
\end{equation}
```
where:
  - `q_{liq}`, `q_{ice}` are cloud liquid water and cloud ice specific humidities,
  - `q_{c0}` is the condensate specific humidity threshold above which water is removed,
  - `\tau_{precip}` is the relaxation timescale.

If based on saturation excess the tendency is defined as:
```math
\begin{equation}
  \left. \frac{d \, q_{tot}}{dt} \right|_{precip} =-
    \frac{max(0, q_{liq} + q_{ice} - S_{0} \, q_{vap}^{sat})}{\tau_{precip}}
\end{equation}
```
where:
  - `q_{liq}`, `q_{ice}` are cloud liquid water and cloud ice specific humidities,
  - `S_{0}` is the supersaturation threshold above which water is removed,
  - `q_{vap}^{sat}` is the saturation specific humidity,
  - `\tau_{precip}` is the relaxation timescale.

## Coupling to the state variables

Removing the `q_tot` changes the mass of the working fluid.
Therefore the state variables `\rho` and `\rho q_{tot}` are by:

```math
\begin{equation}
\left. \frac{d \, \rho}{dt} \right|_{precip} =
  \left. \frac{d \, \rho q_{tot}}{dt} \right|_{precip} =
  \frac{\rho}{q_{dry}} \, \left. \frac{d \, q_{tot}}{dt} \right|_{precip}
\end{equation}
```
where:
  - `\rho` is the air density,
  - `q_{dry} = 1 - q_{tot}` is the dry air specific humidity.

The change to the state variable `\rho e` is computed as a
liquid and ice fraction weighted sum of the
changes to internal energy due to removing cloud liquid water and cloud ice:

```math
\begin{equation}
\left. \frac{d \, \rho e}{dt} \right|_{precip} =
  \left(
    \frac{q_{liq}}{q_{liq} + q_{ice}} (c_{vl} - c_{vd}) (T - T_0) +
    \frac{q_{ice}}{q_{liq} + q_{ice}} ((c_{vi} - c_{vd}) (T - T_0) - e_{int, i0}) +
                \frac{e}{\rho}
  \right) \rho \, \left. \frac{d \, q_{tot}}{dt} \right|_{precip}
\end{equation}
```
where:
 - `e` is the specific total energy,
 - `T` is the temperature,
 - `T_0` is the reference temperature,
 - `e_{int, i0}` is the specific internal energy of ice at `T_0`
 - `c_{vd}`, `c_{vl}` and `c_{vi}` are the isochoric specific heats of dry air,
    liquid water and ice.
