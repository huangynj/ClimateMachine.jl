### Soil heat model

export SoilHeatModel, PrescribedTemperatureModel

abstract type AbstractHeatModel <: AbstractSoilComponentModel end

"""
    struct PrescribedTemperatureModel{F1} <: AbstractHeatModel

Model structure for a prescribed temperature model.

Document.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedTemperatureModel{F1} <: AbstractHeatModel
    "Temperature"
    T::F1
end

"""
    PrescribedTemperatureModel(
        T::Function = (aux,t) -> eltype(aux)(0.0)
    )

Outer constructor for the PrescribedTemperatureModel defining default values, and
making it so changes to those defaults are supplied via keyword args.

The functions supplied by the user are point-wise evaluated and are 
evaluated in the Balance Law functions (kernels?) compute_gradient_argument,
 nodal_update, etc. whenever the prescribed temperature content variables are 
needed by the water model.
"""
function PrescribedTemperatureModel(
    T::Function = (aux,t) -> eltype(aux)(0.0)
)
    return PrescribedTemperatureModel{typeof(T)}(T)
end

"""
    SoilHeatModel{FT, FiT, BCD, BCN} <: AbstractHeatModel

The necessary components for the Heat Equation in a soil water matrix.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilHeatModel{FT, FiT, BCD, BCN} <: AbstractHeatModel
    "Initial conditions for temperature"
    initialT::FiT
    "Dirichlet BC structure"
    dirichlet_bc::BCD
    "Neumann BC structure"
    neumann_bc::BCN
end

"""
    SoilHeatModel(
        ::Type{FT};
        initialT::FT = FT(NaN),
        dirichlet_bc::AbstractBoundaryFunctions = nothing,
        neumann_bc::AbstractBoundaryFunctions = nothing
    ) where {FT}

Constructor for the SoilHeatModel. Defaults imply ....
"""
function SoilHeatModel(
    ::Type{FT};
    initialT = (aux) -> FT(NaN),
    dirichlet_bc::AbstractBoundaryFunctions = nothing,
    neumann_bc::AbstractBoundaryFunctions = nothing
) where {FT}
    args = (
        initialT,
        dirichlet_bc,
        neumann_bc
    )
    return SoilHeatModel{FT, typeof.(args)...}(args...)
end


#Need these get_temperature functions for SoilHeatModel.
"""
    function get_temperature(
        heat::PrescribedTemperatureModel,
        aux::Vars,
        state::Vars,
        t::Real
    )

Returns the temperature when the heat model chosen is a user prescribed one.

This is useful for driving Richard's equation without a back reaction on temperature.
"""
function get_temperature(
    heat::PrescribedTemperatureModel,
    aux::Vars,
    state::Vars,
    t::Real
)
    T = heat.T(aux, t)
    return T
end


"""
    function get_initial_temperature(
        m::PrescribedTemperatureModel,
        aux::Vars,
        t::Real
    )    

Returns the temperature from the prescribed model.
Needed for soil_init_aux! of SoilWaterModel.
"""
function get_initial_temperature(
    m::PrescribedTemperatureModel,
    aux::Vars,
    t::Real,
)
    return m.T(aux, t)
end


function get_clima_params_for_heat(land::LandModel, FT)
    _T_ref = FT(T_0(land.param_set))
    _LH_f0 = FT(LH_f0(land.param_set))

    _ρ_i = FT(ρ_cloud_ice(land.param_set))
    _cp_i = FT(cp_i(land.param_set) * _ρ_i)

    _ρ_l = FT(ρ_cloud_liq(land.param_set))
    _cp_l = FT(cp_l(land.param_set) * _ρ_l)
    
    return  _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l
end

#Probably we dont need κ in aux, unless it is helpful for debugging. 

vars_state(heat::SoilHeatModel, st::Prognostic, FT) = @vars(I::FT)
vars_state(heat::SoilHeatModel, st::Auxiliary, FT) = @vars(T::FT)#, κ::FT)
vars_state(heat::SoilHeatModel, st::Gradient, FT) = @vars(T::FT)
vars_state(heat::SoilHeatModel, st::GradientFlux, FT) = @vars(κ∇T::SVector{3, FT})

function soil_init_aux!(
	land::LandModel,
	soil::SoilModel,
	heat::SoilHeatModel,
	aux::Vars,
	geom::LocalGeometry
	)
    aux.soil.heat.T = heat.initialT(aux)
end

function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    state::Vars,
    aux::Vars,
    t::Real
)
    FT = eltype(state)
    _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)#

    ϑ_l, θ_ice = get_water_content(land.soil.water,aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    c_ds = soil.param_functions.c_ds
    cs = volumetric_heat_capacity(θ_l, θ_ice, c_ds, _cp_l, _cp_i)
    aux.soil.heat.T = temperature_from_I(_T_ref, state.soil.heat.I, θ_ice, _ρ_i, _LH_f0, cs)


end

function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real
)

    FT = eltype(state)
    _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)

    ϑ_l, θ_ice = get_water_content(land.soil.water,aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    c_ds = soil.param_functions.c_ds
    cs = volumetric_heat_capacity(θ_l, θ_ice, c_ds, _cp_l, _cp_i)

    transform.soil.heat.T = temperature_from_I(_T_ref, state.soil.heat.I, θ_ice, _ρ_i, _LH_f0, cs)#aux.soil.heat.T
end

function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)#
    ϑ_l, θ_ice = get_water_content(land.soil.water,aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    κ_dry = soil.param_functions.κ_dry
    S_r = relative_saturation(θ_l, θ_ice, soil.param_functions.porosity)
    kersten  = kersten_number(θ_ice, S_r, soil.param_functions.a, soil.param_functions.b,
                              soil.param_functions.ν_om, soil.param_functions.ν_sand,
                              soil.param_functions.ν_gravel)
    κ_sat = saturated_thermal_conductivity(θ_l, θ_ice, soil.param_functions.κ_sat_unfrozen,
                                           soil.param_functions.κ_sat_frozen)
    diffusive.soil.heat.κ∇T = thermal_conductivity(κ_dry, kersten, κ_sat) * ∇transform.soil.heat.T
end

function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)
    I_l = internal_energy_liquid_water(_cp_l, aux.soil.heat.T, _T_ref, _ρ_l)
    diffusive_water_flux = -I_l .* get_diffusive_water_flux(soil.water, diffusive, FT)
    flux.soil.heat.I -= diffusive.soil.heat.κ∇T + diffusive_water_flux

end
