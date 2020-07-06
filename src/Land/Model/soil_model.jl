#### Soil model

export SoilModel, Dirichlet, Neumann

"""
    SoilModel{W, H} <: BalanceLaw

A BalanceLaw for soil modeling.
Users may over-ride prescribed default values for each field.

# Usage

    SoilModel(
        water,
        heat,
    )


# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilModel{W, H} <: BalanceLaw
    "Water model"
    water::W
    "Heat model"
    heat::H
end

"""
    vars_state_conservative(soil::SoilModel, FT)

Conserved state variables (Prognostic Variables)
"""
function vars_state_conservative(soil::SoilModel, FT)
    @vars begin
        water::vars_state_conservative(soil.water, FT)
        heat::vars_state_conservative(soil.heat, FT)
    end
end

"""
    vars_state_auxiliary(soil::SoilModel, FT)

Names of variables required for the balance law that aren't related to 
derivatives of the state variables (e.g. spatial coordinates or various 
integrals) or those needed to solve expensive auxiliary equations (e.g., 
temperature via a non-linear equation solve)
"""
function vars_state_auxiliary(soil::SoilModel, FT)
    @vars begin
        water::vars_state_auxiliary(soil.water, FT)
        heat::vars_state_auxiliary(soil.heat, FT)
    end
end

"""
    vars_state_gradient(soil::SoilModel, FT)

Names of the gradients of functions of the conservative state variables. 
Used to represent values before **and** after differentiation
"""
function vars_state_gradient(soil::SoilModel, FT)
    @vars begin
        water::vars_state_gradient(soil.water, FT)
        heat::vars_state_gradient(soil.heat, FT)
    end
end

"""
    vars_state_gradient_flux(soil::SoilModel, FT)

Names of the gradient fluxes necessary to impose Neumann boundary 
conditions.
"""
function vars_state_gradient_flux(soil::SoilModel, FT)
    @vars begin
        water::vars_state_gradient_flux(soil.water, FT)
        heat::vars_state_gradient_flux(soil.heat, FT)
    end
end

"""
    flux_first_order!(
        Land::LandModel,
        soil::SoilModel,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model equations.
"""
function flux_first_order!(
    land::LandModel,
    soil::SoilModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end


"""
    compute_gradient_argument!(
        land::LandModel,
        soil::SoilModel,
        transform::Vars,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Specify how to compute the arguments to the gradients.
"""
function compute_gradient_argument!(
    land::LandModel,
    soil::SoilModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

      compute_gradient_argument!(
        land,
        soil.heat,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(land, soil.water, transform, state, aux, t)
end


"""
    compute_gradient_flux!(
        land::LandModel,
        soil::SoilModel,
        diffusive::Vars,
        ∇transform::Grad,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Specify how to compute gradient fluxes.
"""
function compute_gradient_flux!(
    land::LandModel,
    soil::SoilModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

      compute_gradient_flux!(
        land,
        soil.heat,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        land,
        soil.water,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end

"""
    flux_second_order!(
        land::LandModel,
        soil::SoilModel,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real,
    )

Specify the second order flux for each conservative state variable
"""

function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        soil.heat,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )
    flux_second_order!(
        land,
        soil.water,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end

"""
    land_nodal_update_auxiliary_state!(
        land::LandModel,
        soil::SoilModel,
        state::Vars,
        aux::Vars,
        t::Real,
    )

Update the auxiliary state array.

Call the different methods of land_nodal_update_auxiliary_state!
for the various subcomponents of soil.
"""
function land_nodal_update_auxiliary_state!(
    land::LandModel,
    soil::SoilModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, soil, soil.water, state, aux, t)
    land_nodal_update_auxiliary_state!(land, soil, soil.heat, state, aux, t)
end

"""
    land_init_aux!(
        land::LandModel, 
        soil::SoilModel,
        aux::Vars, 
        geom::LocalGeometry
)

Calls the various versions of init_aux for the subcomponents of soil.
"""
function land_init_aux!(
    land::LandModel,
    soil::SoilModel,
    aux::Vars,
    geom::LocalGeometry,
)
    heat_init_aux!(land, soil, soil.water, aux, geom)
    water_init_aux!(land, soil, soil.water, aux, geom)
end

abstract type bc_functions end

"""
    struct Dirichlet{Fs, Fb} <: bc_functions

A concrete type to hold the surface state and bottom state variable 
values/functions, if Dirichlet boundary conditions are desired.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct Dirichlet{Fs, Fb} <: bc_functions
    "Surface state boundary condition"
    surface_state::Fs = nothing
    "Bottom state boundary condition"
    bottom_state::Fb = nothing
end

"""
    struct Neumann{Fs, Fb} <: bc_functions

A concrete type to hold the surface and/or bottom diffusive flux 
values/functions, if Neumann boundary conditions are desired.

Note that these are intended to be scalar values. In the boundary_state!
functions, they are multiplied by the ẑ vector (i.e. the normal vector n̂
to the domain at the upper boundary, and -n̂ at the lower boundary. These
vectors point out of the domain.)

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct Neumann{Fs, Fb} <: bc_functions
    surface_flux::Fs = nothing
    bottom_flux::Fb = nothing
end

include("./soil_bc.jl")
