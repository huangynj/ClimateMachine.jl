
function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)

end


function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    m::AbstractSoilComponentModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)

end



function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    water_bc = water.dirichlet_bc
    if bctype == 2
        top_boundary_conditions!(land, soil, water, water_bc, state⁺, aux⁺, state⁻, aux⁻, t)
    elseif bctype == 1
        bottom_boundary_conditions!(
            land,
            soil,
            water,
            water_bc,
            state⁺,
            aux⁺,
            state⁻,
            aux⁻,
            t,
        )
    end
end


function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    water_bc = water.neumann_bc
    if bctype == 2
        top_boundary_conditions!(
            land,
            soil,
            water,
            water_bc,
            state⁺,
            diff⁺,
            aux⁺,
            n̂,
            state⁻,
            diff⁻,
            aux⁻,
            t,
        )
    elseif bctype == 1
        bottom_boundary_conditions!(
            land,
            soil,
            water,
            water_bc,
            state⁺,
            diff⁺,
            aux⁺,
            n̂,
            state⁻,
            diff⁻,
            aux⁻,
            t,
        )
    end
end

# Heat
function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    state⁺::Vars,
    aux⁺::Vars,
    nM,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    heat_bc = heat.dirichlet_bc
    if bctype == 2
        top_boundary_conditions!(land, soil, heat, heat_bc, state⁺, aux⁺, state⁻, aux⁻, t)
    elseif bctype == 1
        bottom_boundary_conditions!(
            land,
            soil,
            heat,
            heat_bc,
            state⁺,
            aux⁺,
            state⁻,
            aux⁻,
            t,
        )
    end
end

function soil_boundary_state!(
    nf,
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    heat_bc = heat.neumann_bc
    if bctype == 2
        top_boundary_conditions!(
            land,
            soil,
            heat,
            heat_bc,
            state⁺,
            diff⁺,
            aux⁺,
            n̂,
            state⁻,
            diff⁻,
            aux⁻,
            t,
        )
    elseif bctype == 1
        bottom_boundary_conditions!(
            land,
            soil,
            heat,
            heat_bc,
            state⁺,
            diff⁺,
            aux⁺,
            n̂,
            state⁻,
            diff⁻,
            aux⁻,
            t,
        )
    end
end


# Water
"""
    top_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        bc::Neumann,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Neumann boundary conditions for the top of the soil, if given.
"""
function top_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    bc::Neumann,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.surface_flux != nothing
        diff⁺.soil.water.K∇h = n̂ * bc.surface_flux(aux⁻, t)
    else
        nothing
    end
end

"""
    top_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        bc::Dirichlet,
        state⁺::Vars,
        aux⁺::Vars,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Dirichlet boundary conditions for the top of the soil, if given.
"""
function top_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    bc::Dirichlet,
    state⁺::Vars,
    aux⁺::Vars,
    state⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.surface_state != nothing
        state⁺.soil.water.ϑ_l = bc.surface_state(aux⁻, t)
    else
        nothing
    end
end

"""
    bottom_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        bc::Neumann,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Neumann boundary conditions for the bottom of the soil, if given.
"""
function bottom_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    bc::Neumann,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.bottom_flux != nothing
        diff⁺.soil.water.K∇h = -n̂ * bc.bottom_flux(aux⁻, t)
    else
        nothing
    end
end


"""
    bottom_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        bc::Dirichlet,
        state⁺::Vars,
        aux⁺::Vars,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Dirichlet boundary conditions for the bottom of the soil, if given.
"""
function bottom_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    water::SoilWaterModel,
    bc::Dirichlet,
    state⁺::Vars,
    aux⁺::Vars,
    state⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.bottom_state != nothing
        state⁺.soil.water.ϑ_l = bc.bottom_state(aux⁻, t)
    else
        nothing
    end
end

## Heat
"""
    top_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        bc::Neumann,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Neumann boundary conditions for the top of the soil, if given.
"""
function top_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    bc::Neumann,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.surface_flux != nothing
        diff⁺.soil.heat.κ∇T = n̂ * bc.surface_flux(aux⁻, t)
    else
        nothing
    end
end

"""
    top_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        water::SoilWaterModel,
        bc::Dirichlet,
        state⁺::Vars,
        aux⁺::Vars,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Dirichlet boundary conditions for the top of the soil, if given.
"""
function top_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    bc::Dirichlet,
    state⁺::Vars,
    aux⁺::Vars,
    state⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.surface_state != nothing
       FT = eltype(state⁻)
       _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)
       ϑ_l, θ_ice = get_water_content(land.soil.water, aux⁻, state⁻, t)
       θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
       c_s = volumetric_heat_capacity(θ_l, θ_ice, land.soil.param_functions.c_ds,
                                      _cp_l, _cp_i)
        
       I_bc = internal_energy(
       θ_ice,
       c_s,
       bc.surface_state(aux⁻, t),
       _T_ref,
       _ρ_i,
       _LH_f0)

        state⁺.soil.heat.I = I_bc
    else
        nothing
    end
end

"""
    bottom_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        bc::Neumann,
        state⁺::Vars,
        diff⁺::Vars,
        aux⁺::Vars,
        n̂,
        state⁻::Vars,
        diff⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Neumann boundary conditions for the bottom of the soil, if given.
"""
function bottom_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    bc::Neumann,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n̂,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.bottom_flux != nothing
        diff⁺.soil.heat.κ∇T = -n̂ * bc.bottom_flux(aux⁻, t)
    else
        nothing
    end
end


"""
    bottom_boundary_conditions!(
        land::LandModel,
        soil::SoilModel,
        heat::SoilHeatModel,
        bc::Dirichlet,
        state⁺::Vars,
        aux⁺::Vars,
        state⁻::Vars,
        aux⁻::Vars,
        t,
    )

Specify Dirichlet boundary conditions for the bottom of the soil, if given.
"""
function bottom_boundary_conditions!(
    land::LandModel,
    soil::SoilModel,
    heat::SoilHeatModel,
    bc::Dirichlet,
    state⁺::Vars,
    aux⁺::Vars,
    state⁻::Vars,
    aux⁻::Vars,
    t,
)
    if bc.bottom_state != nothing
       FT = eltype(state⁻)
       _T_ref, _LH_f0, _ρ_i, _ρ_l, _cp_i, _cp_l = get_clima_params_for_heat(land, FT)
       ϑ_l, θ_ice = get_water_content(land.soil.water, aux⁻, state⁻, t)
       θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
       c_s = volumetric_heat_capacity(θ_l, θ_ice, land.soil.param_functions.c_ds,
                                      _cp_l, _cp_i)
        
        I_bc = internal_energy(
        θ_ice,
        c_s,
        bc.bottom_state(aux⁻, t),
        _T_ref,
        _ρ_i,
        _LH_f0)

        state⁺.soil.heat.I = I_bc
    else
        nothing
    end
end
