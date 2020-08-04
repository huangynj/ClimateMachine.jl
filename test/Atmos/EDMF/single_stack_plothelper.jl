using ClimateMachine.SingleStackUtils
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using Plots
include(joinpath(clima_dir, "docs", "plothelpers.jl"));
using OrderedCollections

function dict_of_states(solver_config)
    FT = eltype(solver_config.Q)
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        solver_config.dg.grid,
        solver_config.Q,
        vars_state(solver_config.dg.balance_law, Prognostic(), FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        solver_config.dg.grid,
        solver_config.dg.state_auxiliary,
        vars_state(solver_config.dg.balance_law, Auxiliary(), FT),
        exclude = ["z"],
    )
    return OrderedDict(state_vars..., aux_vars...);
end
function plot_results(solver_config, all_data, time_data, output_dir)
    FT = eltype(solver_config.Q)
    z = get_z(solver_config.dg.grid)
    mkpath(output_dir)
    vs = vars_state(solver_config.dg.balance_law, Prognostic(), FT)
    for fn in flattenednames(vs)
        file_name = "prog_"*replace(fn, "."=>"_")
        export_plot(
            z,
            all_data,
            (fn,),
            joinpath(output_dir, "$(file_name).png");
            xlabel=fn,
            ylabel="z [m]",
            time_data=time_data,
            round_digits=5,
        );
    end
    vs = vars_state(solver_config.dg.balance_law, Auxiliary(), FT)
    for fn in flattenednames(vs)
        file_name = "aux_"*replace(fn, "."=>"_")
        export_plot(
            z,
            all_data,
            (fn,),
            joinpath(output_dir, "$(file_name).png");
            xlabel=fn,
            ylabel="z [m]",
            time_data=time_data,
            round_digits=5,
        );
    end
end

# After simulation runs, use:
# FT = eltype(solver_config.Q)
# all_data = [dict_of_states(solver_config)]
# time_data = FT[0]
# plot_results(solver_config, all_data, time_data, joinpath(clima_dir, "output", "steady_sol"))

