using Plots, StatsPlots, LaTeXStrings, Dates

using ..Covid19: plot_confidence_timeseries!


# Ehh, this can be made nicer...
function state_prediction_plot(data::ImperialUSAcases.Data, state_idx, predictions_state::AbstractMatrix, e_deaths_state::AbstractMatrix, Rt_state::AbstractMatrix, infectiousness::AbstractMatrix; normalize_pop::Bool = false, main_title="")
    pop = data.turing_data.population[state_idx]
    num_total_days = data.turing_data.num_total_days
    num_observed_days = length(data.turing_data.cases[state_idx])

    state_name = data.states[state_idx]
    start_date = first(data.state_to_dates[state_name])
    dates = cumsum(fill(Day(1), data.turing_data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = data.turing_data.deaths[state_idx]
    daily_cases = data.turing_data.cases[state_idx]

    p1 = plot(; xaxis = false, legend = :outertopright)
    bar!(preproc(daily_deaths), label="Observed daily deaths")
    title!(replace(state_name, "_" => " ") * " " * main_title)
    vline!([data.turing_data.epidemic_start[state_idx]], label="epidemic start", linewidth=2)
    vline!([num_observed_days], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :outertopright, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_state); label = "Expected daily deaths")
    bar!(preproc(daily_deaths), label="Recorded daily deaths (observed)", alpha=0.5)

    p3 = plot(; legend = :outertopright, xaxis=false)
    plot_confidence_timeseries!(p3, Rt_state; no_label = true)
    title!(L"$R_t$")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rt_state)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    p4 = plot(; legend = :outertopright, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_state); label = "Expected daily cases")
    plot_confidence_timeseries!(p4, preproc(infectiousness); label = "infectiousness")
    bar!(preproc(daily_cases), label="Recorded daily cases (observed)", alpha=0.5)

    vals = preproc(cumsum(e_deaths_state; dims = 1))
    p5 = plot(; legend = :outertopright, xaxis=false)
    plot_confidence_timeseries!(p5, vals; label = "Expected cumulative deaths")
    plot!(preproc(cumsum(daily_deaths)), label="Recorded cumulative deaths", color=:red)

    vals = preproc(cumsum(predictions_state; dims = 1)./ data.turing_data.population[state_idx])
    p6 = plot(; legend = :outertopright)
    plot_confidence_timeseries!(p6, vals; label = "Expected cumulative cases")
    plot!(preproc(cumsum(daily_cases)./ data.turing_data.population[state_idx]), label="Recorded cumulative cases", color=:red)

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(2000, 2000), sharex=true)
    xticks!(1:20:num_total_days, date_strings[1:20:end], xrotation=45)

    return p
end

function state_prediction_plot(data::ImperialUSAcases.Data, state_idx, cases, e_deaths, Rt, infectiousness; kwargs...)
    num_observed_days = length(e_deaths)
    e_deaths_state = hcat([e_deaths[t][state_idx] for t = 1:num_observed_days]...)
    Rt_state = hcat([Rt[t][state_idx] for t = 1:num_observed_days]...)
    predictions_state = hcat([cases[t][state_idx] for t = 1:num_observed_days]...)
    infectiousness_state = hcat([infectiousness[t][state_idx] for t = 1:num_observed_days]...)

    return state_prediction_plot(data, state_idx, predictions_state, e_deaths_state, Rt_state, infectiousness_state; kwargs...)
end

function states_prediction_plot(data::ImperialUSAcases.Data, vals::AbstractArray{<:Real, 3}; normalize_pop = false, no_label=false, kwargs...)
    lqs, mqs, hqs = [], [], []
    labels = []

    # `vals` is assumed to be of the shape `(num_states, num_days, num_samples)`
    num_states = size(vals, 1)

    for state_idx in 1:num_states
        val = vals[state_idx, :, :]
        n = size(val, 1)

        pop = data.turing_data.population[state_idx]
        num_total_days = data.turing_data.num_total_days
        num_observed_days = length(data.turing_data.cases[state_idx])

        state_name = data.states[state_idx]

        # A tiny bit of preprocessing of the data
        preproc(x) = normalize_pop ? x ./ pop : x

        tmp = preproc(val)
        qs = [quantile(tmp[t, :], [0.025, 0.5, 0.975]) for t = 1:n]
        lq, mq, hq = (eachrow(hcat(qs...))..., )

        push!(lqs, lq)
        push!(mqs, mq)
        push!(hqs, hq)
        push!(labels, state_name)
    end

    lqs = reduce(hcat, collect.(lqs))
    mqs = reduce(hcat, collect.(mqs))
    hqs = reduce(hcat, collect.(hqs))

    p = plot(; kwargs...)
    for state_idx in 1:num_states
        label = no_label ? "" : labels[state_idx]
        plot!(mqs[:, state_idx]; ribbon=(mqs[:, state_idx] - lqs[:, state_idx], hqs[:, state_idx] - mqs[:, state_idx]), label=label)
    end

    return p
end
