using CSV
using DataFrames
using Plots

function make_title(filename::String)
    name = splitext(filename)[1]
    parts = split(name, "_") # Breaks the string into pieces

    prefix = parts[1]

    skip_keys = Set(["n", "m", "S", "n_perm", "perm"]) # keys to ignore

    values = String[]
    for p in parts
        if occursin("=", p)
            key, val = split(p, "=")
            if !(key in skip_keys)
                push!(values, val)
            end
        end
    end

    return isempty(values) ? prefix : prefix * "_" * join(values, "_")
end

function make_plots(; 
    input_dir = joinpath(pwd(), "values", "fp_tp"),
    output_dir = joinpath(pwd(), "plots", "fp_tp")
)

    mkpath(output_dir)

    # readdir: returns list of file and folders in that folder as strings.
    # f -> endswith(lowercase(f), ".csv"): for a file name (string) f, return true iff ends with .csv
    # filter: keeps only the elements of a collection that satisfy a condition
    csv_files = filter(f -> endswith(lowercase(f), ".csv"), readdir(input_dir))

    for file in csv_files
        lower_file = lowercase(file) # Converts the current filename to lowercase
        filepath = joinpath(input_dir, file)

        # Determine prefix (fp or tp)
        prefix =
            startswith(lower_file, "tp") ? "tp" :
            startswith(lower_file, "fp") ? "fp" :
            nothing

        if prefix === nothing
            println("Skipping $file (does not start with tp or fp)")
            continue
        end

        ylabel = prefix == "tp" ? "True Positive Rate" : "False Positive Rate"

        df = CSV.read(filepath, DataFrame)

        # Required columns based on prefix
        theta_col = "θs"
        hipm_col  = "$(prefix)_hipm"
        wow_col   = "$(prefix)_wow"

        # Check columns exist
        if !(theta_col in names(df) &&
             hipm_col in names(df) &&
             wow_col in names(df))
            println("Skipping $file (missing required columns)")
            continue
        end

        # Plot
        p = plot(
        df[!, theta_col],
        df[!, hipm_col],
        label = "hipm",
        linewidth = 2,
        marker = :circle,
        xlabel = "θ",
        ylabel = ylabel,
        title = make_title(file),
        legend = :best,
        aspect_ratio = :equal,
        xlims = (0, 1),
        ylims = (0, 1)
    )

        plot!(
            p,
            df[!, theta_col],
            df[!, wow_col],
            label = "wow",
            linewidth = 2,
            marker = :square
        )

        # Save
        # splitext(file)[1]: splits file name with two elements one before "." and one after (including ".").
        output_path = joinpath(output_dir, splitext(file)[1] * ".png")
        savefig(p, output_path)

        println("Saved: $output_path")
    end
end

# Run
make_plots()