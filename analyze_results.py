from utils.plot_utils import (
    plot_catc,
    plot_catc_by_dataset,
    plot_familywise_pareto,
    plot_care_ai_efficiency,
    export_pareto_table,
    plot_carbon_consistency_per_dataset,
    plot_carbon_consistency_eaps_vs_codecarbon,
)


def main():

    RESULTS_CSV = "results/results.csv"
    CODECARBON_CSV = "results/codecarbon_results.csv"

    # ---------------- Global overview ----------------
    plot_catc(
        csv_path=RESULTS_CSV,
        out_path="plots/accuracy_carbon_overview.png",
    )

    # ---------------- CATC per dataset ----------------
    plot_catc_by_dataset(
        csv_path=RESULTS_CSV,
        out_dir="plots/catc",
    )

    # ---------------- Family-wise Pareto fronts ----------------
    plot_familywise_pareto(
        csv_path=RESULTS_CSV,
        out_dir="plots/family_pareto",
    )

    # ---------------- CARE-AI efficiency plots ----------------
    plot_care_ai_efficiency(
        csv_path=RESULTS_CSV,
        out_dir="plots/care_ai_efficiency",
    )

    # ---------------- Pareto tables (Appendix) ----------------
    export_pareto_table(
        csv_path=RESULTS_CSV,
        out_dir="tables/pareto",
    )

    # ---------------- Carbon consistency validation (optional) ----------------
    # NOTE:
    # These plots are provided for internal validation and are NOT part of the
    # core AI-CARE results reported in the paper. They are disabled by default
    # to keep the artifact strictly paper-aligned.

    # plot_carbon_consistency_per_dataset(
    #     eaps_csv=RESULTS_CSV,
    #     codecarbon_csv=CODECARBON_CSV,
    #     out_dir="figures/codecarbon_validation",
    # )

    # plot_carbon_consistency_eaps_vs_codecarbon(
    #     eaps_csv=RESULTS_CSV,
    #     codecarbon_csv=CODECARBON_CSV,
    #     out_dir="figures",
    # )

    print("✅ CATC, Pareto, and CARE-AI plots generated.")


if __name__ == "__main__":
    main()
