"""Run the Poincaré rotational-transform analysis for the Zot80 trace."""

if __package__:
    from .compute_iota_v3_poincare import main
else:
    from compute_iota_v3_poincare import main


if __name__ == "__main__":
    main("trace_zot80_output")
