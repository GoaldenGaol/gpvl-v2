"""Fertility forecast: TFR(t+18) = f(dim4(t))."""

from volition.data.dim4 import load_countries, validation_stats
from volition.regime import classify_regime


def main() -> None:
    df = load_countries()
    stats = validation_stats(df)

    print("dim4 → TFR Fertility Forecast (frozen 2023)")
    print(f"Pearson r = {stats['pearson_r']:.3f}, R² = {stats['r_squared']:.3f}")
    print(f"Countries: {stats['n_countries']}\n")

    top = df.nlargest(5, "dim4")
    for _, row in top.iterrows():
        regime = classify_regime(row["dim4"]).name
        print(
            f"{row['country']:20s}  dim4={row['dim4']:.2f}  "
            f"TFR={row['tfr_future']:.2f}  regime={regime}"
        )


if __name__ == "__main__":
    main()