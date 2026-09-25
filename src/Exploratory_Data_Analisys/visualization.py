import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os



def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_final.json"
    if not os.path.exists(json_path):
        print(f"ERROR --> file {json_path} not found")
        return

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    data_frame = pd.DataFrame(data)
    os.makedirs("./plots", exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 10})

    # зависимости целевой величины (прайса) от числовых факторов
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    sns.regplot(
        data=data_frame,
        x="year",
        y="price_rub",
        ax=axes[0, 0],
        color="blue",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=None
    )
    axes[0, 0].set_title("Price (Year)")
    axes[0, 0].set_xlabel("год выпуска тачки")
    axes[0, 0].set_ylabel("цена (руб)")

    sns.regplot(
        data=data_frame,
        x="mileage_km",
        y="price_rub",
        ax=axes[0, 1],
        color="blue",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=None
    )
    axes[0, 1].set_title("Price (Mileage)")
    axes[0, 1].set_xlabel("пробег (км)")
    axes[0, 1].set_ylabel("цена (руб)")

    sns.regplot(
        data=data_frame,
        x="power_hp",
        y="price_rub",
        ax=axes[1, 0],
        color="blue",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=None
    )
    axes[1, 0].set_title("Price (Power HP)")
    axes[1, 0].set_xlabel("мощность двигателя (hp)")
    axes[1, 0].set_ylabel("цена (руб)")

    sns.regplot(
        data=data_frame,
        x="engine_vol_l",
        y="price_rub",
        ax=axes[1, 1],
        color="blue",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ci=None
    )
    axes[1, 1].set_title("Price (Engine Volume)")
    axes[1, 1].set_xlabel("объем двигателя (л)")
    axes[1, 1].set_ylabel("цена (руб)")

    plt.tight_layout()
    plt.savefig("./plots/numerical_dependencies.png", dpi=300)
    plt.show()
    plt.close()

    # зависимости целевой величины (прайса) от категориальных факторов
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    sns.boxplot(
        data=data_frame,
        x="drive",
        y="price_rub",
        ax=axes[0, 0],
        palette="Set2"
    )
    axes[0, 0].set_title("Price (Drive)")
    axes[0, 0].set_xlabel("привод")
    axes[0, 0].set_ylabel("цена (руб)")

    sns.boxplot(
        data=data_frame,
        x="transmission",
        y="price_rub",
        ax=axes[0, 1],
        palette="Set3"
    )
    axes[0, 1].set_title("Price (Transmission)")
    axes[0, 1].set_xlabel("коробка передач")
    axes[0, 1].set_ylabel("цена (руб)")

    sns.boxplot(
        data=data_frame,
        x="fuel_type",
        y="price_rub",
        ax=axes[1, 0],
        palette="Pastel1"
    )
    axes[1, 0].set_title("Price (Fuel Type)")
    axes[1, 0].set_xlabel("тип топлива")
    axes[1, 0].set_ylabel("цена (руб)")

    sns.boxplot(
        data=data_frame,
        x="body_type",
        y="price_rub",
        ax=axes[1, 1],
        palette="Pastel2"
    )
    axes[1, 1].set_title("Price (Body Type)")
    axes[1, 1].set_xlabel("тип кузова")
    axes[1, 1].set_ylabel("цена (руб)")

    plt.tight_layout()
    plt.savefig("./plots/categorical_dependencies.png", dpi=300)
    plt.show()
    plt.close()

    # heatmap для корелляций факторов и целевой величины
    plt.figure(figsize=(12, 12))
    numeric_cols = data_frame.select_dtypes(include = ["float64", "int64"]).columns
    corr = data_frame[numeric_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Pearson's Correlation Matrix (numerical factors & price_rub)")
    plt.tight_layout()
    plt.show()
    plt.savefig("./plots/heatmap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()