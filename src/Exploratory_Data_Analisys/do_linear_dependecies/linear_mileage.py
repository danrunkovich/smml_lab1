import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    input_json = "../../../dataset/audi_a6_dataset_transformed.json"
    if not os.path.exists(input_json):
        print (f"ОЩШИБКА --> файл {input_json} не найден")
        return

    print(f"Загрузка данных из: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    target_col = "scaled_price"
    if target_col not in df.columns:
        print(f"ОШИБКА: Колонка {target_col} отсутствует в датасете!")
        return

    old_corr_year = df["scaled_year"].corr(df[target_col])
    old_corr_mileage = df["scaled_mileage"].corr(df[target_col])


    df["scaled_mileage"] = np.power(0.7, df["scaled_mileage"])
    df["scaled_year_sq"] = np.power(df["scaled_year"], 2)

    new_corr_year_lin = df["scaled_year"].corr(df[target_col])
    new_corr_year_sq = df["scaled_year_sq"].corr(df[target_col])
    new_corr_mileage = df["scaled_mileage"].corr(df[target_col])

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ ПРЕОБРАЗОВАНИЯ ФАКТОРОВ:")
    print(f"  Пробег scaled_mileage (0.5^x):      было {old_corr_mileage:+.4f}  -->  стало {new_corr_mileage:+.4f}")
    print(f"  Год scaled_year (1-я степень):      осталось {new_corr_year_lin:+.4f}")
    print(f"  Год scaled_year_sq (2-я степень):   НОВЫЙ ФАКТОР с корреляцией {new_corr_year_sq:+.4f}")
    print("=" * 70)


    output_json = "../../../dataset/audi_a6_dataset_transformed_linearized.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    df.to_json(output_json, orient="records", force_ascii=False, indent=2)
    print(f"\nНовый JSON успешно сохранен: {output_json}")


    plots_dir = "../plots/transformed_data_plots"
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    active_features = [col for col in df.columns if col not in {"url", "price_rub"}]
    corr_matrix = df[active_features].corr()

    sorted_idx = corr_matrix[target_col].sort_values(ascending=False).index
    sorted_corr_matrix = corr_matrix.loc[sorted_idx, sorted_idx]

    plt.figure(figsize=(13, 11))
    sns.heatmap(
        sorted_corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Матрица корреляций Пирсона (включая scaled_year, scaled_year_sq и 0.7^mileage)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    heatmap_file = os.path.join(plots_dir, "linearized_correlation_heatmap.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Матрица корреляций сохранена: {heatmap_file}")


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Зависимость scaled_price от преобразованных признаков",
        fontsize=14,
        fontweight="bold",
    )

    sns.regplot(
        data=df,
        x="scaled_year_sq",
        y=target_col,
        ax=axes[0],
        ci=None,
        scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 30},
        line_kws={"color": "crimson", "linewidth": 2.2},
    )
    axes[0].set_title(
        f"scaled_year_sq vs scaled_price\n(r = {new_corr_year_sq:+.3f})",
        fontsize=12,
    )
    axes[0].set_xlabel("scaled_year_sq (квадрат масштабированного года)")
    axes[0].set_ylabel("scaled_price")
    axes[0].set_ylim(-0.05, 1.05)

    sns.regplot(
        data=df,
        x="scaled_mileage",
        y=target_col,
        ax=axes[1],
        ci=None,  # Без доверительного интервала
        scatter_kws={"alpha": 0.45, "color": "#2ca02c", "s": 30},
        line_kws={"color": "crimson", "linewidth": 2.2},
    )
    axes[1].set_title(
        f"0.5^(scaled_mileage) vs scaled_price\n(r = {new_corr_mileage:+.3f})",
        fontsize=12,
    )
    axes[1].set_xlabel("0.5 ^ (scaled_mileage)")
    axes[1].set_ylabel("scaled_price")
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plots_file = os.path.join(plots_dir, "linearized_two_dependencies.png")
    plt.savefig(plots_file, dpi=300)
    plt.close()
    print(f"Графики двух зависимостей сохранены: {plots_file}")
    print("\nГотово! Датасет обновлен и визуализирован.")


if __name__ == "__main__":
    main()