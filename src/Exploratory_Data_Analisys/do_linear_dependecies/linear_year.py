import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    input_json = "../../../dataset/audi_a6_dataset_transformed_without_mileage.json"
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

    df["scaled_year_sq"] = np.power(df["scaled_year"], 2)

    new_corr_year_lin = df["scaled_year"].corr(df[target_col])
    new_corr_year_sq = df["scaled_year_sq"].corr(df[target_col])

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ ПРЕОБРАЗОВАНИЯ ФАКТОРОВ:")
    print(f"  Год scaled_year:      осталось {new_corr_year_lin:+.4f}")
    print(f"  Год scaled_year_sq:   НОВЫЙ ФАКТОР с корреляцией {new_corr_year_sq:+.4f}")
    print("=" * 70)


    output_json = "../../../dataset/audi_a6_dataset_transformed_without_mileage_linearized.json"
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
        "Матрица корреляций Пирсона",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    heatmap_file = os.path.join(plots_dir, "linearized_correlation_heatmap_without_mileage.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Матрица корреляций сохранена: {heatmap_file}")


    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(
        "Зависимость scaled_price от преобразованных признаков",
        fontsize=14,
        fontweight="bold",
    )

    sns.regplot(
        data=df,
        x="scaled_year_sq",
        y=target_col,
        ax=axes,
        ci=None,
        scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 30},
        line_kws={"color": "crimson", "linewidth": 2.2},
    )
    axes.set_title(
        f"scaled_year_sq vs scaled_price\n(r = {new_corr_year_sq:+.3f})",
        fontsize=12,
    )
    axes.set_xlabel("scaled_year_sq (квадрат масштабированного года)")
    axes.set_ylabel("scaled_price")
    axes.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plots_file = os.path.join(plots_dir, "linearized_scaled_year_square_dependency.png")
    plt.savefig(plots_file, dpi=300)
    plt.close()
    print(f"График зависимостеи сохранен: {plots_file}")
    print("\nГотово! Датасет обновлен и визуализирован.")


if __name__ == "__main__":
    main()