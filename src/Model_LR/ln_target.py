import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    input_json = "../../dataset/audi_a6_dataset_transformed_linearized.json"
    if not os.path.exists(input_json):
        print(f"ОШИБКА: Файл {input_json} не найден!")
        return

    print(f"Загрузка данных из: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if "price_rub" not in df.columns:
        print("ОШИБКА: Колонка 'price_rub' отсутствует в датасете!")
        return


    raw_price = df["price_rub"].astype(float)
    #raw_price = np.clip(raw_price, 100_000, None)

    log_price = np.log(raw_price)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_log_price"] = scaler.fit_transform(
        log_price.values.reshape(-1, 1)
    ).flatten()

    had_old_target = "scaled_price" in df.columns
    if had_old_target:
        old_corr_year = df["scaled_year"].corr(df["scaled_price"])
        new_corr_year = df["scaled_year"].corr(df["scaled_log_price"])

        old_corr_mileage = df["scaled_mileage"].corr(df["scaled_price"])
        new_corr_mileage = df["scaled_mileage"].corr(df["scaled_log_price"])

        print("\n" + "=" * 65)
        print("СРАВНЕНИЕ КОРРЕЛЯЦИЙ (Старый scaled_price vs Новый scaled_log_price):")
        print(f"  scaled_year:    было {old_corr_year:+.4f}  -->  стало {new_corr_year:+.4f}")
        print(f"  scaled_mileage: было {old_corr_mileage:+.4f}  -->  стало {new_corr_mileage:+.4f}")
        print("=" * 65)

        df = df.drop(columns=["scaled_price"])

    target_col = "scaled_log_price"


    output_json = "../../dataset/audi_a6_dataset_transformed_log_target.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    df.to_json(output_json, orient="records", force_ascii=False, indent=2)
    print(f"\nНовый датасет сохранен: {output_json}")


    plots_dir = "./plots"
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
        f"Матрица корреляций с логарифмированным таргетом ({target_col})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    heatmap_file = os.path.join(plots_dir, "log_target_correlation_heatmap.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Матрица корреляций сохранена: {heatmap_file}")


    feature_cols = [col for col in active_features if col != target_col]
    num_features = len(feature_cols)

    n_cols = 4
    n_rows = int(np.ceil(num_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    fig.suptitle(
        f"Зависимость нового таргета ({target_col}) от всех признаков (МНК-тренд без CI)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    axes_flat = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes_flat[idx]


        sns.regplot(
            data=df,
            x=feature,
            y=target_col,
            ax=ax,
            ci=None,
            scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 25},
            line_kws={"color": "crimson", "linewidth": 2.2},
        )

        corr_val = df[feature].corr(df[target_col])
        ax.set_title(f"{feature}\n(r = {corr_val:+.3f})", fontsize=11)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(target_col, fontsize=10)
        ax.set_ylim(-0.05, 1.05)



    # Скрываем неиспользованные пустые ячейки в сетке
    for extra_idx in range(num_features, len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])

    plt.tight_layout()
    trends_file = os.path.join(plots_dir, "all_features_vs_log_price_trends.png")
    plt.savefig(trends_file, dpi=300)
    plt.close()
    print(f"Сетка графиков зависимостей сохранена: {trends_file}")

    print("\n" + "=" * 65)
    print("Готово! Новый таргет scaled_log_price успешно рассчитан и визуализирован.")
    print("=" * 65)


if __name__ == "__main__":
    main()