import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_transformed.json"
    if not os.path.exists(json_path):
        print(f"ОШИБКА: Файл {json_path} не найден!")
        return

    print(f"Загрузка данных из: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    target_col = "scaled_price"
    if target_col not in df.columns:
        print(f"ОШИБКА: Целевая колонка '{target_col}' отсутствует в датасете!")
        return

    ignore_cols = {"url", "price_rub", target_col}
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    num_features = len(feature_cols)
    print(f"Найдено факторов для анализа: {num_features}")


    n_cols = 4
    n_rows = int(np.ceil(num_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    fig.suptitle(
        "Линии тренда (МНК) целевого фактора (scaled_price) от всех признаков",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    axes_flat = axes.flatten()
    sns.set_theme(style="whitegrid")


    for idx, feature in enumerate(feature_cols):
        ax = axes_flat[idx]

        # Проверяем, бинарный ли признак (для One-Hot колонок)
        #is_binary = df[feature].nunique() <= 2
        #jitter = 0.04 if is_binary else 0.0

        # sns.regplot с ci=None убирает доверительный интервал
        sns.regplot(
            data=df,
            x=feature,
            y=target_col,
            ax=ax,
            ci=None,
            #x_jitter=jitter,  # Разброс по X для читаемости нулей и единиц
            scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 25},
            line_kws={"color": "crimson", "linewidth": 2.2},
        )

        # Расчет коэффициента корреляции Пирсона для заголовка
        corr_val = df[feature].corr(df[target_col])

        ax.set_title(f"{feature}\n(r = {corr_val:+.3f})", fontsize=11)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("scaled_price [0, 1]", fontsize=10)
        ax.set_ylim(-0.05, 1.05)

        #if is_binary:
            #ax.set_xticks([0, 1])

    # Скрываем пустые неиспользованные ячейки в сетке
    for extra_idx in range(num_features, len(axes_flat)):
        fig.delaxes(axes_flat[extra_idx])


    output_dir = "./plots/transformed_data_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_features_vs_target_trends.png")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("=" * 60)
    print(f"Все {num_features} графиков успешно построены и сохранены в:")
    print(f"-> {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()