import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def main() -> None:
    json_path = "../../dataset/audi_a6_design_matrix.json"
    if not os.path.exists(json_path):
        print (f"ОШИБКА --> файл {json_path} не найден")
        return

    print(f"Загрузка данных из: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    target_col = "log_price"
    if target_col not in df.columns:
        print(f"ОШИБКА: Колонка {target_col} отсутствует в файле!")
        return

    y_data = df[target_col].dropna()


    mean_val = np.mean(y_data)
    std_val = np.std(y_data)


    jb_stat, jb_p_val = stats.jarque_bera(y_data)

    print("\n" + "=" * 65)
    print("ХАРАКТЕРИСТИКИ РАСПРЕДЕЛЕНИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:")
    print(f"  Среднее значение:         {mean_val:.4f}")
    print(f"  Стандартное отклонение:   {std_val:.4f}")
    print(f"  Тест Жарка-Бера p-value:  {jb_p_val:.4e}")
    if jb_p_val < 0.05:
        print(
            "  ВЫВОД: Распределение статистически значимо ОТЛИЧАЕТСЯ от нормального (p < 0.05)."
        )
    else:
        print(
            "  ВЫВОД: Нет оснований отвергать гипотезу о нормальности (p >= 0.05)."
        )

    print("=" * 65)


    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Анализ нормальности распределения целевого фактора ({target_col})",
        fontsize=15,
        fontweight="bold",
    )

    ax1 = axes[0]
    # Эмпирическая гистограмма
    sns.histplot(
        y_data,
        kde=True,
        ax=ax1,
        color="#1f77b4",
        stat="density",
        bins=25,
        alpha=0.5,
        label="Эмпирическая плотность (KDE)",
    )

    # Теоретическая кривая идеального нормального распределения Гаусса
    x_axis = np.linspace(y_data.min(), y_data.max(), 200)
    normal_curve = stats.norm.pdf(x_axis, mean_val, std_val)
    ax1.plot(
        x_axis,
        normal_curve,
        "r--",
        linewidth=2.2,
        label=f"Идеальная нормаль N({mean_val:.2f}, {std_val:.2f})",
    )

    ax1.set_title(
        f"Гистограмма распределения",
        fontsize=12,
    )
    ax1.set_xlabel(target_col)
    ax1.set_ylabel("Плотность вероятности")
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    # stats.probplot сопоставляет квантили данных с нормальным распределением
    res = stats.probplot(y_data, dist="norm", plot=ax2)

    # Кастомизация внешнего вида Q-Q plot
    ax2.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax2.get_lines()[0].set_markeredgecolor("none")
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[0].set_markersize(6)
    ax2.get_lines()[1].set_color("crimson")
    ax2.get_lines()[1].set_linewidth(2.2)

    ax2.set_title(
        "Q-Q Plot",
        fontsize=12,
    )
    ax2.set_xlabel("Теоретические квантили нормального распределения")
    ax2.set_ylabel(f"Фактические квантили ({target_col})")

    # Сохранение изображения
    plt.tight_layout()
    output_plot_path = os.path.join(plots_dir, "log_target_normality_qqplot.png")
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"\nГрафики успешно сохранены в: {output_plot_path}")


if __name__ == "__main__":
    main()