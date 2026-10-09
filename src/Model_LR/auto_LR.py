import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import statsmodels.api as sm


def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_transformed_log_target.json"
    if not os.path.exists(json_path):
        print(f"ОШИБКА: Файл {json_path} не найден!")
        return

    print(f"Загрузка данных из: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    target_col = "scaled_log_price"
    if target_col not in df.columns:
        print(f"ОШИБКА: Целевая колонка '{target_col}' отсутствует в файле!")
        return

    ignore_cols = {"url", "price_rub", target_col}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()


    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit()


    y_pred = results.fittedvalues
    residuals = results.resid  # Вектор e = y - y_hat

    df["y_predicted"] = y_pred
    df["residual_error"] = residuals


    print("\n" + "=" * 85)
    print("                РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ РЕГРЕССИИ ДЛЯ scaled_log_price")
    print("=" * 85)

    r2 = results.rsquared
    r2_adj = results.rsquared_adj
    f_stat = results.fvalue
    f_pvalue = results.f_pvalue

    print(f"  Коэффициент детерминации R^2:            {r2:.4f} ({r2*100:.2f}%)")
    print(f"  Скорректированный R^2 (adj):            {r2_adj:.4f} ({r2_adj*100:.2f}%)")
    print(f"  F-статистика Фишера:                    {f_stat:.2f} (p-value: {f_pvalue:.2e})")
    print(f"  Информационный критерий Акаике (AIC):   {results.aic:.2f}")
    print("-" * 85)

    print("ТАБЛИЦА ВЕСОВЫХ КОЭФФИЦИЕНТОВ МОДЕЛИ:")
    coeff_summary = pd.DataFrame({
        "Коэффициент w_j": results.params,
        "Станд. ошибка SE": results.bse,
        "t-критерий": results.tvalues,
        "p-value": results.pvalues,
        "Значимость (p < 0.05)": results.pvalues < 0.05
    })
    print(coeff_summary.round(5).to_string())
    print("-" * 85)

    # Статистика вектора ошибок
    e_mean = residuals.mean()
    e_std = residuals.std()
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    print("АНАЛИЗ СТОЛБЦА ОШИБОК (ОСТАТКОВ e = y - y_hat):")
    print(f"  Среднее значение остатков (Mean):       {e_mean:.6e} (теоретически строго 0)")
    print(f"  Стандартное отклонение ошибок (S_e):    {e_std:.4f}")
    print(f"  Тест Жака-Бера на нормальность ошибок:  JB = {jb_stat:.2f} (p-value: {jb_pvalue:.4e})")
    if jb_pvalue >= 0.05:
        print("  ВЫВОД ПО ОШИБКАМ: Ошибки распределены НОРМАЛЬНО (H0 не отвергается, p >= 0.05)!")
    else:
        print("  ВЫВОД ПО ОШИБКАМ: Присутствуют небольшие отклонения от строгого Гаусса (p < 0.05).")
    print("=" * 85)


    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Диагностический анализ остатков линейной регрессии (Residual Analysis)",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )

    ax1 = axes[0, 0]
    sns.scatterplot(
        x=y_pred,
        y=residuals,
        ax=ax1,
        color="#1f77b4",
        alpha=0.6,
        edgecolor="none",
        s=40
    )
    ax1.axhline(0, color="crimson", linestyle="--", linewidth=2.0)
    ax1.set_title("1. Остатки от предсказанных значений (e vs y_hat)\n(Проверка линейности и гетероскедастичности)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Предсказанные значения (y_hat)")
    ax1.set_ylabel("Остатки (e = y - y_hat)")

    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax2.get_lines()[0].set_markeredgecolor("none")
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color("crimson")
    ax2.get_lines()[1].set_linewidth(2.2)
    ax2.set_title("2. Q-Q Plot ошибок модели\n(Проверка нормальности распределения e)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Теоретические квантили N(0, 1)")
    ax2.set_ylabel("Фактические квантили остатков")

    ax3 = axes[1, 0]
    sns.histplot(
        residuals,
        kde=True,
        ax=ax3,
        color="#1f77b4",
        stat="density",
        bins=25,
        alpha=0.5,
        label="Эмпирическая плотность ошибок"
    )
    # Накладываем теоретический колокол Гаусса
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax3.plot(
        x_range,
        stats.norm.pdf(x_range, 0, e_std),
        "r--",
        linewidth=2.0,
        label=f"Идеальная нормаль N(0, {e_std:.3f})"
    )
    ax3.set_title("3. Плотность распределения остатков\n(Симметричность относительно нуля)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Величина ошибки (e)")
    ax3.set_ylabel("Плотность вероятности")
    ax3.legend(loc="upper right")

    # Scale-Location Plot (Стабильность разброса ошибок)
    ax4 = axes[1, 1]
    # Стандартизированные остатки: sqrt(|e / std|)
    norm_residuals = np.sqrt(np.abs(residuals / e_std))
    sns.scatterplot(
        x=y_pred,
        y=norm_residuals,
        ax=ax4,
        color="#2ca02c",
        alpha=0.6,
        edgecolor="none",
        s=40
    )
    sns.regplot(
        x=y_pred,
        y=norm_residuals,
        ax=ax4,
        scatter=False,
        ci=None,
        color="crimson",
        line_kws={"linewidth": 2.0}
    )
    ax4.set_title("4. Scale-Location Plot: sqrt(|e|) vs y_hat\n(Проверка постоянства дисперсии / гомоскедастичности)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Предсказанные значения (y_hat)")
    ax4.set_ylabel("sqrt(|Стандартизованный остаток|)")

    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "regression_residuals_diagnostics.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\nДиагностический постер ошибок сохранен: {plot_file}")

    output_res_json = "../../dataset/audi_a6_log_target_with_residuals.json"
    os.makedirs(os.path.dirname(output_res_json), exist_ok=True)
    df.to_json(output_res_json, orient="records", force_ascii=False, indent=2)
    print(f"Датасет с остатками сохранен: {output_res_json}")


if __name__ == "__main__":
    main()