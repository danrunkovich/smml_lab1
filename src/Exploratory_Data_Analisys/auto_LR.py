import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_transformed_without_mileage_log_target.json"
    if not os.path.exists(json_path):
        print (f"Ошибка --> файл {json_path} не найден")
        return

    print(f"Загрузка данных из: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    target_col = "log_price"
    ignore_cols = {"url", "price_rub", target_col, "scaled_price"}

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    X = df[feature_cols].copy()

    y = df[target_col].copy()

    # sm.add_constant добавляет столбец 'const' = 1.0 в начало матрицы X
    X_with_const = sm.add_constant(X)



    model = sm.OLS(y, X_with_const)
    results = model.fit()


    # e_i = y_i - y_hat_i
    y_pred = results.fittedvalues
    residuals = results.resid  # Вектор остатков e

    # Добавляем остатки и предсказания в датафрейм для анализа
    df["y_predicted"] = y_pred
    df["residual_error"] = residuals


    print('-' * 20 + "ОТЧЕТ О ЛИНЕЙНОЙ РЕГРЕССИИ" + '-' * 20)
    # Метрики качества модели
    r2 = results.rsquared

    print(f"Коэффициент детерминации R^2: {r2:.4f} ({r2*100:.2f}%)")
    print("-" * 80)

    # Таблица коэффициентов модели
    print("ТАБЛИЦА ВЕСОВЫХ КОЭФФИЦИЕНТОВ МОДЕЛИ:")
    coeff_summary = pd.DataFrame(
        {
            "Коэффициент (Вес)": results.params,
            "Станд. ошибка": results.bse,
            "t-критерий": results.tvalues,
            "p-value": results.pvalues,
            "Значимость (p < 0.05)": results.pvalues < 0.05,
        }
    )
    print(coeff_summary.to_string())
    print("-" * 80)

    # Анализ вектора остатков (ошибок)
    print("СТАТИСТИКА СТОЛБЦА ОШИБОК (ОСТАТКОВ e = y - y_hat):")
    print(f"Среднее значение остатков: {residuals.mean():.6e} (теоретически строго 0)")
    print(f"Стандартное отклонение остатков (S_e): {residuals.std():.4f}")
    print(f"Минимальная ошибка (недопрогноз): {residuals.min():.4f}")
    print(f"Максимальная ошибка (перепрогноз): {residuals.max():.4f}")
    print("-" * 80)


    # Сохраняем датасет с вычисленными ошибками для построения графиков
    output_with_residuals = "../../dataset/audi_a6_dataset_transformed_without_mileage_linearized_with_residuals.json"
    df.to_json(
        output_with_residuals, orient="records", force_ascii=False, indent=2
    )
    print(
        f"\nФайл с добавленным столбцом ошибок сохранен: {output_with_residuals}"
    )


if __name__ == "__main__":
    main()