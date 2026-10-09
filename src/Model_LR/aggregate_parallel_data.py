import json
import os
import numpy as np
import pandas as pd


def main() -> None:

    input_json = "../../dataset/audi_a6_design_matrix.json"
    if not os.path.exists(input_json):
        print(f"ОШИБКА: Файл {input_json} не найден!")
        return

    print(f"Загрузка матрицы планирования из: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)


    ignore_cols = {"url", "price_rub", "scaled_log_price"}
    factor_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Число факторов в строке матрицы X: {len(factor_cols)}")
    print(f"Всего исходных наблюдений (автомобилей): {len(df)}")


    grouped = df.groupby(factor_cols, as_index=False)

    plan_rows = []

    for _, group in grouped:
        factors_dict = group.iloc[0][factor_cols].to_dict()

        raw_prices = group["price_rub"].dropna().astype(float).values

        #raw_prices = np.clip(raw_prices, 100_000, None)

        log_prices = np.log(raw_prices)
        mean_log_y = float(np.mean(log_prices))

        # выборочная дисперсия параллельных замеров S_i^2
        m_reps = len(log_prices)
        var_log_y = (
            float(np.var(log_prices, ddof=1)) if m_reps > 1 else 0.0
        )

        urls = group["url"].tolist()

        row_entry = {
            **factors_dict,
            "m_replicates": m_reps,  # Число параллельных опытов (машин в наборе)
            "raw_prices": list(raw_prices),  # Исходные цены в рублях
            "log_prices": [
                round(float(x), 4) for x in log_prices
            ],  # Строка матрицы результатов Y
            "mean_log_price": round(
                mean_log_y, 4
            ),  # Элемент вектора откликов Y_bar
            "var_log_price": round(
                var_log_y, 6
            ),  # Дисперсия внутри набора S_i^2
            "urls": urls,
        }
        plan_rows.append(row_entry)


    plan_df = pd.DataFrame(plan_rows)
    N_unique = len(plan_df)
    total_samples = plan_df["m_replicates"].sum()

    print("\n" + "=" * 75)
    print("           РЕЗУЛЬТАТЫ СБОРКИ МАТРИЦЫ ПЛАНИРОВАНИЯ И ОТКЛИКОВ")
    print("=" * 75)
    print(f"  Количество уникальных наборов (строк матрицы X, N):  {N_unique}")
    print(f"  Общее число значений в матрице результатов Y:         {total_samples}")
    print(f"  Минимальное число машин в наборе:                    {plan_df['m_replicates'].min()}")
    print(f"  Максимальное число машин в наборе:                   {plan_df['m_replicates'].max()}")
    print(f"  Среднее число дубликатов на набор:                   {plan_df['m_replicates'].mean():.2f}")
    print("-" * 75)

    # Распределение числа параллельных данных по наборам
    reps_dist = plan_df["m_replicates"].value_counts().sort_index()
    print("Распределение параллельных данных по числу машин в наборе:")
    for m_val, count in reps_dist.items():
        print(f"  Наборов с {m_val} машинами: {count}")
    print("=" * 75)

    # Пример первых двух наборов
    print("\nПример первых 2 уникальных наборов (строк плана):")
    for i in range(min(2, len(plan_df))):
        sample = plan_df.iloc[i]
        print(f"\nНабор #{i + 1}:")
        print(f"  Год: {sample.get('scaled_year')}, Пробег: {sample.get('scaled_mileage')}, Мощность: {sample.get('scaled_power')}")
        print(f"  Число параллельных машин (m): {sample['m_replicates']}")
        print(f"  Исходные цены: {sample['raw_prices']}")
        print(f"  Логарифмы цен: {sample['log_prices']}")
        print(f"  Усредненный логарифм цены (Y_bar): {sample['mean_log_price']}")


    output_json = "../../dataset/audi_a6_grouped_plan.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(plan_rows, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"Новый агрегированный датасет сохранен: {output_json}")
    print("=" * 75)


if __name__ == "__main__":
    main()