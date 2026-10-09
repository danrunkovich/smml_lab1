import json
import os
import numpy as np
import pandas as pd


def main() -> None:
    input_json = "../../dataset/audi_a6_design_matrix.json"
    if not os.path.exists(input_json):
        print(f"ОШИБКА: Файл {input_json} не найден!")
        return

    print(f"Загрузка плана эксперимента из: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    df_plan = pd.DataFrame(plan_data)


    service_cols = {
        "price_rub",
        "log_price",
        "url",
    }
    feature_cols = [c for c in df_plan.columns if c not in service_cols]

    # Гарантируем, что x0_const идет первым
    if "x0_const" in feature_cols:
        feature_cols.remove("x0_const")
        feature_cols = ["x0_const"] + feature_cols

    # Матрица планирования X: размерность (N x (p + 1))
    X = df_plan[feature_cols].values.astype(float)

    # Вектор откликов Y_bar: размерность (N x 1)
    Y_bar = df_plan["log_price"].values.astype(float)

    N_samples, n_params = X.shape
    p_factors = n_params - 1

    print(f"Размерность матрицы планирования X: {X.shape} (N = {N_samples}, параметров = {n_params})")
    print(f"Размерность вектора откликов Y_bar: {Y_bar.shape}")


    XtX = X.T @ X
    XtY = X.T @ Y_bar

    # Проверка числа обусловленности
    cond_num = np.linalg.cond(XtX)

    # Аналитическое решение СНУ
    try:
        B = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        print("Внимание: матрица (X^T * X) вырождена, используется псевдообращение (pinv).")
        B = np.linalg.pinv(XtX) @ XtY

    B_flat = B.flatten()


    # Прогноз модели: Y_hat = X * B
    Y_pred = X @ B

    # Остатки модели на усредненных откликах
    residuals = Y_bar - Y_pred

    # Суммы квадратов
    ss_res = float(np.sum(residuals ** 2))
    y_mean = float(np.mean(Y_bar))
    ss_tot = float(np.sum((Y_bar - y_mean) ** 2))

    # R^2
    r_squared = 1.0 - (ss_res / ss_tot)

    # Скорректированный R^2 (adj)
    df_tot = N_samples - 1
    df_res = N_samples - n_params
    if df_res > 0:
        r_squared_adj = 1.0 - (1.0 - r_squared) * (df_tot / df_res)
    else:
        r_squared_adj = r_squared


    print("\n" + "=" * 80)
    print("                     РЕЗУЛЬТАТЫ РАСЧЕТА МОДЕЛИ ПО СНУ")
    print("=" * 80)
    print(f"  Коэффициент детерминации R^2:            {r_squared:.4f} ({r_squared * 100:.2f}%)")
    print(f"  Скорректированный R^2 (adj):            {r_squared_adj:.4f} ({r_squared_adj * 100:.2f}%)")
    print(f"  Число обусловленности матрицы СНУ:      {cond_num:.2f}")
    print(f"  Сумма квадратов остатков (SS_res):      {ss_res:.4f}")
    print(f"  Общая сумма квадратов (SS_tot):         {ss_tot:.4f}")
    print("-" * 80)

    print("ВЕКТОР-СТОЛБЕЦ ОЦЕНОК РЕГРЕССИОННЫХ КОЭФФИЦИЕНТОВ B:")
    coeff_list = []
    for idx, (col_name, b_val) in enumerate(zip(feature_cols, B_flat)):
        coeff_entry = {
            "index": idx,
            "feature": col_name,
            "coefficient_b": round(float(b_val), 6)
        }
        coeff_list.append(coeff_entry)
        print(f"  b_{idx:<2} ({col_name:<25}) = {b_val:+.6f}")
    print("=" * 80)

    # Вывод аналитического уравнения модели
    print("\nМАТЕМАТИЧЕСКОЕ УРАВНЕНИЕ МОДЕЛИ:")
    eq_terms = [f"{B_flat[0]:+.4f}"]
    for col_name, b_val in zip(feature_cols[1:], B_flat[1:]):
        eq_terms.append(f"{b_val:+.4f} * {col_name}")
    print("  ln(Price) = " + " ".join(eq_terms))
    print("=" * 80)


    output_data = {
        "metrics": {
            "r_squared": round(r_squared, 6),
            "r_squared_adj": round(r_squared_adj, 6),
            "condition_number": round(cond_num, 4),
            "ss_res": round(ss_res, 6),
            "ss_tot": round(ss_tot, 6),
            "n_unique_points": N_samples,
            "p_parameters": n_params
        },
        "coefficients_table": coeff_list,
        "vector_B": [round(float(x), 6) for x in B_flat]
    }

    output_json = "./model_coefficients.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты и вектор B успешно сохранены в:")
    print(f"-> {output_json}")


if __name__ == "__main__":
    main()