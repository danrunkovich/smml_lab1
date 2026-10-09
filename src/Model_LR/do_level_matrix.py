import json
import os
import numpy as np
import pandas as pd


def main() -> None:
    raw_json_path = "../../dataset/audi_a6_dataset_input.json"
    if not os.path.exists(raw_json_path):
        print (f"ОШИБКА --> файл {raw_json_path} не найден")
        return

    transformed_json_path = "../../dataset/audi_a6_dataset_transformed_log_target.json"
    if not os.path.exists(transformed_json_path):
        print (f"ОШИБКА --> файл {raw_json_path} не найден")
        return

    print(f"1. Чтение сырого датасета: {raw_json_path}")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_df = pd.DataFrame(raw_data)

    print(f"2. Чтение трансформированного датасета: {transformed_json_path}")
    with open(transformed_json_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)
    df = pd.DataFrame(trans_data)

    target_col = "log_price"
    if target_col not in df.columns:
        print(f"ОШИБКА: Целевая колонка '{target_col}' не найдена!")
        return


    min_year = int(raw_df["year"].min())
    max_year = int(raw_df["year"].max())
    num_years = max_year - min_year + 1
    step_year = 1.0 / num_years

    step_year_sq = step_year

    num_volumes = max(len(raw_df["engine_vol_l"].dropna().unique()), 5)
    step_vol = 1.0 / num_volumes

    num_powers = max(len(raw_df["power_hp"].dropna().unique()), 10)
    step_power = 1.0 / num_powers

    min_km = float(raw_df["mileage_km"].min())
    max_km = float(raw_df["mileage_km"].max())
    km_range = max(max_km - min_km, 100_000)
    num_mileage_steps = max(int(km_range / 10_000), 10)
    step_mileage = 1.0 / num_mileage_steps

    # Справочник шагов для непрерывных фичей
    numeric_steps = {
        "scaled_year": (step_year, num_years),
        "scaled_year_sq": (step_year_sq, num_years),
        "scaled_mileage": (step_mileage, num_mileage_steps),
        "scaled_power": (step_power, num_powers),
        "scaled_engine_vol": (step_vol, num_volumes),
    }


    def quantize_to_level(value: float, step: float) -> float:
        val = np.clip(value, 0.0, 1.0)
        idx = int(np.floor(val / step))
        # Ограничиваем сверху, чтобы точка 1.0 не выпадала за сетку
        return round(idx * step, 4)

    X_quantized = pd.DataFrame(index=df.index)

    for col_name, (step, total_steps) in numeric_steps.items():
        if col_name in df.columns:
            X_quantized[col_name] = df[col_name].apply(
                lambda val: quantize_to_level(val, step)
            )

    ignore_cols = set(numeric_steps.keys()) | {"url", "price_rub", target_col}
    cat_cols = [c for c in df.columns if c not in ignore_cols]

    for col_name in cat_cols:
        X_quantized[col_name] = df[col_name].astype(int)


    X_quantized.insert(0, "x0_const", 1.0)


    print("\n" + "=" * 80)
    print("                    ПАСПОРТ УРОВНЕЙ ФАКТОРОВ МОДЕЛИ")
    print("=" * 80)
    print(
        f"{'Фактор':<22} | {'Тип':<12} | {'Число уровней':<15} | {'Шаг в [0,1]':<15}"
    )
    print("-" * 80)

    for col, (step, count) in numeric_steps.items():
        if col in df.columns:
            print(f"{col:<22} | числовой     | {count:<15} | {step:.5f}")

    for col in cat_cols:
        print(f"{col:<22} | OHE-бинарный | 2 (уровни 0, 1)| 1")
    print("=" * 80)


    output_df = pd.concat(
        [df[["url", "price_rub", target_col]], X_quantized], axis=1
    )

    output_json = "../../dataset/audi_a6_design_matrix.json"
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    output_df.to_json(
        output_json, orient="records", force_ascii=False, indent=2
    )

    print(f"\nИтоговая матрица планирования успешно создана!")
    print(f"Размерность матрицы X: {X_quantized.shape}")
    print(f"JSON-файл сохранен: {output_json}")

    output_excel = "./matrix.xlsx"
    output_df.to_excel(output_excel, index=False)
    print(f"Excel-файл успешно сохранен: {output_excel}")

    print("\nПервые 3 строки полученной расширенной матрицы планирования X:")
    print(X_quantized.iloc[:3, :8])


if __name__ == "__main__":
    main()