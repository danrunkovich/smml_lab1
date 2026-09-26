import json
import os
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_input.json"

    if not os.path.exists(json_path):
        json_path = "../dataset/audi_a6_dataset_input.json"
        if not os.path.exists(json_path):
            print(f"ОШИБКА: Файл {json_path} не найден!")
            return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    total_cars = len(df)
    print(f"Общее число тачек = {total_cars}\n")

    cat_cols = ["drive", "transmission", "fuel_type", "body_type"]

    print("-" * 10 + "Количество тачек определенной катгории для каждого категориального признака + процентное содержание" + "-" * 10)

    stats_dict = {}

    for col in cat_cols:
        clean_col = df[col].astype(str).str.strip().str.lower()
        counts = clean_col.value_counts()

        percentages = clean_col.value_counts(normalize=True) * 100

        summary_table = pd.DataFrame(
            {
                "Количество": counts,
                "Доля": percentages.round(2).map("{:.2f}%".format),
            }
        )

        stats_dict[col] = summary_table

        print(f"\nКатегориальная фича: {col.upper()} (всего значений: {counts.sum()})")
        print()
        print(summary_table.to_string())
        print()


if __name__ == "__main__":
    main()