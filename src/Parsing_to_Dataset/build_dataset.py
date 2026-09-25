import json
import os
import pandas as pd

json_path = "../../dataset/audi_a6_dataset_final.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

for idx, row in df.iterrows():
    year_str = str(row["year"])
    km_str = str(row["mileage_km"])
    if km_str.startswith(year_str) and len(km_str) > 5:
        cleaned_km = km_str[len(year_str) :]
        df.at[idx, "mileage_km"] = int(cleaned_km) if cleaned_km else 0

# Удаляем возможные дубли по URL
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

# Упорядочим колонки для красивого отчета
columns_order = [
    "url",
    "year",
    "price_rub",
    "mileage_km",
    "power_hp",
    "engine_vol_l",
    "drive",
    "transmission",
    "fuel_type",
    "body_type",
]
df = df[columns_order]

# Сохраняем в Excel
output_excel = "../../dataset/Audi_A6_dataset_final.xlsx"
df.to_excel(output_excel, index=False)

print("=" * 60)
print(f"ИТОГО В ДАТАСЕТЕ: {len(df)} уникальных строк.")
print(f"Файл успешно сохранен: {output_excel}")
print("=" * 60)
print("\nПример первых 5 строк:")
print(
    df[
        [
            "year",
            "price_rub",
            "mileage_km",
            "engine_vol_l",
            "drive",
            "transmission",
        ]
    ].head(5)
)