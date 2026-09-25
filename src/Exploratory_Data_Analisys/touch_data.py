import json
import os
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_input.json"

    if not os.path.exists(json_path):
        print(f"ERROR --> file {json_path} not found")
        return

    print(f"Загрузка данных из {json_path}")
    with open(json_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    df = pd.DataFrame(raw_data)

    for col in ["drive", "transmission", "fuel_type", "body_type"]:
        df[col] = df[col].astype(str).str.strip().str.lower()


    # масштабирование числовых факторов при помощи MinMaxScaler получаем крч диапозон значений в [0,1]
    num_cols = ["year", "mileage_km", "power_hp", "engine_vol_l", "price_rub"]
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_numeric = scaler.fit_transform(df[num_cols])
    scaled_num_df = pd.DataFrame(
        scaled_numeric,
        columns=[
            "scaled_year",
            "scaled_mileage",
            "scaled_power",
            "scaled_engine_vol",
            "scaled_price",
        ],
    )


    # преобразование при помощи One-Hot-Encoding для категориальных факторов в числовые факторы
    categories_spec = [
        ["полный", "передний"], # drive
        ["вариатор", "автомат", "механика", "робот"],  # transmission
        ["дизель", "бензин", "электро", "гибрид"],  # fuel_type
        ["универсал", "седан"],  # body_type
    ]

    cat_cols = ["drive", "transmission", "fuel_type", "body_type"]

    ohe = OneHotEncoder(
        categories=categories_spec,
        sparse_output=False,
        handle_unknown="ignore"
    )

    ohe_matrix = ohe.fit_transform(df[cat_cols])

    # генерация названий новых колонок
    ohe_feature_names = []
    for col_name, cats in zip(cat_cols, categories_spec):
        for cat in cats:
            ohe_feature_names.append(f"{col_name}_{cat}")

    ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_feature_names, dtype=int)


    transformed_df = pd.concat([scaled_num_df, ohe_df], axis=1)

    json_output_df = pd.concat(
        [df[["url", "price_rub"]], transformed_df], axis=1
    )

    output_json_path = "../../dataset/audi_a6_transformed.json"
    json_output_df.to_json(
        output_json_path, orient="records", force_ascii=False, indent=2
    )
    print(f"Новый JSON успешно сохранен: {output_json_path}")


    excel_path = "../../dataset/Audi_A6_dataset.xlsx"

    if os.path.exists(excel_path):
        print(f"Обновление Excel файла: {excel_path}")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active

        START_COL = 13

        for c_idx, col_name in enumerate(transformed_df.columns, start=START_COL):
            ws.cell(row=1, column=c_idx, value=col_name)

        for r_idx, row_values in enumerate(transformed_df.values, start=2):
            for c_idx, val in enumerate(row_values, start=START_COL):
                cell_val = round(float(val), 4) if isinstance(val, float) else int(val)
                ws.cell(row=r_idx, column=c_idx, value=cell_val)

        wb.save(excel_path)

    else:
        print(f"Предупреждение: Файл {excel_path} не найден. Создаем новый.")
        full_export = pd.concat([df, transformed_df], axis=1)
        full_export.to_excel(excel_path, index=False)


    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 10))
    corr_matrix = transformed_df.corr()

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Корреляционная матрица преобразованных признаков (Scaled + OHE (крч вообще все))")
    plt.tight_layout()
    plt.savefig("./plots/transformed_data_plots/transformed_correlation_heatmap.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Зависимость нормализованной цены от числовых факторов (в шкале [0, 1])",
        fontsize=14,
    )

    factors = [
        ("scaled_year", "Год выпуска (Scaled)", axes[0, 0], "blue"),
        ("scaled_mileage", "Пробег (Scaled)", axes[0, 1], "green"),
        ("scaled_power", "Мощность (Scaled)", axes[1, 0], "purple"),
        ("scaled_engine_vol", "Объем двигателя (Scaled)", axes[1, 1], "red"),
    ]

    for col, title, ax, color in factors:
        sns.regplot(
            data=transformed_df,
            x=col,
            y="scaled_price",
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.4},
            line_kws={"color": "black", "linewidth": 2},
            ci=None
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("scaled_price [0, 1]")

    plt.tight_layout()
    plt.savefig("./plots/transformed_data_plots/transformed_numerical_regression.png", dpi=300)
    plt.close()

    print("Все пушка все записано все скопировано газ проверять")


if __name__ == "__main__":
    main()