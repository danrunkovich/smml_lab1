import json
import os


def main() -> None:
    json_path = "../../dataset/audi_a6_dataset_input.json"
    if not os.path.exists(json_path):
        print(f"ОШИБКА: Файл {json_path} не найден!")
        return

    print(f"Чтение исходных данных из: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_count = len(data)


    rare_categories = {"электро", "гибрид"}
    filtered_data = []
    removed_count = 0

    for item in data:
        fuel = str(item.get("fuel_type", "")).strip().lower()

        if fuel in rare_categories:
            removed_count += 1
            continue

        item["fuel_type"] = fuel

        item.pop("mileage", None)

        filtered_data.append(item)

    print(f"Было записей: {initial_count}")
    print(
        f"Удалено автомобилей (электро/гибрид): {removed_count} шт."
    )
    print(f"Осталось записей в выборке: {len(filtered_data)}")
    print("Поле 'mileage' успешно удалено из всех записей.")


    output_json_path = (
        "../../dataset/audi_a6_dataset_deleted_fuel_type+mileage.json"
    )
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"\nНовый JSON сохранен: {output_json_path}")


if __name__ == "__main__":
    main()