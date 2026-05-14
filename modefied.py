import os
import random
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------

FOLDER = "compressor_train"

MIN_LEN = 5
MAX_LEN = 10

SPIKE_PROB = 0.18
DROP_PROB = 0.12

# ----------------------------------------


def parse_power(power_str):
    try:
        return [float(x) for x in str(power_str).split(",")]
    except:
        return None


def build_growth_patterns(label1_rows):
    """
    Собираем реальные приросты из label=1
    """
    growths = []

    for p in label1_rows["power"]:
        arr = parse_power(p)

        if arr is None or len(arr) < 2:
            continue

        diffs = np.diff(arr)

        # берем только нормальные возрастающие участки
        for d in diffs:
            if -10 <= d <= 20:
                growths.append(float(d))

    # fallback
    if len(growths) == 0:
        growths = [0.5, 1, 1.5, 2, 2.5]

    return growths


def generate_power_sequence(last_value, growths):
    """
    Генерация возрастающей строки power,
    где последнее значение = исходному
    """

    length = random.randint(MIN_LEN, MAX_LEN)

    values = [last_value]

    # строим назад от последнего значения
    for _ in range(length - 1):
        step = random.choice(growths)

        # иногда шум
        step += random.uniform(-1.0, 1.0)

        prev = values[0] - step

        # защита от отрицательных
        prev = max(prev, 0)

        values.insert(0, round(prev, 1))

    # иногда делаем жесткий пик
    if random.random() < SPIKE_PROB:
        idx = random.randint(0, len(values) - 1)
        values[idx] = round(random.uniform(100, 200), 1)

    # иногда делаем жесткое падение
    if random.random() < DROP_PROB:
        idx = random.randint(0, len(values) - 1)
        values[idx] = round(random.uniform(15, 30), 1)

    # отрицательных быть не должно
    values = [max(0, round(v, 1)) for v in values]

    return ",".join(map(str, values))


for filename in os.listdir(FOLDER):

    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(FOLDER, filename)

    try:
        df = pd.read_csv(filepath)

        # гарантируем int
        df["label"] = df["label"].astype(int)
        df["model"] = df["model"].astype(str)

        label1_count = (df["label"] == 1).sum()

        mask_999_0 = (
            (df["model"] == "999") &
            (df["label"] == 0)
        )

        df_999 = df[mask_999_0].copy()

        target_count = max(1, label1_count // 2)

        # ----------------------------------------
        # уменьшаем
        # ----------------------------------------

        if len(df_999) > target_count:
            df_999 = df_999.sample(target_count, random_state=42)

        # ----------------------------------------
        # увеличиваем
        # ----------------------------------------

        elif len(df_999) < target_count:

            need = target_count - len(df_999)

            if len(df_999) > 0:
                extra = df_999.sample(
                    need,
                    replace=True,
                    random_state=42
                ).copy()

                df_999 = pd.concat([df_999, extra], ignore_index=True)

        # ----------------------------------------
        # реальные паттерны роста из label=1
        # ----------------------------------------

        label1_rows = df[df["label"] == 1]

        growths = build_growth_patterns(label1_rows)

        # ----------------------------------------
        # генерируем новые power
        # ----------------------------------------

        for idx in df_999.index:

            try:
                last_value = float(df_999.at[idx, "power"])
            except:
                last_value = random.uniform(70, 120)

            new_power = generate_power_sequence(
                last_value,
                growths
            )

            df_999.at[idx, "power"] = new_power

        # ----------------------------------------
        # удаляем старые 999/0
        # ----------------------------------------

        df = df[~mask_999_0]

        # добавляем новые
        df = pd.concat([df, df_999], ignore_index=True)

        # сохраняем
        df.to_csv(filepath, index=False)

        print(
            f"{filename} | label1={label1_count} | "
            f"target999={target_count} | final999={len(df_999)}"
        )

    except Exception as e:
        print(f"Ошибка в {filename}: {e}")