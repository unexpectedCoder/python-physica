# Подключение необходимых пакетов
from pathlib import Path

import numpy as np
import pandas as pd
import pint
import pint_pandas
from scipy import stats
from scipy.constants import c, e, h
from uncertainties import ufloat

# Настройка единиц измерения
ureg = pint.UnitRegistry(fmt_locale="ru_RU.UTF-8")  # русский язык единиц измерения
ureg.load_definitions("units.txt")  # загрузка пользовательских единиц измерения
ureg.formatter.default_format = "~"  # отображение сокращённых единиц измерения
pint_pandas.PintType.ureg = ureg  # настройка единиц измерения для pandas

# Директории
measurements_dir = Path("measurements")
output_dir = Path(".")

# Чтение данных (измерений)
data = pd.read_csv(
    measurements_dir / "435nm.csv", header=[0, 1], dtype=float
).pint.quantify(level=1)

# Приборная погрешность:
# в случае цифрового вольтметра - единица наименьшего разряда
voltmeter_error = 0.1 * ureg.V  # в данном случае - 0,1 В

# Массив напряжений из таблицы
voltages = data["Voltage"]

### РАСЧЁТЫ ###
# Среднее напряжение
mean_voltage = voltages.mean()
# Коэффициент Стьюдента t_student
confidence = 0.95
n = len(voltages)
t_student = stats.t.ppf((1 + confidence) / 2, n - 1)
# Случайная (статистическая) погрешность
# - std - standard deviation - стандартное (среднеквадратическое) отклонение
statistical_error = t_student * voltages.std() / np.sqrt(n)
# Полная погрешность
total_error = np.sqrt(statistical_error**2 + voltmeter_error**2)

# Задерживающее напряжение с учётом абсолютной ошибки
U_mag = ufloat(mean_voltage.magnitude, total_error.magnitude)
U = U_mag * ureg.V

# Вывод результатов в файл
fname = output_dir / "results-external-photoeffect.txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write(f"Задерживающее напряжение: U = {U}\n")
    f.write(f"  - среднее значение: {mean_voltage}\n")
    f.write(f"  - приборная погрешность: {voltmeter_error}\n")
    f.write(f"  - статистическая погрешность: {statistical_error}\n")
    f.write(f"  - полная погрешность: {total_error}\n")

# Константы
c = c * ureg("m/s")
h = h * ureg("J*s")
e = -e * ureg("C")
# Длина волны
wavelength = 435 * ureg("nm")
# Частота
freq = c / wavelength
# Работа выхода
A = h * freq - e * U

# Добавление итоговых результатов в файл
with open(fname, "a", encoding="utf-8") as f:
    f.write(f"\nЧастота фотона: ν = {freq.to('Hz'):~e}\n")
    f.write(f"Энергия фотона: E = {(h * freq).to('eV')}\n")
    f.write(f"Работа выхода: A = {A.to('eV')} при P = {confidence}\n")
