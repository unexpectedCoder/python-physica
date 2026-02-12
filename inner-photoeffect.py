import locale
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_pandas
from scipy.constants import c, h
from scipy.optimize import root_scalar

### Подготовка ###
# В качестве десятичного разделителя использовать запятую, а не точку
locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")
# Стиль графика взять из файла с его описанием
plt.style.use("sciart.mplstyle")
# Директории
measurements_dir = Path("measurements")
figures_dir = Path("figures")
output_dir = Path(".")
# Настройка системы единиц измерения
ureg = pint.UnitRegistry(fmt_locale="ru_RU.UTF-8")
ureg.load_definitions("units.txt")
ureg.formatter.default_format = "~"  # Использовать краткий формат
pint_pandas.PintType.ureg = ureg
# Физические константы
h = h * ureg("joule * second")  # постоянная Планка
c = c * ureg("meter / second")  # скорость света в вакууме

### Чтение данных измерений из файлов ###
angle_data = pd.read_csv(
    measurements_dir / "angle-photocurrent.csv", header=[0, 1], dtype=float
).pint.quantify(level=1)
distance_data = pd.read_csv(
    measurements_dir / "distance-photocurrent.csv", header=[0, 1], dtype=float
).pint.quantify(level=1)
spectra_data = pd.read_csv(
    measurements_dir / "spectra.csv", header=[0, 1], dtype=float
).pint.quantify(level=1)

### Расчёты ###
current_cols = ["current1", "current2", "current3"]
angle_data["mean_current"] = angle_data[current_cols].mean(axis=1)
distance_data["mean_current"] = distance_data[current_cols].mean(axis=1)
spectra_data["energy"] = h * c / spectra_data["wavelength"]
spectra_data["current_AB"] = spectra_data["current"] / (
    spectra_data["A"] * spectra_data["B"]
)

### Визуализация ###
# - подготовка данных
cosine = np.cos(np.radians(angle_data["angle"].pint.magnitude))
mean_current = angle_data["mean_current"].pint.magnitude
coeffs = np.polyfit(cosine, mean_current, 1)
x = cosine.iloc[[0, -1]]
# - график фототока от угла падения
fig1, ax1 = plt.subplots(num="angle-photocurrent")
ax1.plot(
    cosine,
    mean_current,
    ls="",
    marker="o",
    c="k",
    markerfacecolor="none",
    markersize=5,
)
ax1.plot(x, np.polyval(coeffs, x), c="k")
ax1.set(xlabel=r"$\cos\mathrm{\alpha}$", ylabel=r"$\langle I_{\text{ф}} \rangle$, мкА")

# - подготовка данных
distances = distance_data["distance"].pint.to("m").pint.magnitude
distances_inv_sqr = 1 / distances**2
mean_current = distance_data["mean_current"].pint.magnitude
coeffs = np.polyfit(distances_inv_sqr, mean_current, 1)
x = [0, distances_inv_sqr.max()]
# - график фототока от обратного квадрата расстояния
fig2, ax2 = plt.subplots(num="distance-photocurrent")
ax2.plot(distances_inv_sqr, mean_current, ls="", marker="x", c="k", markersize=5)
ax2.plot(x, np.polyval(coeffs, x), c="k")
ax2.set(xlabel=r"$r^{-2}$, м$^{-2}$", ylabel=r"$\langle I_{\text{ф}} \rangle$, мкА")

# - подготовка данных
energy = spectra_data["energy"].pint.to("eV").pint.magnitude
coeffs = np.polyfit(energy, spectra_data["current_AB"].pint.magnitude, 3)
x = np.linspace(energy.min(), energy.max(), 100)
y = np.polyval(coeffs, x)
where_positive = y >= 0
x, y = x[where_positive], y[where_positive]
# - спектральная характеристика фотоэлемента
fig3, ax3 = plt.subplots(num="spectra")
ax3.plot(
    energy,
    spectra_data["current_AB"].pint.magnitude,
    ls="",
    marker="o",
    markersize=5,
    markerfacecolor="none",
    c="k",
)
ax3.plot(x, y, c="k")
# - проведение касательной
slope = np.gradient(y, x)[0]
tangent = lambda x: slope * (x - energy[0]) + y[0]
x = np.linspace(0, energy[0], 50)
y = tangent(x)
where_positive = y >= 0
x, y = x[where_positive], y[where_positive]
ax3.plot(x, y, c="gray")
ax3.set(xlabel="$E$, эВ", ylabel=r"$I_{\text{ф}}$, мкА")

# Сохранение графиков
fig1.savefig(figures_dir / "angle-photocurrent")
fig2.savefig(figures_dir / "distance-photocurrent")
fig3.savefig(figures_dir / "spectra")

### Вывод основных результатов ###
print("Результаты:")
energy_gap = root_scalar(tangent, bracket=[0, energy[0]]).root * ureg.eV
print(" - ширина запрещённой зонны:", round(energy_gap, 3))
# Сохранение в файл
with open(output_dir / "results-inner-photoeffect.txt", "w", encoding="utf-8") as f:
    f.write(f"Ширина запрещённой зоны: {energy_gap:.3f}\n")

### Показ интерактивных графиков ###
# plt.show()
