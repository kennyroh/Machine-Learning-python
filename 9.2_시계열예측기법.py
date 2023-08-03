import numpy as np
from matplotlib import pyplot as plt


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, .05)
amplitude = 40
slope = 0.05
noise_level = 5

# add series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# add noise
series += noise(time, noise_level, seed=42)

split_time = 1000
naive_forecast = series[split_time - 1:-1]


def moving_average_forecast(series, window_size=10):
    forecast = np.cumsum(series, dtype=float)
    forecast[window_size:] = forecast[window_size:] - forecast[:-window_size]
    return forecast[window_size - 1:] / window_size


moving_avg = moving_average_forecast(series, window_size=10)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time, series)
plot_series(time[-30:], moving_avg)


diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

diff_moving_avg = moving_average_forecast(diff_series, window_size=50)[split_time - 365 - 30:]

diff_moving_avg_plus_smooth_past = \
    moving_average_forecast(diff_series, window_size=50)[split_time - 365 - 30:] + \
    diff_moving_avg


