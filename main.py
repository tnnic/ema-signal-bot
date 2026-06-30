import asyncio
import os
from io import BytesIO

import aiohttp
import mplfinance as mpf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Bot
from telegram.request import HTTPXRequest

load_dotenv(override=True)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RAW_TIMEFRAME = os.getenv("TIMEFRAME", "D")  # по умолчанию дневка
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "5000"))  # минимальный объём для фильтра
# как в TradingView (Bybit Spot)
CATEGORY = os.getenv("BYBIT_CATEGORY", "spot")
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "1000"))
PLOT_BARS = int(os.getenv("PLOT_BARS", "50"))
OHLCV_RETRIES = int(os.getenv("OHLCV_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "1.0"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
EMA_PERIOD = int(os.getenv("EMA_PERIOD", "10"))
SMA_PERIOD = int(os.getenv("SMA_PERIOD", "200"))


def normalize_timeframe(raw: str) -> str:
    """
    Нормализует ввод таймфрейма для Bybit kline interval.
    Поддерживает: D/W/M, а также 1D/1W/1M.
    """
    if raw is None:
        return "D"
    tf = str(raw).strip().upper()
    if tf in {"1D", "D"}:
        return "D"
    if tf in {"1W", "W"}:
        return "W"
    if tf in {"1M", "M"}:
        return "M"
    return tf


TIMEFRAME = normalize_timeframe(RAW_TIMEFRAME)
TIMEFRAME_LABELS = {"D": "дней", "W": "недель", "M": "месяцев"}
TIMEFRAME_UNIT = TIMEFRAME_LABELS.get(TIMEFRAME, "баров")

if not TOKEN or not CHAT_ID:
    raise ValueError("TELEGRAM_TOKEN или TELEGRAM_CHAT_ID не заданы")

telegram_request = HTTPXRequest(connection_pool_size=32, pool_timeout=30.0)
bot = Bot(token=TOKEN, request=telegram_request)

print(f"TIMEFRAME={TIMEFRAME} CATEGORY={CATEGORY}")

signals_found = 0
signals_sent = 0

# === График matplotlib ===


def make_plot(df, symbol, signal, signal_idx=None):
    """Строит свечной график с линиями EMA и маркером сигнальной свечи (как в TradingView)."""
    df_plot = df.copy()
    df_plot = df_plot.set_index("timestamp")
    df_plot = df_plot.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )

    close = df_plot["close"]
    indicators = [
        (close.ewm(span=EMA_PERIOD, adjust=False).mean(), "blue"),
        (close.rolling(window=SMA_PERIOD).mean(), "orange"),
    ]

    # Для картинки берём только последние бары, индикаторы считаем по полной истории.
    df_plot = df_plot.tail(PLOT_BARS)

    apds = []
    for series, color in indicators:
        plot_series = series.tail(PLOT_BARS)
        if plot_series.notna().any():
            apds.append(mpf.make_addplot(plot_series, color=color, width=1.2))
    # Маркер только на сигнальной свече (чтобы совпадало с TradingView)
    if signal_idx is not None:
        # signal_idx приходит в индексах исходного df; после tail() пересчитываем.
        start_idx = max(0, len(df) - len(df_plot))
        local_idx = signal_idx - start_idx
    else:
        local_idx = None

    if local_idx is not None and 0 <= local_idx < len(df_plot):
        marker = np.full(len(df_plot), np.nan)
        if signal == "BUY":
            marker[local_idx] = df_plot["low"].iloc[local_idx]
            apds.append(
                mpf.make_addplot(
                    marker, type="scatter", marker="^", markersize=100, color="green"
                )
            )
        elif signal == "SELL":
            marker[local_idx] = df_plot["high"].iloc[local_idx]
            apds.append(
                mpf.make_addplot(
                    marker, type="scatter", marker="v", markersize=100, color="red"
                )
            )

    buf = BytesIO()
    mpf.plot(
        df_plot,
        type="candle",
        addplot=apds,
        style="charles",
        title=f"{symbol} – {signal} Signal",
        ylabel="Price",
        volume=True,
        mav=(),
        savefig=buf,
    )
    buf.seek(0)
    return buf


# === Получение OHLCV ===


async def get_ohlcv(session, symbol):
    """Получает OHLCV-данные по символу."""
    url = f"https://api.bytick.com/v5/market/kline?category={CATEGORY}&symbol={symbol}&interval={TIMEFRAME}&limit={KLINE_LIMIT}"
    for attempt in range(OHLCV_RETRIES):
        try:
            async with session.get(url) as resp:
                data = await resp.json()
                if data["retCode"] != 0:
                    print(f"Ошибка API для {symbol}: {data}")
                    return None
                df = pd.DataFrame(
                    data["result"]["list"],
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["hl2"] = (df["high"] + df["low"]) / 2
                # Bybit обычно возвращает свечи от новых к старым — нормализуем порядок.
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < OHLCV_RETRIES - 1:
                await asyncio.sleep(OHLCV_RETRY_DELAY * (attempt + 1))
                continue
            print(f"Ошибка при получении OHLCV для {symbol}: {e}")
            return None
        except Exception as e:
            print(f"Ошибка при получении OHLCV для {symbol}: {e}")
            return None


# === Проверка сигнала ===


def check_signal(df):
    """
    Пересечение EMA и SMA на последней закрытой свече (как ta.crossover / ta.crossunder в Pine).
    """
    if len(df) < SMA_PERIOD + 1:
        return None, None

    close = df["close"].astype(float)
    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    sma = close.rolling(window=SMA_PERIOD).mean()

    last_closed = len(df) - 2
    prev = last_closed - 1

    ema_curr, ema_prev = ema.iloc[last_closed], ema.iloc[prev]
    sma_curr, sma_prev = sma.iloc[last_closed], sma.iloc[prev]

    if pd.isna(ema_curr) or pd.isna(sma_curr) or pd.isna(ema_prev) or pd.isna(sma_prev):
        return None, None

    cross_up = ema_prev <= sma_prev and ema_curr > sma_curr
    cross_down = ema_prev >= sma_prev and ema_curr < sma_curr

    if cross_up:
        return "BUY", last_closed
    if cross_down:
        return "SELL", last_closed
    return None, None


# === Отправка сигнала ===


async def send_signal(symbol, signal, df, signal_idx):
    """Отправляет сигнал в Telegram."""
    img = make_plot(df, symbol, signal, signal_idx)
    text = f"{signal} signal for {symbol} on timeframe {TIMEFRAME}"
    try:
        print(f"Пытаюсь отправить сигнал: {symbol}, {signal}")
        await bot.send_photo(chat_id=CHAT_ID, photo=img, caption=text)
    except Exception as e:
        print(f"Ошибка при отправке сигнала: {e}")


# === Проверка одной пары ===
# ограничиваем параллелизм, чтобы не забить Telegram/Bybit
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def process_symbol(session, symbol):
    """Обрабатывает один символ: фильтр по объёму, проверка сигнала, отправка."""
    global signals_found, signals_sent
    try:
        async with semaphore:
            df = await get_ohlcv(session, symbol)
        if df is None:
            print(f"Нет данных для {symbol}")
            return

        # Фильтр по объёму за последние N баров текущего таймфрейма
        window = int(os.getenv("VOLUME_WINDOW", "5"))
        # volume_usdt = df["volume"].astype(float) * df["close"].astype(float)
        volume_usdt = df["turnover"].astype(float)

        # 1. Проверяем медианный объем (устойчив к аномальным всплескам)
        # Если в окне есть "мертвые" свечи, медиана это сразу покажет.
        recent_volumes = volume_usdt.iloc[-window:]
        median_volume_usdt = recent_volumes.median()

        if median_volume_usdt < MIN_VOLUME:
            print(
                f"Медианный объём в USDT за {window} {TIMEFRAME_UNIT} ниже минимума для {symbol}: {median_volume_usdt}"
            )
            return

        # 2. Дополнительная защита: полностью исключаем свечи с нулевым объемом
        # Если хотя бы одна свеча в окне имеет 0 объем, монета может быть неликвидной
        # или находиться на технической паузе.
        if (recent_volumes == 0).any():
            print(f"Обнаружена свеча с нулевым объемом для {symbol}, пропускаем")
            return

        signal, signal_idx = check_signal(df)
        if signal:
            signals_found += 1
            print(f"Сигнал пересечения EMA/SMA для {symbol}: {signal}")
            await send_signal(symbol, signal, df, signal_idx)
            signals_sent += 1
    except Exception as e:
        print(f"Ошибка при обработке {symbol}: {e}")


# === Получение всех символов ===


async def get_all_symbols(session):
    """Получает список всех торговых пар с Bybit."""
    url = f"https://api.bytick.com/v5/market/tickers?category={CATEGORY}"
    try:
        async with session.get(url) as resp:
            data = await resp.json()
            if data["retCode"] != 0:
                print(f"Ошибка получения списка пар: {data}")
                return []
            symbols = [item["symbol"] for item in data["result"]["list"]]
            # Оставляем только обычные спотовые пары *USDT (без PERP/датированных инструментов)
            symbols = [
                s
                for s in symbols
                if s.endswith("USDT") and "-" not in s and "PERP" not in s
            ]
            return symbols
    except Exception as e:
        print(f"Ошибка при получении списка пар: {e}")
        return []


# === Пример асинхронного запуска для нескольких символов ===


async def main():
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT
    )
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        symbols = await get_all_symbols(session)
        print(f"Найдено {len(symbols)} пар")
        await asyncio.gather(
            *(process_symbol(session, s) for s in symbols),
            return_exceptions=True,
        )
    print(
        f"Итого найдено сигналов: {signals_found}, отправлено: {signals_sent}"
    )


if __name__ == "__main__":
    asyncio.run(main())
