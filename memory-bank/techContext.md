# Tech Context

## Стек
- Python
- `aiohttp` для запросов к Bybit
- `python-telegram-bot` для отправки в Telegram
- `pandas`, `numpy` для обработки данных
- `mplfinance` для графиков

## Конфиги (env)
- `TIMEFRAME`: таймфрейм свечей (нормализуется: `D/W/M` и `1D/1W/1M`)
- `BYBIT_CATEGORY`: по умолчанию `spot`
- `KLINE_LIMIT`: лимит свечей для расчёта
- `VOLUME_WINDOW`: окно для среднего объёма (в барах текущего таймфрейма)

