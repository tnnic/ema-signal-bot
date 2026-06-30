# System Patterns

## Архитектура
- `main.py` — единый entrypoint: загрузка env, запросы к Bybit, расчёт сигнала, отправка в Telegram.

## Паттерны расчёта
- Свечи берутся через Bybit v5 `market/kline` и сортируются по времени по возрастанию.
- Сигнал определяется на предпоследней свече (`last_closed = len(df) - 2`).
- Структура в `check_signal()` реализует упрощённую LuxAlgo-логику pivot-leg + crossover/crossunder по close относительно pivot уровней.

