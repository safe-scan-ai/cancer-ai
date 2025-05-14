from pygelf import GelfTcpHandler
import logging
import time

class ApiKeyGelfTcpHandler(GelfTcpHandler):
    def __init__(self, host, port, api_key, **kwargs):
        super().__init__(host, port, **kwargs)
        self.api_key = api_key

    def emit(self, record):
        if not hasattr(record, '_api_key'):
            record._api_key = self.api_key
        super().emit(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

api_key = "4222b331-e9b0-47df-ba9b-ddb3cbd42dea"
logger.addHandler(ApiKeyGelfTcpHandler(
    host='185.252.234.155',
    port=12299,
    api_key=api_key, 
    include_extra_fields=True 
))

# Wysyłanie przykładowych logów
for x in range(1000):
    time.sleep(1)
    for y in range(5):
        logger.info(f"Test log {x}-{y}")


# curl -X POST http://185.252.234.155:12299/logs \
#      -H "X-API-Key: 4222b331-e9b0-47df-ba9b-ddb3cbd42dea" \
#      -H "Content-Type: application/json" \
#      -d '{
#             "message": "Test log message",
#             "level": "info",
#             "timestamp": "'$(date --iso-8601=seconds)'"
#           }'

