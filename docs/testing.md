# Testing

## Unit Tests

No server needed:

```bash
PYTHONPATH=src python -m pytest tests/test_translator.py tests/test_tool_use.py tests/test_clients.py tests/test_usage.py tests/test_admin.py -v
```

## End-to-End & SDK Tests

Requires a running Prism server:

```bash
# Terminal 1 — start the server
PYTHONPATH=src python -m prism

# Terminal 2 — run tests
PYTHONPATH=src \
  PRISM_TEST_URL=http://127.0.0.1:9877 \
  PRISM_TEST_KEY=your-client-key \
  PRISM_TEST_MODEL=your-model-name \
  python -m pytest tests/ -v -s
```
