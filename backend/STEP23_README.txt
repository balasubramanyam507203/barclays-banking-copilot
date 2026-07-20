STEP 23 - REDIS RAG RESPONSE CACHE

1. Extract this bundle from the project root.
2. Add step23_env_additions.txt to backend/.env once.
3. Install dependencies with pip install -r requirements.txt.
4. Confirm Redis is running: redis-cli ping -> PONG.
5. Run python -m pytest tests/test_cache_service.py -v.
6. Run python -m pytest.
7. Start Uvicorn and ask the same question twice using the same role,
   region, clearance, index, and model settings.
8. The first request should log cache_hit=false and call the model.
9. The second request should log rag_cache_hit/cache_hit=true and return
   model_called=false with no new model token usage.

The cache is permission-aware and fail-open. It hashes the question and
access scope instead of storing the raw question in the Redis key.
