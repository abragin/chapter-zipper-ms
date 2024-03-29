# chapter-zipper-ms
A microservice used by https://github.com/abragin/book-zipper to match a single chapter.

# Preload transformer
gunicorn may fail to start because loading transformer model takes too long. To fix it run:

```
python matching_service/next_match_nn.py
```

# Usage example
```
$ gunicorn matching_service.app
...
$ curl -X POST http://localhost:8000/match_chapter -H 'Content-Type: application/json' -d "@sample_request.json"
```

Cuda is used if available, but can be turned off by setting `FORCE_CPU` environment variable. If used without cuda huge timeout should be set for gunicorn (e.g. `--timeout 240`)
