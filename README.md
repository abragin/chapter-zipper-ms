# chapter-zipper-ms
A microservice used by https://github.com/abragin/book-zipper to match a single chapter.

# Usage example
```
$ gunicorn matching_service.app
...
$ curl -X POST http://localhost:8000/match_chapter -H 'Content-Type: application/json' -d "@sample_request.json"
```
