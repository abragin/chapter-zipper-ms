import falcon
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from .chapter_zipper import match_chapter


match_request_schema = {
    "type": "object",
    "properties": {
        "source_ps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string"
                }
            },
        "target_ps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string"
                }
            },
    },
    "required": ["source_ps", "target_ps"]
}

class MatcherResource:
    def on_post(self, req, resp):
        req_contents = req.get_media()
        try:
            validate(req_contents, match_request_schema)
        except ValidationError as ve:
            desc = f"Json schema validation failed: {ve.message}"
            raise falcon.HTTPUnprocessableEntity(
                    description = desc
                    )
        else:
            source_ps = req_contents["source_ps"]
            target_ps = req_contents["target_ps"]
            resp.media = match_chapter(source_ps, target_ps)

app = application = falcon.App()
app.add_route('/match_chapter', MatcherResource())
