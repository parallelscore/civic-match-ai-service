from pydantic import BaseModel


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class CamelModel(BaseModel):
    """Base model that converts snake_case to camelCase automatically."""

    class Config:
        alias_generator = to_camel
        populate_by_name = True
