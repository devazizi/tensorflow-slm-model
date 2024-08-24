from pydantic import BaseModel


class GreetingRequest(BaseModel):
    user_id: int
    message: str
