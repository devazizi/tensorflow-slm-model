from fastapi import APIRouter, Depends
from .schema import GreetingRequest
from model.train import TrainedModel
from typing import Annotated
from infra.clickhouse import get_clickhouse

router = APIRouter()


@router.post("/talk")
def talk_route(greeting_request: GreetingRequest, db: Annotated[None, Depends(get_clickhouse)]):

    return TrainedModel().load_model().generate_greeting(greeting_request.message)