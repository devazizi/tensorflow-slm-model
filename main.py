from fastapi import FastAPI, Depends
from app.router import router as api_router
from infra.clickhouse import get_clickhouse, create_tables
from model.train import Model

app = FastAPI()

#
@app.on_event("startup")
def startup():
    # create_tables(get_clickhouse())
    Model().create_model().train_model()


app.include_router(router=api_router, prefix='/api', dependencies=[Depends(get_clickhouse)])
