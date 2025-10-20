from fastapi import FastAPI
from pydantic import BaseModel

class ExampleRequest(BaseModel):
    string: str
    numeric: int

app = FastAPI()

@app.get('/healthz')
def healthz():
    return {"status": "ok"}

@app.post('/testpost')
def testpost(request: ExampleRequest):
    return {"string": request.string, "numeric": request.numeric}