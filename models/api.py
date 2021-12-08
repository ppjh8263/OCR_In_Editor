from fastapi import FastAPI
import uvicorn
from server.modules.inference import load_model_at_run
from server.modules.docs import description
from server.api import *

# import io

app = FastAPI(description=description)

#메인페이지 실행할때 모델로드
@app.get("/")
def read_root():
    load_model_at_run()
    return "Boost Camp AI tech CV7's API"


if __name__ == '__main__':
    uvicorn.run('api:app', port=6006, host='0.0.0.0', reload=True,)