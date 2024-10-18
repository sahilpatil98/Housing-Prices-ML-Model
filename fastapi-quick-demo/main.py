from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/square")
async def square(num: int):
    result = num ** 2
    return {"square": result}


