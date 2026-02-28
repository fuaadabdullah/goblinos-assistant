# pyright: reportMissingImports=false
#!/usr/bin/env python3

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Test API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
