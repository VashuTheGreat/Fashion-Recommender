from fastapi import FastAPI,File,UploadFile,Request
from temp import give_path_get_link
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

app = FastAPI()


@app.get("/",response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "data": "Hello, FastAPI!"})


@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    content = await file.read()
    with open("test.jpg", "wb") as f:
        f.write(content)
    result,json_data=give_path_get_link("test.jpg")
    return {"file": file, "result": result,"json_data": json_data}
