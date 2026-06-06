from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.services.model_registry import available_model_ids

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

@app.on_event("startup")
async def log_runtime_mode():
    mode = "mock" if settings.USE_MOCK_INFERENCE else "real"
    print(
        f"[SkinScan] startup mode={mode} device={settings.MODEL_DEVICE} "
        f"models_available={available_model_ids()}",
        flush=True,
    )

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối API router
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}", "docs": "/docs"}
