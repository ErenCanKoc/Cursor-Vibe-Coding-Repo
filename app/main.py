"""FastAPI sunucusu: GEO fan-out query üretici hizmeti."""

from fastapi import FastAPI, HTTPException

from fanout_ai import FanOutRequest, FanOutResult, generate_geo_content_from_request

app = FastAPI(
    title="Fan-Out GEO API",
    version="0.1.0",
    summary="GEO uyumlu fan-out snippet'lerini OpenAI structured outputs ile üretir.",
)


@app.get("/health", summary="Servis durum kontrolü")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/fanout",
    response_model=FanOutResult,
    summary="GEO fan-out blokları üret",
    tags=["fanout"],
)
def create_fanout(payload: FanOutRequest) -> FanOutResult:
    """Giriş metni ve anahtar kelime için fan-out bloklarını döndürür."""

    result = generate_geo_content_from_request(payload)
    if result is None:
        raise HTTPException(status_code=502, detail="Fan-out üretimi başarısız oldu.")

    return result
