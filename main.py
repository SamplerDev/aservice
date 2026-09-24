import os
import io
from enum import Enum
from typing import Optional, List
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv





load_dotenv()

app = FastAPI(title="AI Agentic Microservice (Gemini)")

# Inicializar cliente oficial de Google GenAI
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------------------------------------------------------
# ESQUEMAS PYDANTIC (Structured Outputs)
# ------------------------------------------------------------------

class OfertaViaje(BaseModel):
    destino: str = Field(description="Ciudad, región o país principal del viaje")
    fecha_salida: str = Field(description="Fecha en formato YYYY-MM-DD o aproximada")
    descripcion: str = Field(description="Resumen de inclusiones, estadía, transporte y precio")
    contacto: Optional[str] = Field(None, description="Teléfono o medio de contacto extraído")
    cupos: Optional[int] = Field(1, description="Cantidad estimada de cupos o lugares disponibles mencionados")

class ExtractRequest(BaseModel):
    image_url: str

class ChatMessage(BaseModel):
    role: str  # "user" o "model"
    content: str

class ChatRequest(BaseModel):
    user_message: str
    travel_catalog: List[dict]  # Catálogo filtrado desde Node.js (solo activos y con cupo)
    history: Optional[List[ChatMessage]] = []

class ConfirmAction(str, Enum):
    APROBAR = "APROBAR"
    RECHAZAR = "RECHAZAR"
    DESCONOCIDO = "DESCONOCIDO"

class ConfirmResult(BaseModel):
    accion: ConfirmAction
    notas: Optional[str] = None

class ConfirmRequest(BaseModel):
    admin_message: str

# ------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------------

async def verify_internal_key(api_key: str = Security(api_key_header)):
    """Verifica que la petición venga del servidor Node.js autorizado."""
    if not INTERNAL_API_KEY or api_key != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acceso no autorizado: X-API-Key inválida o ausente"
        )


def optimizar_flyer_para_vision(imagen_bytes: bytes, max_size: int = 1024) -> bytes:
    img = Image.open(io.BytesIO(imagen_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "Python AI Microservice (Gemini)"}

# 1. Extracción Multimodal desde Flyer
@app.post("/agent/extract", response_model=OfertaViaje, dependencies=[Depends(verify_internal_key)])
async def extract_travel_info(payload: ExtractRequest):
    try:
        img_response = requests.get(payload.image_url)
        if img_response.status_code != 200:
            raise HTTPException(status_code=400, detail="No se pudo descargar la imagen")

        imagen_optimizada = optimizar_flyer_para_vision(img_response.content, max_size=1024)

        image_part = types.Part.from_bytes(
            data=imagen_optimizada, 
            mime_type="image/jpeg"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Analiza este flyer de viaje y extrae la información requerida de forma estructurada.", image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=OfertaViaje,
            ),
        )
        return OfertaViaje.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en extracción Gemini: {str(e)}")

# 2. Asistente Conversacional RAG
@app.post("/agent/chat", dependencies=[Depends(verify_internal_key)])
async def chat_agent(payload: ChatRequest):
    try:
        system_instruction = f"""
        Eres un asesor de viajes por WhatsApp atento y comercial.
        
        Tus reglas:
        1. Responde ÚNICAMENTE basándote en este catálogo de viajes disponibles que TIENEN STOCK:
        {payload.travel_catalog}
        
        2. Menciona claramente el destino, fecha, detalles y si quedan POCOS cupos (menciona los cupos si son < 3).
        3. Si el cliente pregunta por un destino o fecha que NO figura en este catálogo, infórmale amablemente que las plazas para ese viaje están agotadas o que no hay salidas disponibles por el momento, e insítalo a consultar por las otras opciones activas.
        4. Mantén respuestas concisas, amables y adecuadas para un chat de WhatsApp.
        """

        contents = []
        for msg in payload.history:
            role = "user" if msg.role.lower() in ["user", "human"] else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.user_message)]))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Chat Gemini: {str(e)}")

# 3. Parser de Confirmación Humana
@app.post("/agent/confirm", response_model=ConfirmResult, dependencies=[Depends(verify_internal_key)])
async def parse_admin_confirmation(payload: ConfirmRequest):
    try:
        prompt = f"Determina si el siguiente mensaje del administrador autoriza la publicación ('APROBAR') o la rechaza ('RECHAZAR'):\n'{payload.admin_message}'"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ConfirmResult,
            )
        )
        return ConfirmResult.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Confirmación Gemini: {str(e)}")